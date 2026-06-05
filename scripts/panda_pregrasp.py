"""PREGRASP controller validation sandbox.

The whole controller (cost terms + weights AND the MPPI/sim knobs) is configured
in ``sbmpc/ocp_configs/pregrasp.yaml`` — edit that, then run this. It:

  1. builds the controller from the yaml,
  2. runs the closed loop and shows the robot live in MuJoCo (unless --headless),
  3. asserts the three validation criteria and prints PASS/FAIL:
       * end-effector error to the pregrasp goal (as small as possible),
       * actuation within the robot torque/velocity limits,
       * feedback-gain stability over time.

Usage::

    python scripts/panda_pregrasp.py                 # viewer + validation
    python scripts/panda_pregrasp.py --headless      # metrics only
    python scripts/panda_pregrasp.py --ocp pregrasp  # pick a different yaml
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.controller.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.ocp import load_ocp_config
from sbmpc.simulation import build_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ocp", default="pregrasp", help="OCP yaml name or path (cost + mpc + sim)")
    p.add_argument("--headless", action="store_true", help="run without the MuJoCo viewer")
    p.add_argument("--success-tol", type=float, default=0.01, help="EE-error success threshold [m]")
    p.add_argument("--limit-fraction", type=float, default=0.9,
                   help="fail if a joint exceeds this fraction of its torque/velocity limit")
    return p.parse_args()


def _row(name, vec, limit=None):
    cells = " ".join(f"{v:7.2f}" for v in vec)
    extra = "" if limit is None else "   %lim: " + " ".join(f"{v / l * 100:5.0f}" for v, l in zip(vec, limit))
    return f"  {name:14s} {cells}{extra}"


def validate(planner, q, v, tau, gain_norm, *, success_tol, limit_fraction) -> bool:
    tau_lim = np.asarray(planner.torque_limits, dtype=float)
    vel_lim = np.asarray(planner.velocity_limits, dtype=float)
    ee_err = np.array([
        float(np.linalg.norm(np.asarray(planner.ee_position(jnp.asarray(qi))) - np.asarray(planner.goal_pos)))
        for qi in q
    ])
    peak_tau = np.max(np.abs(tau), axis=0) if tau.size else np.zeros_like(tau_lim)
    peak_vel = np.max(np.abs(v), axis=0) if v.size else np.zeros_like(vel_lim)

    print("\n" + "=" * 66 + "\nPREGRASP CONTROLLER VALIDATION\n" + "=" * 66)

    print("\n-- 1. TASK SUCCESS (end-effector error) --")
    print(f"  initial={ee_err[0]:.4f} m   min={np.nanmin(ee_err):.4f} m   final={ee_err[-1]:.4f} m")
    reached = bool(np.isfinite(ee_err[-1]) and ee_err[-1] <= success_tol)
    print(f"  reached (<{success_tol*1000:.0f} mm): {'YES' if reached else 'NO'}")

    print("\n-- 2. WITHIN ROBOT LIMITS --")
    print("  joint           j1      j2      j3      j4      j5      j6      j7")
    print(_row("peak |tau| Nm", peak_tau, tau_lim))
    print(_row("peak |v| rad/s", peak_vel, vel_lim))
    worst_tau = float(np.nanmax(peak_tau / tau_lim))
    worst_vel = float(np.nanmax(peak_vel / vel_lim))
    within = np.isfinite(worst_tau) and np.isfinite(worst_vel) and worst_tau <= limit_fraction and worst_vel <= limit_fraction
    print(f"  worst torque={worst_tau*100:.0f}%   worst velocity={worst_vel*100:.0f}%   "
          f"-> {'OK' if within else f'EXCEEDS {limit_fraction*100:.0f}%'}")

    print("\n-- 3. GAIN STABILITY (|K| over time) --")
    g = np.asarray(gain_norm)
    gains_ok = True
    if g.size and np.any(g > 0):
        tail = g[len(g) // 2:]
        jump = float(np.max(np.abs(np.diff(g)))) if g.size > 1 else 0.0
        gains_ok = bool(np.all(np.isfinite(g)) and tail.std() <= 0.25 * (abs(tail.mean()) + 1e-9))
        print(f"  final={g[-1]:.3f}  max={g.max():.3f}  mean={g.mean():.3f}  "
              f"tail_std={tail.std():.3f}  max_step_jump={jump:.3f}  -> {'stable' if gains_ok else 'UNSTABLE'}")
    else:
        print("  (no gains — set mpc.gains: true in the yaml to assess gain stability)")

    ok = reached and within and gains_ok
    print("\n" + "-" * 66)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}  (reached={reached}, within_limits={within}, gains_ok={gains_ok})")
    print("-" * 66)
    return ok


def main() -> None:
    args = parse_args()
    ocp = load_ocp_config(args.ocp)
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner, ocp_config=ocp)
    config = make_panda_pregrasp_config(planner, visualize=not args.headless, ocp=ocp)

    print(f"JAX backend: {jax.default_backend()}  |  ocp={ocp.name}")
    print(f"mpc: horizon={ocp.mpc.horizon} samples={ocp.mpc.num_samples} "
          f"lambda={ocp.mpc.lambda_mpc} std_dev_scale={ocp.mpc.std_dev_scale} "
          f"init_guess={ocp.mpc.initial_guess} gains={ocp.mpc.gains}")
    print(f"goal_pos = {np.asarray(planner.goal_pos)}")
    if config.MPC.gains:
        print("note: gains=true compiles the exact-gain backprop on the first step "
              "(can take ~1 min — not a hang).")

    sim = build_all(
        config, objective, objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics, obstacles=False,
    )
    sim.planner = planner
    gain_norm: list[float] = []

    def post_update(sim) -> None:
        gain_norm.append(float(jnp.linalg.norm(sim.controller.gains[0])))
        if sim.iter % 25 == 0:
            q = sim.current_state_vec()[: planner.nq]
            err = float(jnp.linalg.norm(planner.ee_position(q) - planner.goal_pos))
            print(f"  step {sim.iter:04d}  ee_err={err:.4f} m  |K|={gain_norm[-1]:.3f}", flush=True)

    sim.post_update = post_update
    print("running closed-loop sim (close the viewer window to stop early) ...", flush=True)
    sim.simulate()

    n = sim.iter
    q = np.asarray(sim.state_traj[: n + 1, : planner.nq])
    v = np.asarray(sim.state_traj[: n + 1, planner.nq : planner.nq + planner.nv])
    tau = np.asarray(sim.input_traj[:n, :])
    validate(planner, q, v, tau, gain_norm,
             success_tol=args.success_tol, limit_fraction=args.limit_fraction)


if __name__ == "__main__":
    main()
