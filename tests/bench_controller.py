"""Benchmark and stability diagnostic for the Panda MPPI controller.

Usage:
  # Visual test (opens MuJoCo viewer, default params):
  pixi run python tests/bench_controller.py --visual

  # Headless timing sweep (find max config at 50 Hz):
  pixi run python tests/bench_controller.py --sweep

  # Gain stability check (20 steps with gains enabled):
  pixi run python tests/bench_controller.py --gains

  # Single-config headless benchmark:
  pixi run python tests/bench_controller.py --horizon 8 --samples 64 --steps 30
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all

TARGET_HZ = 50.0
TARGET_MS = 1000.0 / TARGET_HZ


def _run_headless(planner, objective, config, n_steps, label=""):
    """Run n_steps headlessly and return timing stats + gain stability info."""
    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    sim.planner = planner

    times_ms = []
    gain_norms = []
    gain_finite = []

    # Warm-up: one step to trigger JIT compilation
    sim.step()

    for _ in range(n_steps):
        sim.step()
        times_ms.append(sim.last_command_time_ms)
        if config.MPC.gains:
            K = sim.controller.gains[0]
            norm = float(jnp.linalg.norm(K))
            gain_norms.append(norm)
            gain_finite.append(bool(jnp.all(jnp.isfinite(K))))

    times = np.array(times_ms)
    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    ok = p95_ms < TARGET_MS
    marker = "✓" if ok else "✗"

    row = (
        f"{marker} {label:40s}  "
        f"mean={mean_ms:6.1f}ms  p95={p95_ms:6.1f}ms  "
        f"[{'PASS' if ok else 'FAIL'} @{TARGET_HZ:.0f}Hz]"
    )

    if config.MPC.gains and gain_norms:
        gn = np.array(gain_norms)
        all_finite = all(gain_finite)
        row += (
            f"  |K| mean={np.mean(gn):.2f} std={np.std(gn):.2f} "
            f"{'finite=OK' if all_finite else 'WARN:non-finite'}"
        )

    return row, mean_ms, p95_ms, ok


def run_visual(planner, objective, config):
    """Open MuJoCo viewer and run until window closes."""
    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    sim.planner = planner

    def post_update(s):
        state = s.current_state_vec()
        q = state[: planner.nq]
        ee_pos = planner.ee_position(q)
        err = float(jnp.linalg.norm(ee_pos - planner.goal_pos))
        gain_norm = float(jnp.linalg.norm(s.controller.gains[0])) if config.MPC.gains else 0.0
        print(
            f"iter={s.iter:04d} "
            f"ee_pos={np.round(np.asarray(ee_pos), 3)} "
            f"err={err:.3f}m "
            f"plan={s.last_command_time_ms:.1f}ms"
            + (f" |K|={gain_norm:.3f}" if config.MPC.gains else "")
        )

    sim.post_update = post_update
    print(f"\nJAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"horizon={config.MPC.horizon}  samples={config.MPC.num_parallel_computations}  "
          f"control_points={config.MPC.num_control_points}  gains={config.MPC.gains}")
    sim.simulate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--sweep", action="store_true", help="Sweep horizon × samples")
    parser.add_argument("--gains", action="store_true", help="Enable gain computation")
    parser.add_argument("--steps", type=int, default=30, help="Steps per benchmark run")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--control-points", type=int, default=4)
    args = parser.parse_args()

    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)

    if args.visual:
        config = make_panda_pregrasp_config(planner, visualize=True, gains=args.gains)
        run_visual(planner, objective, config)
        return

    if args.sweep:
        print(f"\n=== MPPI timing sweep (target: p95 < {TARGET_MS:.0f}ms = {TARGET_HZ:.0f}Hz) ===")
        print(f"JAX backend: {jax.default_backend()}")
        print(f"Steps per config (post warm-up): {args.steps}\n")

        horizons = [4, 8, 12, 16]
        samples_list = [32, 256, 512, 1024]
        control_points = 4

        for h in horizons:
            for s in samples_list:
                config = make_panda_pregrasp_config(planner, visualize=False, gains=args.gains)
                config.MPC.horizon = h
                config.MPC.num_parallel_computations = s
                config.MPC.num_control_points = control_points
                config.MPC.initial_guess = planner.nominal_torque_sequence(h, config.MPC.dt)
                label = f"horizon={h:2d} samples={s:3d} cp={control_points}"
                row, *_ = _run_headless(planner, objective, config, args.steps, label)
                print(row)
        return

    # Single-config benchmark
    config = make_panda_pregrasp_config(planner, visualize=False, gains=args.gains)
    if args.horizon is not None:
        config.MPC.horizon = args.horizon
    if args.samples is not None:
        config.MPC.num_parallel_computations = args.samples
    if args.control_points is not None:
        config.MPC.num_control_points = args.control_points
    if any(v is not None for v in (args.horizon, args.samples, args.control_points)):
        config.MPC.initial_guess = planner.nominal_torque_sequence(
            config.MPC.horizon, config.MPC.dt
        )

    label = (f"horizon={config.MPC.horizon} samples={config.MPC.num_parallel_computations} "
             f"cp={config.MPC.num_control_points} gains={config.MPC.gains}")
    print(f"\nJAX backend: {jax.default_backend()}")
    row, mean_ms, p95_ms, ok = _run_headless(planner, objective, config, args.steps, label)
    print(row)


if __name__ == "__main__":
    main()
