"""Validate the PREGRASP controller in simulation — metrics, plots, optional video.

Runs the closed-loop MPPI controller on the Panda pregrasp task using the existing
sim infrastructure (``build_all`` -> ``Simulation``): warm-start once from the
nominal seed, then roll the optimizer forward (MPC), re-planning from the true
state every step. Reports:

  * task success: end-effector position error to the goal (final + min);
  * actuation within limits: per-joint peak |torque| and |velocity| vs the FR3
    limits (% of limit, PASS/FAIL);
  * gain stability: feedback-gain norm over time (only with --gains).

Visualization here is **headless-safe**: it always saves metric plots, and can
render an offscreen MP4/GIF of the motion (--video). The live MuJoCo viewer
(--viewer) needs a real display/GL and will fail on a headless box.

Examples::

    # fast default: feedforward, metrics + plots + offscreen video
    python examples/validate_pregrasp.py
    # include feedback gains (slower: compiles the exact-gain backprop once)
    python examples/validate_pregrasp.py --gains
    # live viewer (only on a machine with a display)
    python examples/validate_pregrasp.py --viewer
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gains", action="store_true",
                   help="compute feedback gains + assess gain stability (slower: one-time JIT compile)")
    p.add_argument("--viewer", action="store_true",
                   help="open the live MuJoCo viewer (needs a real display/GL; fails headless)")
    p.add_argument("--iterations", type=int, default=250, help="sim iterations")
    p.add_argument("--horizon", type=int, default=None, help="override MPC horizon")
    p.add_argument("--samples", type=int, default=None, help="override MPPI sample count")
    p.add_argument("--success-tol", type=float, default=0.01, help="EE-error success threshold [m]")
    p.add_argument("--limit-fraction", type=float, default=0.9,
                   help="fail if a joint exceeds this fraction of its torque/velocity limit")
    p.add_argument("--plot-path", default="validate_pregrasp.png", help="metric plots ('' to skip)")
    p.add_argument("--video-path", default="validate_pregrasp.mp4",
                   help="offscreen motion video ('' to skip)")
    return p.parse_args()


def _fmt_row(name, vec, limit=None):
    cells = " ".join(f"{v:7.2f}" for v in vec)
    extra = ""
    if limit is not None:
        extra = "   %lim: " + " ".join(f"{v / l * 100:5.0f}" for v, l in zip(vec, limit))
    return f"  {name:14s} {cells}{extra}"


def report(*, ee_err, v, tau, gain_norm, tau_lim, vel_lim, success_tol, limit_fraction):
    ee_err = np.asarray(ee_err)
    v = np.abs(np.asarray(v))
    tau = np.abs(np.asarray(tau))
    peak_tau = tau.max(axis=0) if tau.size else np.zeros_like(tau_lim)
    peak_vel = v.max(axis=0) if v.size else np.zeros_like(vel_lim)
    worst_tau = float(np.max(peak_tau / tau_lim))
    worst_vel = float(np.max(peak_vel / vel_lim))

    print("\n" + "=" * 66)
    print("PREGRASP CONTROLLER VALIDATION")
    print("=" * 66)
    print("\n-- TASK SUCCESS (end-effector error to goal) --")
    print(f"  initial = {ee_err[0]:.4f} m   min = {ee_err.min():.4f} m   final = {ee_err[-1]:.4f} m")
    reached = bool(ee_err[-1] <= success_tol)
    first = int(np.argmax(ee_err <= success_tol)) if np.any(ee_err <= success_tol) else -1
    print(f"  success(<{success_tol*1000:.0f} mm): {'YES' if reached else 'NO'}"
          + (f"  (first at step {first})" if first >= 0 else ""))

    print("\n-- ACTUATION WITHIN LIMITS --")
    print("  joint           j1      j2      j3      j4      j5      j6      j7")
    print(_fmt_row("peak |tau| Nm", peak_tau, tau_lim))
    print(_fmt_row("peak |v| rad/s", peak_vel, vel_lim))
    within = (worst_tau <= limit_fraction) and (worst_vel <= limit_fraction)
    print(f"  worst torque = {worst_tau*100:.0f}%   worst velocity = {worst_vel*100:.0f}%   "
          f"-> {'OK' if within else f'EXCEEDS {limit_fraction*100:.0f}%'}")

    print("\n-- GAIN STABILITY (|K| over time) --")
    g = np.asarray(gain_norm)
    gains_ok = True
    if g.size and np.any(g > 0):
        tail = g[len(g) // 2:]
        jitter = float(np.max(np.abs(np.diff(g)))) if g.size > 1 else 0.0
        gains_ok = bool(np.all(np.isfinite(g)) and tail.std() <= 0.25 * (abs(tail.mean()) + 1e-9))
        print(f"  final = {g[-1]:.3f}   max = {g.max():.3f}   mean = {g.mean():.3f}   "
              f"tail std = {tail.std():.3f}   max step jump = {jitter:.3f}")
        print(f"  finite & converged: {'YES' if gains_ok else 'NO'}")
    else:
        print("  (no feedback gains — run with --gains to assess gain stability)")

    verdict = reached and within and gains_ok
    print("\n" + "-" * 66)
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}  (reached={reached}, within_limits={within})")
    print("-" * 66)


def save_plots(path, *, ee_err, v, tau, gain_norm, tau_lim, vel_lim):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plots skipped: {exc}]")
        return
    v = np.abs(np.asarray(v)); tau = np.abs(np.asarray(tau)); g = np.asarray(gain_norm)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ee_err); ax[0, 0].set_title("EE position error [m]"); ax[0, 0].grid(True)
    for j in range(tau.shape[1] if tau.ndim == 2 else 0):
        ax[0, 1].plot(tau[:, j] / tau_lim[j], label=f"j{j+1}")
    ax[0, 1].axhline(1.0, color="k", ls="--"); ax[0, 1].set_title("|torque| / limit"); ax[0, 1].grid(True)
    ax[0, 1].legend(fontsize=7, ncol=2)
    for j in range(v.shape[1] if v.ndim == 2 else 0):
        ax[1, 0].plot(v[:, j] / vel_lim[j], label=f"j{j+1}")
    ax[1, 0].axhline(1.0, color="k", ls="--"); ax[1, 0].set_title("|velocity| / limit"); ax[1, 0].grid(True)
    ax[1, 0].legend(fontsize=7, ncol=2)
    ax[1, 1].plot(g); ax[1, 1].set_title("feedback gain norm |K|"); ax[1, 1].grid(True)
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"saved plots -> {os.path.abspath(path)}")


def save_video(path, planner, q_traj, *, fps=30):
    """Offscreen render of the recorded motion. Degrades gracefully if GL/codec missing."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(planner.scene_path)
        data = mujoco.MjData(model)
        arm = [model.jnt_qposadr[model.joint(j).id]
               for j in range(model.njnt)
               if model.joint(j).name.endswith(tuple(f"joint{i}" for i in range(1, 8)))
               and "finger" not in model.joint(j).name]
        arm = arm[:7]
        renderer = mujoco.Renderer(model, height=480, width=640)
        stride = max(1, len(q_traj) // 150)
        frames = []
        for q in q_traj[::stride]:
            data.qpos[arm] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            frames.append(renderer.render())
        renderer.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[video skipped (offscreen GL unavailable): {exc}]")
        return
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, frames, fps=fps)
        print(f"saved video -> {os.path.abspath(path)}")
    except Exception as exc:  # noqa: BLE001
        # fallback: dump first/mid/last frames as PNGs
        try:
            import imageio.v2 as imageio
            for tag, fr in (("start", frames[0]), ("mid", frames[len(frames) // 2]), ("end", frames[-1])):
                imageio.imwrite(path.rsplit(".", 1)[0] + f"_{tag}.png", fr)
            print(f"[mp4 unavailable; saved start/mid/end PNGs next to {path}]")
        except Exception as exc2:  # noqa: BLE001
            print(f"[video skipped: {exc} / {exc2}]")


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp
    from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
        PandaPregraspObjective,
        PandaPregraspPlanner,
        make_panda_pregrasp_config,
    )
    from sbmpc.simulation import build_all

    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=args.viewer, gains=args.gains)
    config.sim_iterations = args.iterations
    if args.horizon is not None:
        config.MPC.horizon = args.horizon
    if args.samples is not None:
        config.MPC.num_parallel_computations = args.samples

    if args.gains:
        print("note: --gains compiles the exact-gain backprop on first run (can take a minute) ...",
              flush=True)
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
    print(f"JAX backend: {jax.default_backend()}  |  gains={config.MPC.gains}  "
          f"horizon={config.MPC.horizon}  samples={config.MPC.num_parallel_computations}")
    print(f"goal_pos = {np.asarray(planner.goal_pos)}\nrunning closed-loop sim ...", flush=True)
    sim.simulate()

    n = sim.iter
    q = np.asarray(sim.state_traj[: n + 1, : planner.nq])
    v = np.asarray(sim.state_traj[: n + 1, planner.nq : planner.nq + planner.nv])
    tau = np.asarray(sim.input_traj[:n, :])
    tau_lim = np.asarray(planner.torque_limits, dtype=float)
    vel_lim = np.asarray(planner.velocity_limits, dtype=float)
    ee_err = np.array([
        float(np.linalg.norm(np.asarray(planner.ee_position(jnp.asarray(qi))) - np.asarray(planner.goal_pos)))
        for qi in q
    ])

    report(ee_err=ee_err, v=v, tau=tau, gain_norm=gain_norm, tau_lim=tau_lim, vel_lim=vel_lim,
           success_tol=args.success_tol, limit_fraction=args.limit_fraction)
    if args.plot_path:
        save_plots(args.plot_path, ee_err=ee_err, v=v, tau=tau, gain_norm=gain_norm,
                   tau_lim=tau_lim, vel_lim=vel_lim)
    if args.video_path:
        save_video(args.video_path, planner, q)


if __name__ == "__main__":
    main()
