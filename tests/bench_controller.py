"""Benchmark and stability diagnostic for the Panda MPPI controller.

Timing methodology mirrors bench_dynamics.py:
  - JIT time measured separately (first call after build_all warm-up)
  - Steady-state = average over N_TRIALS calls, each fully blocked on GPU
  - controller.command() called directly — NOT via sim.step() —
    to avoid async GPU bleed from integrate_sim into the measured window.

Quality check (--quality / --sweep --quality):
  - Runs actual state evolution (sim.step()) for N_QUALITY steps
  - Tracks ee_error convergence toward goal
  - Tracks gain matrix stability: norm, condition number, finite check
  - Reports whether config is viable (timing + converging + gains stable)

Usage:
  # Visual test (opens MuJoCo viewer):
  pixi run python tests/bench_controller.py --visual

  # Timing + gain stability for default gains config:
  pixi run python tests/bench_controller.py --gains --quality

  # Timing + convergence for specific config:
  pixi run python tests/bench_controller.py --gains --samples 512 --quality

  # Grid search timing only (fast):
  pixi run python tests/bench_controller.py --sweep --gains

  # Grid search with convergence check (thorough):
  pixi run python tests/bench_controller.py --sweep --gains --quality
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all

TARGET_HZ = 50.0
TARGET_MS = 1000.0 / TARGET_HZ
N_WARMUP = 10
N_TRIALS = 100
N_QUALITY = 60   # simulation steps for convergence / gain-stability check


def _reset_initial_guess(planner, config):
    config.MPC.initial_guess = planner.nominal_torque_sequence(
        config.MPC.horizon,
        config.MPC.dt,
    )


def _block_command(controller, state, ref):
    """Run one command step and block until all GPU work is done."""
    result = controller.command(state, ref, num_steps=1)
    jax.block_until_ready(result)
    if controller.gains_obj.compute_gains:
        jax.block_until_ready(controller.gains_obj.cur_gains)
    return result


def _run_headless(planner, objective, config, n_trials=N_TRIALS, label=""):
    """
    Timing benchmark: controller.command() on a fixed state.
    Mimics bench_dynamics.py: JIT time + average over n_trials blocked calls.
    Returns (row_str, jit_ms, mean_ms, ok).
    """
    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    state = sim.current_state_vec()
    ref = sim.const_reference

    # build_all already fired one warm-up; measure remaining JIT/XLA work.
    t_jit = time.perf_counter()
    _block_command(sim.controller, state, ref)
    jit_ms = (time.perf_counter() - t_jit) * 1000.0

    for _ in range(1, N_WARMUP):
        _block_command(sim.controller, state, ref)

    t0 = time.perf_counter()
    for _ in range(N_WARMUP, N_WARMUP + n_trials):
        _block_command(sim.controller, state, ref)
    mean_ms = (time.perf_counter() - t0) / n_trials * 1000.0

    ok = mean_ms < TARGET_MS
    marker = "✓" if ok else "✗"
    row = (
        f"{marker} {label:52s}  "
        f"JIT={jit_ms:6.0f}ms  mean={mean_ms:6.2f}ms  "
        f"[{'PASS' if ok else 'FAIL'} @{TARGET_HZ:.0f}Hz]"
    )
    return row, jit_ms, mean_ms, ok


def _run_quality(planner, objective, config, n_steps=N_QUALITY):
    """
    Quality + gain-stability check with real state evolution.

    Runs n_steps of actual sim (state advances each step).
    Returns a dict with convergence and gain-stability metrics.
    """
    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    sim.warm_start_fn = lambda state: planner.nominal_torque_sequence_from_state(
        state,
        config.MPC.horizon,
        config.MPC.dt,
    )

    errors = []
    gain_norms = []
    gain_conds = []
    gain_finite_all = True

    diverged_at = None
    for step_i in range(n_steps):
        sim.step()  # controller.command() + integrate_sim
        state = np.asarray(jax.block_until_ready(sim.current_state_vec()))
        if not np.all(np.isfinite(state)):
            diverged_at = step_i
            break
        q = jnp.array(state[: planner.nq])
        ee_pos = np.asarray(planner.ee_position(q))
        goal = np.asarray(planner.goal_pos)
        errors.append(float(np.linalg.norm(ee_pos - goal)))

        if config.MPC.gains:
            K = np.asarray(jax.block_until_ready(sim.controller.gains))
            if not np.all(np.isfinite(K)):
                gain_finite_all = False
            gain_norms.append(float(np.linalg.norm(K, "fro")))
            sv = np.linalg.svd(K, compute_uv=False)
            gain_conds.append(float(sv[0] / (sv[-1] + 1e-12)))

    if not errors:
        return {"err_0": float("nan"), "err_mid": float("nan"), "err_final": float("nan"),
                "converging": False, "monotone": False, "errors": [], "diverged_at": diverged_at}

    err_0 = errors[0]
    err_final = errors[-1]
    half_idx = len(errors) // 2
    err_mid = errors[half_idx]

    # Converging = final error at least 10 % below initial
    converging = err_final < err_0 * 0.9
    # Monotone = error strictly decreasing in second half
    second_half = errors[half_idx:]
    monotone = all(second_half[i] >= second_half[i + 1] for i in range(len(second_half) - 1))

    result = {
        "err_0": err_0,
        "err_mid": err_mid,
        "err_final": err_final,
        "converging": converging,
        "monotone": monotone,
        "errors": errors,
        "diverged_at": diverged_at,
    }

    if config.MPC.gains and gain_norms:
        gn = np.array(gain_norms)
        gc = np.array(gain_conds)
        ss_mean = float(np.mean(gn[len(gn)//2:]))  # steady-state mean (second half)
        norm_peak = float(np.max(gn))
        # Stable: no transient spike > 50× steady-state mean AND final ≤ 5× initial.
        # cp=4 spikes 100-274×; cp=8 spikes 18-36× — 50× cleanly separates them.
        norm_stable = norm_peak < ss_mean * 50.0 and gn[-1] < gn[0] * 5.0
        result.update({
            "gain_norm_mean": float(np.mean(gn)),
            "gain_norm_std": float(np.std(gn)),
            "gain_norm_peak": norm_peak,
            "gain_norm_ss_mean": ss_mean,
            "gain_cond_mean": float(np.mean(gc)),
            "gain_sigma_max": float(np.linalg.svd(
                np.asarray(sim.controller.gains), compute_uv=False
            )[0]),
            "gain_finite": gain_finite_all,
            "gain_norm_stable": norm_stable,
        })

    return result


def _quality_row(q_result, config):
    """Format the quality result into a one-line summary."""
    err_arrow = "↓" if q_result["converging"] else "→" if q_result["err_final"] < q_result["err_0"] else "✗"
    mono = " mono" if q_result.get("monotone") else ""
    row = (
        f"     err: {q_result['err_0']:.3f}m → {q_result['err_final']:.3f}m {err_arrow}{mono}"
    )
    if config.MPC.gains and "gain_norm_mean" in q_result:
        ok_str = "finite" if q_result["gain_finite"] else "NON-FINITE"
        stab_str = "bounded" if q_result["gain_norm_stable"] else "SPIKING"
        row += (
            f"  |K|={q_result['gain_norm_mean']:.2f}±{q_result['gain_norm_std']:.2f}"
            f"  peak={q_result.get('gain_norm_peak', 0):.1f}"
            f"  cond={q_result['gain_cond_mean']:.1f}"
            f"  σ_max={q_result['gain_sigma_max']:.2f}"
            f"  {ok_str} {stab_str}"
        )
    return row


def _is_viable(timing_ok, q_result, config):
    """True if timing passes AND controller converges AND gains are stable (no spikes)."""
    if not timing_ok:
        return False
    if q_result.get("diverged_at") is not None:
        return False
    if not q_result.get("converging", True):
        return False
    if config.MPC.gains:
        if not q_result.get("gain_finite", True):
            return False
        if not q_result.get("gain_norm_stable", True):
            return False
    return True


def run_visual(planner, objective, config):
    """Open MuJoCo viewer and run until window closes."""
    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    sim.warm_start_fn = lambda state: planner.nominal_torque_sequence_from_state(
        state,
        config.MPC.horizon,
        config.MPC.dt,
    )

    def post_update(s):
        state = s.current_state_vec()
        q = state[: planner.nq]
        ee_pos = planner.ee_position(q)
        err = float(jnp.linalg.norm(ee_pos - planner.goal_pos))
        gain_norm = (
            float(jnp.linalg.norm(s.controller.gains)) if config.MPC.gains else 0.0
        )
        print(
            f"iter={s.iter:04d} "
            f"ee_pos={np.round(np.asarray(ee_pos), 3)} "
            f"err={err:.3f}m "
            f"plan={s.last_command_time_ms:.1f}ms"
            + (f" |K|={gain_norm:.3f}" if config.MPC.gains else "")
        )

    sim.post_update = post_update
    print(f"\nJAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(
            f"horizon={config.MPC.horizon}  samples={config.MPC.num_parallel_computations}  "
            f"control_points={config.MPC.num_control_points}  gains={config.MPC.gains}  "
            f"gK={config.MPC.gain_samples_per_cycle}  gM={config.MPC.gain_buffer_size}  "
            f"mjx={config.robot.mjx_opts}"
        )
    sim.simulate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--sweep", action="store_true", help="Sweep horizon × samples × cp")
    parser.add_argument("--gains", action="store_true", help="Enable gain computation")
    parser.add_argument("--quality", action="store_true",
                        help="Run state-evolution quality + gain-stability check")
    parser.add_argument("--steps", type=int, default=N_TRIALS, help="Timing trials per config")
    parser.add_argument("--quality-steps", type=int, default=N_QUALITY,
                        help="Simulation steps for quality check")
    parser.add_argument("--gain-samples-per-cycle", type=int, default=None,
                        help="Buffered-gain: how many of the MPPI samples to backprop per cycle.")
    parser.add_argument("--gain-buffer-size", type=int, default=None,
                        help="Buffered-gain: total accumulated samples before a K update; must be a multiple of --gain-samples-per-cycle.")
    parser.add_argument("--mjx-iterations", type=int, default=None)
    parser.add_argument("--mjx-ls-iterations", type=int, default=None)
    parser.add_argument("--mjx-tolerance", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--control-points", type=int, default=None)
    args = parser.parse_args()

    mjx_opts = {}
    if args.mjx_iterations is not None:
        mjx_opts["iterations"] = args.mjx_iterations
    if args.mjx_ls_iterations is not None:
        mjx_opts["ls_iterations"] = args.mjx_ls_iterations
    if args.mjx_tolerance is not None:
        mjx_opts["tolerance"] = args.mjx_tolerance
    args.mjx_opts = mjx_opts or None

    def _apply_gain_knobs(config):
        config.MPC.gain_method = "exact"
        if args.gain_samples_per_cycle is not None:
            config.MPC.gain_samples_per_cycle = args.gain_samples_per_cycle
        if args.gain_buffer_size is not None:
            config.MPC.gain_buffer_size = args.gain_buffer_size
        config.robot.mjx_opts = args.mjx_opts

    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)

    if args.visual:
        config = make_panda_pregrasp_config(planner, visualize=True, gains=True)
        config.MPC.horizon = args.horizon if args.horizon is not None else 8
        config.MPC.num_parallel_computations = args.samples if args.samples is not None else 1024
        config.MPC.num_control_points = args.control_points if args.control_points is not None else 8
        _apply_gain_knobs(config)
        _reset_initial_guess(planner, config)
        run_visual(planner, objective, config)
        return

    print(f"\nJAX backend: {jax.default_backend()},  devices: {jax.devices()}")
    print(f"N_WARMUP={N_WARMUP}, N_TRIALS={args.steps}, N_QUALITY={args.quality_steps}\n")

    if args.sweep:
        run_qual = args.quality
        print(
            f"=== MPPI grid search  gains={args.gains}  quality={'yes' if run_qual else 'no'} ===\n"
            f"    target: mean < {TARGET_MS:.0f}ms ({TARGET_HZ:.0f}Hz)"
            + (f"  +  convergence + gain stability" if run_qual else "")
            + "\n"
        )

        horizons = [4, 8, 12, 16]
        samples_list = [32, 256, 512, 1024]
        control_points = [4, 8, 12, 16]

        viable = []
        for h in horizons:
            valid_cp = [cp for cp in control_points if 2 <= cp <= h]
            for s in samples_list:
                for cp in valid_cp:
                    config = make_panda_pregrasp_config(planner, visualize=False, gains=args.gains)
                    config.MPC.horizon = h
                    config.MPC.num_parallel_computations = s
                    config.MPC.num_control_points = cp
                    _apply_gain_knobs(config)
                    _reset_initial_guess(planner, config)
                    label = (
                        f"h={h:2d} n={s:4d} cp={cp} "
                        f"gK={config.MPC.gain_samples_per_cycle} gM={config.MPC.gain_buffer_size} "
                        f"mjx={config.robot.mjx_opts}"
                    )
                    t_row, _, mean_ms, t_ok = _run_headless(
                        planner,
                        objective,
                        config,
                        args.steps,
                        label,
                    )
                    print(t_row)
                    if run_qual:
                        q = _run_quality(
                            planner,
                            objective,
                            config,
                            args.quality_steps,
                        )
                        print(_quality_row(q, config))
                        if _is_viable(t_ok, q, config):
                            viable.append((h, s, cp, mean_ms))
                    elif t_ok:
                        viable.append((h, s, cp, mean_ms))

        print(f"\n=== VIABLE CONFIGS ({len(viable)}) ===")
        for h, s, cp, ms in sorted(viable, key=lambda x: -x[1] * x[0]):
            print(f"  h={h:2d}  n={s:4d}  cp={cp}  mean={ms:.2f}ms")
        return

    # ── Single-config benchmark ──────────────────────────────────────────────
    config = make_panda_pregrasp_config(planner, visualize=False, gains=args.gains)
    if args.horizon is not None:
        config.MPC.horizon = args.horizon
    if args.samples is not None:
        config.MPC.num_parallel_computations = args.samples
    if args.control_points is not None:
        config.MPC.num_control_points = args.control_points
    _apply_gain_knobs(config)
    if any(v is not None for v in (args.horizon, args.samples, args.control_points)):
        _reset_initial_guess(planner, config)

    label = (
        f"h={config.MPC.horizon} n={config.MPC.num_parallel_computations} "
        f"cp={config.MPC.num_control_points} gains={config.MPC.gains} "
        f"gK={config.MPC.gain_samples_per_cycle} gM={config.MPC.gain_buffer_size} "
        f"mjx={config.robot.mjx_opts}"
    )
    t_row, _, mean_ms, t_ok = _run_headless(
        planner,
        objective,
        config,
        args.steps,
        label,
    )
    print(t_row)

    # Gains overhead: compare to same config with gains disabled
    if args.gains:
        config_ng = make_panda_pregrasp_config(planner, visualize=False, gains=False)
        config_ng.MPC.horizon = config.MPC.horizon
        config_ng.MPC.num_parallel_computations = config.MPC.num_parallel_computations
        config_ng.MPC.num_control_points = config.MPC.num_control_points
        _reset_initial_guess(planner, config_ng)
        label_ng = (
            f"h={config_ng.MPC.horizon} n={config_ng.MPC.num_parallel_computations} "
            f"cp={config_ng.MPC.num_control_points} gains=False"
        )
        _, _, mean_ng, _ = _run_headless(planner, objective, config_ng, args.steps, label_ng)
        print(f"  → Gains overhead: {mean_ms - mean_ng:.2f}ms  (paper target ~1ms, FD adds nx={planner.nx} batches)")

    # Quality + gain-stability check
    if args.quality or args.gains:
        print(f"\n--- Quality check ({args.quality_steps} steps of state evolution) ---")
        q = _run_quality(
            planner,
            objective,
            config,
            args.quality_steps,
        )
        print(_quality_row(q, config))

        if args.gains and q.get("errors"):
            errors = q["errors"]
            if q.get("diverged_at") is not None:
                print(f"  !! State diverged (NaN) at step {q['diverged_at']}")
            print("\n  ee_error trajectory:")
            err0 = errors[0] if errors[0] > 0 else 1.0
            for i in range(0, len(errors), max(1, len(errors) // 10)):
                e = errors[i]
                bar = "█" * int(e / err0 * 20) if np.isfinite(e) else "???"
                print(f"    step {i:3d}: {e:.4f}m  {bar}")

        viable = _is_viable(t_ok, q, config)
        print(f"\n  Verdict: {'✓ VIABLE' if viable else '✗ NOT VIABLE'}")
        if not t_ok:
            print("    - timing exceeds 20ms target")
        if not q.get("converging"):
            print("    - end-effector not converging to goal")
        if config.MPC.gains:
            if not q.get("gain_finite", True):
                print("    - gains contain NaN/Inf — NOT safe for low-level control")
            if not q.get("gain_norm_stable", True):
                print("    - gain norm growing — NOT safe for low-level control")


if __name__ == "__main__":
    main()
