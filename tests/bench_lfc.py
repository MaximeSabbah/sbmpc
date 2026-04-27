"""MuJoCo-side validation of SB-MPC gains applied through the LFC law.

This intentionally differs from bench_controller.py: it applies

    tau = tau_ff + K_lfc @ (x_desired - x_measured)

inside each control period, with K_lfc using the same sign convention as the
ROS linear_feedback_controller bridge. Use ``--timing-mode gazebo`` to include
the ROS/Gazebo timing semantics where the previous control remains active while
the Python bridge computes the next SB-MPC command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

FRANKA_ARM_VELOCITY_LIMITS = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
    dtype=np.float64,
)

from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all, construct_mj_visualizer_from_model


DEFAULT_JAX_CACHE_DIR = ".jax_cache"


BENCH_PRESETS = {
    "custom": {},
    "exact-feedback": {
        "timing_mode": "gazebo",
        "gain_method": "exact",
        "phase0_async_probe": False,
        "background_gain_worker": True,
        "gain_samples_per_cycle": 128,
        "gain_buffer_size": 512,
        "feedback": True,
        "gains": True,
    },
    "exact-phase0": {
        "timing_mode": "gazebo",
        "gain_method": "exact",
        "phase0_async_probe": True,
        "background_gain_worker": False,
        "gain_samples_per_cycle": 128,
        "gain_buffer_size": 512,
        "feedback": True,
        "gains": True,
    },
    "exact-feedforward": {
        "timing_mode": "gazebo",
        "gain_method": "exact",
        "phase0_async_probe": False,
        "background_gain_worker": True,
        "gain_samples_per_cycle": 128,
        "gain_buffer_size": 512,
        "feedback": False,
        "gains": True,
    },
    "fast-feedforward": {
        "timing_mode": "gazebo",
        "gain_method": "finite_difference",
        "phase0_async_probe": False,
        "background_gain_worker": False,
        "feedback": False,
        "gains": False,
    },
    "fd-feedback": {
        "timing_mode": "gazebo",
        "gain_method": "finite_difference",
        "phase0_async_probe": False,
        "background_gain_worker": False,
        "gain_fd_samples": 128,
        "feedback": True,
        "gains": True,
    },
}


PRESET_FLAG_NAMES = {
    "timing_mode": ("--timing-mode",),
    "gain_method": ("--gain-method",),
    "phase0_async_probe": ("--phase0-async-probe",),
    "background_gain_worker": ("--background-gain-worker",),
    "gain_samples_per_cycle": ("--gain-samples-per-cycle",),
    "gain_buffer_size": ("--gain-buffer-size",),
    "gain_fd_samples": ("--gain-fd-samples",),
    "feedback": ("--feedback", "--no-feedback"),
    "gains": ("--gains", "--no-gains"),
}


def _arg_was_supplied(argv: list[str], flag_names: tuple[str, ...]) -> bool:
    return any(
        token == flag_name or token.startswith(f"{flag_name}=")
        for token in argv
        for flag_name in flag_names
    )


def _apply_preset(args: argparse.Namespace, argv: list[str]) -> None:
    preset = BENCH_PRESETS[args.preset]
    for dest, value in preset.items():
        if not _arg_was_supplied(argv, PRESET_FLAG_NAMES.get(dest, ())):
            setattr(args, dest, value)


def _configure_jax_cache(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def _phase0_async_probe_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "phase0_async_probe", False))


def _background_gain_worker_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "background_gain_worker", False))


def _exact_gain_context_enabled(args: argparse.Namespace) -> bool:
    return _phase0_async_probe_enabled(args) or _background_gain_worker_enabled(args)


def _phase0_refresh_due(args: argparse.Namespace, step_idx: int) -> bool:
    return _phase0_async_probe_enabled(args) and (step_idx + 1) % args.gain_refresh_cycles == 0


def _reset_guess(sim, planner, config, state) -> None:
    sim.controller.sampler.optimal_samples = planner.nominal_torque_sequence_from_state(
        state,
        config.MPC.horizon,
        config.MPC.dt,
    )


def _ee_error(planner, state: np.ndarray) -> float:
    q = jnp.asarray(state[: planner.nq], dtype=jnp.float32)
    ee_pos = np.asarray(jax.block_until_ready(planner.ee_position(q)))
    return float(np.linalg.norm(ee_pos - np.asarray(planner.goal_pos)))


def _build_lfc_sim(args: argparse.Namespace):
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.gains = args.gains
    config.MPC.gain_method = args.gain_method
    config.MPC.gain_fd_epsilon = args.gain_fd_epsilon
    config.MPC.gain_fd_scheme = args.gain_fd_scheme
    config.MPC.gain_fd_num_samples = args.gain_fd_samples
    config.MPC.dt = args.dt
    config.MPC.horizon = args.horizon
    config.MPC.num_parallel_computations = args.samples
    config.MPC.num_control_points = args.control_points
    config.MPC.gain_samples_per_cycle = getattr(args, "gain_samples_per_cycle", None)
    config.MPC.gain_buffer_size = getattr(args, "gain_buffer_size", None)
    config.robot.mjx_opts = getattr(args, "mjx_opts", None)
    config.MPC.initial_guess = planner.nominal_torque_sequence(
        config.MPC.horizon,
        config.MPC.dt,
    )

    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
        warm_start_gains=not _exact_gain_context_enabled(args),
    )
    return planner, config, sim


def _compile_controller(sim, planner, config, args: argparse.Namespace) -> None:
    setup_start = time.perf_counter()
    state = sim.current_state_vec()
    _reset_guess(sim, planner, config, state)
    controller_start = time.perf_counter()
    if _exact_gain_context_enabled(args):
        print("[setup] compiling foreground rollout...", flush=True)
        warm = sim.controller.command(
            state,
            sim.const_reference,
            num_steps=1,
            update_gains=False,
            capture_gain_context=True,
        )
        jax.block_until_ready(warm)
        print("[setup] compiling phase0 exact gain refresh...", flush=True)
        sim.controller.phase0_refresh_exact_gains()
        warm_gain = sim.controller._phase0_exact_window.compute_gain(sim.controller.gains_obj)
        jax.block_until_ready(warm_gain)
        sim.controller._set_current_gains(warm_gain)
        print("[setup] compiling foreground rollout with published gain...", flush=True)
        warm = sim.controller.command(
            state,
            sim.const_reference,
            num_steps=1,
            update_gains=False,
            capture_gain_context=False,
        )
        jax.block_until_ready(warm)
        sim.controller.reset_phase0_exact_gain_probe(reset_published_gain=True)
    else:
        warm = sim.controller.command(state, sim.const_reference, num_steps=1)
        jax.block_until_ready(warm)
        jax.block_until_ready(sim.controller.gains)
    controller_ms = (time.perf_counter() - controller_start) * 1000.0

    lfc_start = time.perf_counter()
    print("[setup] compiling LFC/MuJoCo runtime path...", flush=True)
    warm_state = sim.current_state
    state_np = np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)
    zero_tau = np.zeros(sim.controller.gains.shape[0], dtype=np.float64)
    zero_control = {
        "tau_ff": zero_tau,
        "K_lfc": np.zeros((sim.controller.gains.shape[0], state_np.size), dtype=np.float64),
        "desired": state_np,
    }
    _apply_lfc_control(
        sim,
        planner,
        zero_control,
        config.MPC.dt / args.substeps,
        args.substeps,
        config.MPC.dt,
        clip_torque=args.clip_torque,
        clip_velocity=args.clip_velocity,
    )
    _desired_state_for_control(
        sim,
        args,
        state_np,
        zero_tau,
        config.MPC.dt,
    )
    jax.block_until_ready(sim.current_state_vec())
    sim.current_state = warm_state
    lfc_ms = (time.perf_counter() - lfc_start) * 1000.0
    total_ms = (time.perf_counter() - setup_start) * 1000.0
    print(
        f"[setup] controller_warmup={controller_ms:.1f}ms "
        f"lfc_runtime_warmup={lfc_ms:.1f}ms total={total_ms:.1f}ms",
        flush=True,
    )


def _apply_lfc_control(
    sim,
    planner,
    control: dict[str, np.ndarray] | None,
    duration_sec: float,
    substeps_per_control_period: int,
    control_period_sec: float,
    *,
    clip_torque: bool,
    clip_velocity: bool,
) -> float:
    if control is None or duration_sec <= 0.0:
        return 0.0

    substep_dt = control_period_sec / substeps_per_control_period
    substeps = max(1, int(np.ceil(duration_sec / substep_dt)))
    torque_limits = np.asarray(planner.torque_limits, dtype=np.float64)
    feedback_peak = 0.0

    for substep in range(substeps):
        dt = min(substep_dt, duration_sec - substep * substep_dt)
        if dt <= 0.0:
            break
        measured = np.asarray(
            jax.block_until_ready(sim.current_state_vec()),
            dtype=np.float64,
        )
        feedback = control["K_lfc"] @ (control["desired"] - measured)
        feedback_peak = max(
            feedback_peak,
            float(np.max(np.abs(feedback), initial=0.0)),
        )
        tau = control["tau_ff"] + feedback
        if clip_torque:
            tau = np.clip(tau, -torque_limits, torque_limits)
        next_state = sim.model.integrate_sim(
            sim.current_state,
            jnp.asarray(tau, dtype=jnp.float32),
            dt,
        )
        if clip_velocity:
            velocity_limits = jnp.asarray(FRANKA_ARM_VELOCITY_LIMITS, dtype=jnp.float32)
            next_state = next_state.at[planner.nq : planner.nq + planner.nv].set(
                jnp.clip(
                    next_state[planner.nq : planner.nq + planner.nv],
                    -velocity_limits,
                    velocity_limits,
                )
            )
        sim.current_state = next_state

    return feedback_peak


def _predict_state(sim, state: np.ndarray, tau: np.ndarray, dt: float) -> np.ndarray:
    predicted = sim.model.integrate_sim(
        jnp.asarray(state, dtype=jnp.float32),
        jnp.asarray(tau, dtype=jnp.float32),
        dt,
    )
    return np.asarray(jax.block_until_ready(predicted), dtype=np.float64)


def _desired_state_for_control(
    sim,
    args: argparse.Namespace,
    base_state: np.ndarray,
    tau_ff: np.ndarray,
    control_period_sec: float,
) -> np.ndarray:
    if args.desired_state_mode == "base":
        return base_state
    if args.desired_state_mode == "midpoint":
        return _predict_state(
            sim,
            base_state,
            tau_ff,
            0.5 * control_period_sec,
        )
    if args.desired_state_mode == "next":
        return _predict_state(
            sim,
            base_state,
            tau_ff,
            control_period_sec,
        )
    raise ValueError(f"unsupported desired_state_mode={args.desired_state_mode!r}")


def _plan_control(
    sim,
    planner,
    config,
    args: argparse.Namespace,
    state: np.ndarray,
    step_idx: int,
):
    _reset_guess(sim, planner, config, jnp.asarray(state, dtype=jnp.float32))
    jax.block_until_ready(sim.controller.sampler.optimal_samples)

    start = time.perf_counter()
    if _exact_gain_context_enabled(args):
        input_sequence = sim.controller.command(
            jnp.asarray(state, dtype=jnp.float32),
            sim.const_reference,
            num_steps=1,
            update_gains=False,
            capture_gain_context=True,
        )
    else:
        input_sequence = sim.controller.command(
            jnp.asarray(state, dtype=jnp.float32),
            sim.const_reference,
            num_steps=1,
        )
    jax.block_until_ready(input_sequence)
    planning_ms = (time.perf_counter() - start) * 1000.0
    tau_ff = np.asarray(input_sequence[0], dtype=np.float64)
    if _phase0_refresh_due(args, step_idx):
        probe = sim.controller.phase0_refresh_exact_gains()
    elif _background_gain_worker_enabled(args):
        probe = sim.controller.async_exact_gain_status()
        completed_count = probe.get("completed_batch_count", 0)
        if completed_count == getattr(args, "_last_async_completed_batch_count", -1):
            probe.update(
                {
                    "subset_select_ms": 0.0,
                    "snapshot_pack_ms": 0.0,
                    "gain_grad_ms": 0.0,
                    "gain_synth_ms": 0.0,
                    "gain_refresh_ms": 0.0,
                    "gain_published": False,
                    "source_cycle_id": None,
                }
            )
        else:
            args._last_async_completed_batch_count = completed_count
        if probe.get("worker_error") is not None:
            raise RuntimeError(f"background gain worker failed: {probe['worker_error']}")
    elif _phase0_async_probe_enabled(args):
        probe = sim.controller.phase0_probe_status()
        probe.update(
            {
                "subset_select_ms": 0.0,
                "snapshot_pack_ms": 0.0,
                "gain_grad_ms": 0.0,
                "gain_synth_ms": 0.0,
                "gain_refresh_ms": 0.0,
                "gain_published": False,
                "source_cycle_id": None,
            }
        )
    else:
        probe = {
            "first_gain_ready_cycle": None,
            "published_gain_age_cycles": float("nan"),
            "queue_depth_max": 0,
            "dropped_snapshot_count": 0,
            "rolling_window_fill": 0,
            "subset_select_ms": 0.0,
            "snapshot_pack_ms": 0.0,
            "gain_grad_ms": 0.0,
            "gain_synth_ms": 0.0,
            "gain_refresh_ms": 0.0,
            "gain_published": False,
            "source_cycle_id": None,
        }
    K_sbmpc = np.asarray(jax.block_until_ready(sim.controller.gains), dtype=np.float64)

    return {
        "tau_ff": tau_ff,
        # Same convention as the ROS bridge: SB-MPC exposes du/dx_measured,
        # while LFC multiplies (desired - measured).
        "K_lfc": -K_sbmpc if args.feedback else np.zeros_like(K_sbmpc),
        "gain_norm": float(np.linalg.norm(K_sbmpc)),
        "planning_ms": planning_ms,
        "gain_probe": probe,
    }


def _lfc_step(
    sim,
    planner,
    config,
    args: argparse.Namespace,
    previous_control: dict[str, np.ndarray] | None,
    previous_planning_ms: float,
    step_idx: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    control_period_sec = config.MPC.dt
    timer_to_publish_delay_sec = previous_planning_ms * 1e-3

    if args.timing_mode == "gazebo":
        # In ROS, the previous control remains active from the last publish
        # until the next timer tick, then also while the new plan is computing.
        pre_timer_duration = max(0.0, control_period_sec - timer_to_publish_delay_sec)
        _apply_lfc_control(
            sim,
            planner,
            previous_control,
            pre_timer_duration,
            args.substeps,
            control_period_sec,
            clip_torque=args.clip_torque,
            clip_velocity=args.clip_velocity,
        )

    planning_state = np.asarray(
        jax.block_until_ready(sim.current_state_vec()),
        dtype=np.float64,
    )
    planned_control = _plan_control(sim, planner, config, args, planning_state, step_idx)

    feedback_peak = 0.0
    if args.timing_mode == "gazebo":
        feedback_peak = _apply_lfc_control(
            sim,
            planner,
            previous_control,
            planned_control["planning_ms"] * 1e-3,
            args.substeps,
            control_period_sec,
            clip_torque=args.clip_torque,
            clip_velocity=args.clip_velocity,
        )
        desired_base = (
            np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)
            if args.retime_initial_state
            else planning_state
        )
    else:
        desired_base = planning_state

    desired = _desired_state_for_control(
        sim,
        args,
        desired_base,
        planned_control["tau_ff"],
        control_period_sec,
    )

    current_control = {
        "tau_ff": planned_control["tau_ff"],
        "K_lfc": planned_control["K_lfc"],
        "desired": desired,
    }

    if args.timing_mode == "immediate":
        feedback_peak = _apply_lfc_control(
            sim,
            planner,
            current_control,
            control_period_sec,
            args.substeps,
            control_period_sec,
            clip_torque=args.clip_torque,
            clip_velocity=args.clip_velocity,
        )

    state_np = np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)
    metrics = {
        "error": _ee_error(planner, state_np),
        "gain_norm": planned_control["gain_norm"],
        "feedback_peak": feedback_peak,
        "planning_ms": planned_control["planning_ms"],
        "subset_select_ms": planned_control["gain_probe"]["subset_select_ms"],
        "snapshot_pack_ms": planned_control["gain_probe"]["snapshot_pack_ms"],
        "gain_grad_ms": planned_control["gain_probe"]["gain_grad_ms"],
        "gain_synth_ms": planned_control["gain_probe"]["gain_synth_ms"],
        "gain_refresh_ms": planned_control["gain_probe"]["gain_refresh_ms"],
        "published_gain_age_cycles": planned_control["gain_probe"]["published_gain_age_cycles"],
        "rolling_window_fill": planned_control["gain_probe"]["rolling_window_fill"],
        "queue_depth_max": planned_control["gain_probe"]["queue_depth_max"],
        "dropped_snapshot_count": planned_control["gain_probe"]["dropped_snapshot_count"],
        "first_gain_ready_cycle": planned_control["gain_probe"]["first_gain_ready_cycle"],
        "gain_published": planned_control["gain_probe"]["gain_published"],
        "completed_batch_count": planned_control["gain_probe"].get("completed_batch_count", 0),
        "state": state_np,
    }
    return metrics, current_control


def run_lfc_validation(args: argparse.Namespace) -> dict[str, object]:
    planner, config, sim = _build_lfc_sim(args)
    errors: list[float] = []
    gain_norms: list[float] = []
    feedback_peaks: list[float] = []
    plan_times_ms: list[float] = []
    subset_select_ms: list[float] = []
    snapshot_pack_ms: list[float] = []
    gain_grad_ms: list[float] = []
    gain_synth_ms: list[float] = []
    gain_refresh_ms: list[float] = []
    published_gain_age_cycles: list[float] = []
    rolling_window_fill: list[float] = []
    q_history: list[np.ndarray] = []
    v_history: list[np.ndarray] = []
    previous_control: dict[str, np.ndarray] | None = None
    previous_planning_ms = 0.0

    _compile_controller(sim, planner, config, args)
    if _background_gain_worker_enabled(args):
        print("[setup] starting background exact-gain worker...", flush=True)
        sim.controller.start_async_exact_gain_worker(reset_published_gain=True)
        args._last_async_completed_batch_count = 0

    queue_depth_max = 0
    dropped_snapshot_count = 0
    first_gain_ready_cycle = None
    completed_batch_count = 0
    try:
        for step_idx in range(args.steps):
            metrics, previous_control = _lfc_step(
                sim,
                planner,
                config,
                args,
                previous_control,
                previous_planning_ms,
                step_idx,
            )
            previous_planning_ms = metrics["planning_ms"]
            state_np = metrics["state"]
            errors.append(metrics["error"])
            gain_norms.append(metrics["gain_norm"])
            feedback_peaks.append(metrics["feedback_peak"])
            plan_times_ms.append(metrics["planning_ms"])
            subset_select_ms.append(metrics["subset_select_ms"])
            snapshot_pack_ms.append(metrics["snapshot_pack_ms"])
            gain_grad_ms.append(metrics["gain_grad_ms"])
            gain_synth_ms.append(metrics["gain_synth_ms"])
            gain_refresh_ms.append(metrics["gain_refresh_ms"])
            published_gain_age_cycles.append(metrics["published_gain_age_cycles"])
            rolling_window_fill.append(metrics["rolling_window_fill"])
            queue_depth_max = max(queue_depth_max, metrics["queue_depth_max"])
            dropped_snapshot_count = max(dropped_snapshot_count, metrics["dropped_snapshot_count"])
            completed_batch_count = max(completed_batch_count, metrics["completed_batch_count"])
            if first_gain_ready_cycle is None and metrics["first_gain_ready_cycle"] is not None:
                first_gain_ready_cycle = metrics["first_gain_ready_cycle"]
            q_history.append(state_np[: planner.nq])
            v_history.append(state_np[planner.nq : planner.nq + planner.nv])
            print_every = getattr(args, "print_every", 0)
            if print_every > 0 and step_idx % print_every == 0:
                msg = (
                    f"step={step_idx:04d} "
                    f"err={metrics['error']:.4f}m "
                    f"|K|={metrics['gain_norm']:.3f} "
                    f"fb_peak={metrics['feedback_peak']:.3f}Nm "
                    f"plan={metrics['planning_ms']:.1f}ms"
                )
                if _exact_gain_context_enabled(args):
                    msg += (
                        f" gain_refresh={metrics['gain_refresh_ms']:.1f}ms "
                        f"fill={metrics['rolling_window_fill']:.0f} "
                        f"age={metrics['published_gain_age_cycles']:.0f}"
                    )
                print(msg, flush=True)
    finally:
        if _background_gain_worker_enabled(args):
            sim.controller.stop_async_exact_gain_worker()

    q = np.asarray(q_history, dtype=np.float64)
    v = np.asarray(v_history, dtype=np.float64)
    tail = slice(len(q) // 2, None)
    return {
        "errors": np.asarray(errors, dtype=np.float64),
        "gain_norms": np.asarray(gain_norms, dtype=np.float64),
        "feedback_peaks": np.asarray(feedback_peaks, dtype=np.float64),
        "plan_times_ms": np.asarray(plan_times_ms, dtype=np.float64),
        "subset_select_ms": np.asarray(subset_select_ms, dtype=np.float64),
        "snapshot_pack_ms": np.asarray(snapshot_pack_ms, dtype=np.float64),
        "gain_grad_ms": np.asarray(gain_grad_ms, dtype=np.float64),
        "gain_synth_ms": np.asarray(gain_synth_ms, dtype=np.float64),
        "gain_refresh_ms": np.asarray(gain_refresh_ms, dtype=np.float64),
        "published_gain_age_cycles": np.asarray(published_gain_age_cycles, dtype=np.float64),
        "rolling_window_fill": np.asarray(rolling_window_fill, dtype=np.float64),
        "queue_depth_max": int(queue_depth_max),
        "dropped_snapshot_count": int(dropped_snapshot_count),
        "first_gain_ready_cycle": first_gain_ready_cycle,
        "completed_batch_count": int(completed_batch_count),
        "tail_joint_spans": np.ptp(q[tail], axis=0),
        "joint_velocity_abs_max": float(np.max(np.abs(v), initial=0.0)),
        "joint_velocity_rms_mean": float(np.mean(np.sqrt(np.mean(v * v, axis=1)))),
        "q_history": q,
        "v_history": v,
    }


def run_lfc_visual(args: argparse.Namespace) -> None:
    planner, config, sim = _build_lfc_sim(args)
    _compile_controller(sim, planner, config, args)
    if _background_gain_worker_enabled(args):
        print("[setup] starting background exact-gain worker...", flush=True)
        sim.controller.start_async_exact_gain_worker(reset_published_gain=True)
        args._last_async_completed_batch_count = 0

    visualizer = None
    try:
        visualizer = construct_mj_visualizer_from_model(sim.model, config)
        previous_control: dict[str, np.ndarray] | None = None
        previous_planning_ms = 0.0
        for step_idx in range(args.steps):
            if not visualizer.is_running():
                break

            step_start = time.perf_counter()
            metrics, previous_control = _lfc_step(
                sim,
                planner,
                config,
                args,
                previous_control,
                previous_planning_ms,
                step_idx,
            )
            previous_planning_ms = metrics["planning_ms"]
            state_np = metrics["state"]
            visualizer.set_qpos(state_np[: planner.nq])

            if step_idx % args.print_every == 0:
                print(
                    f"step={step_idx:04d} "
                    f"err={metrics['error']:.4f}m "
                    f"|K|={metrics['gain_norm']:.3f} "
                    f"fb_peak={metrics['feedback_peak']:.3f}Nm "
                    f"plan={metrics['planning_ms']:.1f}ms"
                    + (
                        f" gain_refresh={metrics['gain_refresh_ms']:.1f}ms "
                        f"age={metrics['published_gain_age_cycles']:.0f}"
                        if _exact_gain_context_enabled(args)
                        else ""
                    )
                )

            if args.realtime:
                elapsed = time.perf_counter() - step_start
                sleep_time = config.MPC.dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if _background_gain_worker_enabled(args):
            sim.controller.stop_async_exact_gain_worker()
        if visualizer is not None:
            visualizer.close()


def _joint_vel_hf_energy(v: np.ndarray, dt: float, f_lo: float = 5.0, f_hi: float = 50.0) -> float:
    """Peak-per-joint spectral magnitude in the [f_lo, f_hi] Hz band, normalised.

    Used as a scalar oscillation metric: a steady hold should be near 0, a
    controller ringing in the 5-50 Hz band produces a larger value.
    """
    if len(v) < 4:
        return 0.0
    v_centered = v - np.mean(v, axis=0, keepdims=True)
    window = np.hanning(len(v_centered))[:, np.newaxis]
    spectrum = np.fft.rfft(v_centered * window, axis=0)
    freqs = np.fft.rfftfreq(len(v_centered), d=dt)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return 0.0
    magnitude = np.abs(spectrum[band])
    norm = np.sum(window) * 0.5
    return float(np.max(magnitude) / max(norm, 1e-12))


def _emit_reference(args: argparse.Namespace, result: dict[str, object], path: str) -> None:
    """Write the immediate-mode MuJoCo baseline JSON.

    Only `--timing-mode immediate` is baselined today: `gazebo` mode currently
    NaNs because planning_ms > dt, and is the subject of a separate
    compute-reduction chunk. When that chunk lands and gazebo-mode stabilises,
    this emit path will be extended to cover both modes.
    """
    if args.timing_mode != "immediate":
        raise ValueError(
            "--emit-reference is only supported for --timing-mode immediate "
            "today. gazebo-mode does not converge yet."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plan_times_ms = np.asarray(result["plan_times_ms"], dtype=np.float64)
    errors = np.asarray(result["errors"], dtype=np.float64)
    v = np.asarray(result["v_history"], dtype=np.float64)
    tail_start = len(v) // 2
    hf_energy = _joint_vel_hf_energy(v[tail_start:], args.dt)
    payload = {
        "config": {
            "preset": args.preset,
            "gain_method": args.gain_method,
            "gains": bool(args.gains),
            "phase0_async_probe": bool(args.phase0_async_probe),
            "background_gain_worker": bool(args.background_gain_worker),
            "gain_refresh_cycles": int(args.gain_refresh_cycles),
            "feedback": bool(args.feedback),
            "dt": float(args.dt),
            "horizon": int(args.horizon),
            "samples": int(args.samples),
            "control_points": int(args.control_points),
            "substeps": int(args.substeps),
            "timing_mode": args.timing_mode,
            "steps": int(args.steps),
            "backend": jax.default_backend(),
            "jax_cache_dir": args.jax_cache_dir,
        },
        "errors": errors.tolist(),
        "gain_norms": np.asarray(result["gain_norms"], dtype=np.float64).tolist(),
        "feedback_peaks": np.asarray(result["feedback_peaks"], dtype=np.float64).tolist(),
        "plan_times_ms": plan_times_ms.tolist(),
        "subset_select_ms": np.asarray(result["subset_select_ms"], dtype=np.float64).tolist(),
        "snapshot_pack_ms": np.asarray(result["snapshot_pack_ms"], dtype=np.float64).tolist(),
        "gain_grad_ms": np.asarray(result["gain_grad_ms"], dtype=np.float64).tolist(),
        "gain_synth_ms": np.asarray(result["gain_synth_ms"], dtype=np.float64).tolist(),
        "gain_refresh_ms": np.asarray(result["gain_refresh_ms"], dtype=np.float64).tolist(),
        "published_gain_age_cycles": np.asarray(result["published_gain_age_cycles"], dtype=np.float64).tolist(),
        "tail_joint_spans": np.asarray(result["tail_joint_spans"], dtype=np.float64).tolist(),
        "summary": {
            "error_initial": float(errors[0]),
            "error_final": float(errors[-1]),
            "error_min": float(np.min(errors)),
            "tail_error_mean": float(np.mean(errors[tail_start:])),
            "tail_error_std": float(np.std(errors[tail_start:])),
            "plan_ms_p50": float(np.percentile(plan_times_ms, 50)),
            "plan_ms_p95": float(np.percentile(plan_times_ms, 95)),
            "plan_ms_p99": float(np.percentile(plan_times_ms, 99)),
            "joint_velocity_abs_max": float(result["joint_velocity_abs_max"]),
            "joint_velocity_rms_mean": float(result["joint_velocity_rms_mean"]),
            "joint_vel_hf_energy": hf_energy,
            "gain_norm_final": float(np.asarray(result["gain_norms"])[-1]),
            "feedback_peak_max": float(np.max(np.asarray(result["feedback_peaks"]))),
            "gain_refresh_ms_mean": float(np.mean(np.asarray(result["gain_refresh_ms"]))),
            "published_gain_age_cycles_mean": float(np.nanmean(np.asarray(result["published_gain_age_cycles"]))),
            "queue_depth_max": int(result["queue_depth_max"]),
            "dropped_snapshot_count": int(result["dropped_snapshot_count"]),
            "first_gain_ready_cycle": result["first_gain_ready_cycle"],
            "completed_batch_count": int(result["completed_batch_count"]),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote reference -> {path}")


def _print_summary(result: dict[str, object]) -> None:
    errors = result["errors"]
    gain_norms = result["gain_norms"]
    feedback_peaks = result["feedback_peaks"]
    plan_times_ms = result["plan_times_ms"]
    tail_joint_spans = result["tail_joint_spans"]
    published_gain_age = np.asarray(result["published_gain_age_cycles"], dtype=np.float64)
    finite_gain_age = published_gain_age[np.isfinite(published_gain_age)]
    print(
        f"error: {errors[0]:.4f} -> {errors[-1]:.4f} m "
        f"min={np.min(errors):.4f} tail_std={np.std(errors[len(errors)//2:]):.5f}"
    )
    print(
        f"gain_norm: mean={np.mean(gain_norms):.3f} "
        f"peak={np.max(gain_norms):.3f} final={gain_norms[-1]:.3f}"
    )
    print(
        f"feedback_peak: max={np.max(feedback_peaks):.3f} Nm "
        f"tail_mean={np.mean(feedback_peaks[len(feedback_peaks)//2:]):.3f} Nm"
    )
    print(
        f"planning_ms: mean={np.mean(plan_times_ms):.2f} "
        f"max={np.max(plan_times_ms):.2f}"
    )
    if np.any(np.asarray(result["gain_refresh_ms"]) > 0.0):
        print(
            f"gain_refresh_ms: mean={np.mean(result['gain_refresh_ms']):.2f} "
            f"max={np.max(result['gain_refresh_ms']):.2f}"
        )
        print(
            f"subset_select_ms={np.mean(result['subset_select_ms']):.3f}  "
            f"snapshot_pack_ms={np.mean(result['snapshot_pack_ms']):.3f}  "
            f"gain_grad_ms={np.mean(result['gain_grad_ms']):.3f}  "
            f"gain_synth_ms={np.mean(result['gain_synth_ms']):.3f}"
        )
        print(
            f"exact_gain: first_gain_ready_cycle={result['first_gain_ready_cycle']} "
            f"queue_depth_max={result['queue_depth_max']} "
            f"dropped_snapshot_count={result['dropped_snapshot_count']} "
            f"completed_batch_count={result['completed_batch_count']} "
            f"rolling_window_fill_final={int(np.asarray(result['rolling_window_fill'])[-1])} "
            f"gain_age_mean="
            f"{(f'{np.mean(finite_gain_age):.2f}' if finite_gain_age.size else 'n/a')}"
        )
    print(
        f"joint_velocity: rms_mean={result['joint_velocity_rms_mean']:.3f} "
        f"abs_max={result['joint_velocity_abs_max']:.3f}"
    )
    print(
        "tail_joint_spans="
        + ", ".join(
            f"j{i + 1}:{span:.4f}" for i, span in enumerate(tail_joint_spans)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=tuple(BENCH_PRESETS), default="custom",
                        help="Common benchmark setup. Detailed flags override preset values.")
    parser.add_argument("--visual", action="store_true", help="Open MuJoCo viewer.")
    parser.add_argument("--realtime", action="store_true", help="Sleep to roughly match the 50 Hz control period when possible.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--timing-mode", choices=("immediate", "gazebo"), default="immediate")
    parser.add_argument("--retime-initial-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--desired-state-mode", choices=("base", "midpoint", "next"), default="base")
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable LFC feedback. Use --no-feedback for feedforward-only validation.")
    parser.add_argument("--gains", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute SB-MPC gains. Preset fast-feedforward disables this.")
    parser.add_argument("--clip-torque", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-velocity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gain-method", choices=("exact", "finite_difference"), default="exact")
    parser.add_argument("--gain-fd-epsilon", type=float, default=1e-3)
    parser.add_argument("--gain-fd-scheme", choices=("forward", "central"), default="forward")
    parser.add_argument("--gain-fd-samples", type=int, default=256)
    parser.add_argument("--phase0-async-probe", action="store_true",
                        help="Run the no-thread exact-gain background probe with separate gain refresh timing.")
    parser.add_argument("--background-gain-worker", action="store_true",
                        help="Run exact gain refresh in a same-process background thread.")
    parser.add_argument("--gain-refresh-cycles", type=int, default=1,
                        help="Probe-only: refresh the exact gain window every R control cycles.")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--control-points", type=int, default=8)
    parser.add_argument("--gain-samples-per-cycle", type=int, default=None,
                        help="Buffered-gain: how many MPPI samples to backprop per cycle (None = all).")
    parser.add_argument("--gain-buffer-size", type=int, default=None,
                        help="Buffered-gain: total accumulated samples before a K update (must be a multiple of --gain-samples-per-cycle).")
    parser.add_argument("--mjx-iterations", type=int, default=None)
    parser.add_argument("--mjx-ls-iterations", type=int, default=None)
    parser.add_argument("--mjx-tolerance", type=float, default=None)
    parser.add_argument("--jax-cache-dir", type=str, default=DEFAULT_JAX_CACHE_DIR,
                        help="Persistent JAX compilation cache directory. Use an empty string to disable.")
    parser.add_argument(
        "--emit-reference",
        type=str,
        default=None,
        help="Write MuJoCo baseline JSON (error trajectory + solve-time + stability) to this path.",
    )
    args = parser.parse_args()
    _apply_preset(args, sys.argv[1:])
    args.jax_cache_dir = _configure_jax_cache(args.jax_cache_dir)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    if args.gain_refresh_cycles <= 0:
        raise ValueError("--gain-refresh-cycles must be positive.")

    mjx_opts = {}
    if args.mjx_iterations is not None:
        mjx_opts["iterations"] = args.mjx_iterations
    if args.mjx_ls_iterations is not None:
        mjx_opts["ls_iterations"] = args.mjx_ls_iterations
    if args.mjx_tolerance is not None:
        mjx_opts["tolerance"] = args.mjx_tolerance
    args.mjx_opts = mjx_opts or None
    if args.phase0_async_probe and args.background_gain_worker:
        raise ValueError("--phase0-async-probe and --background-gain-worker are mutually exclusive.")
    if _exact_gain_context_enabled(args):
        if args.gain_method != "exact":
            raise ValueError("exact gain context modes require --gain-method exact.")
        if args.gain_samples_per_cycle is None or args.gain_buffer_size is None:
            raise ValueError(
                "exact gain context modes require both --gain-samples-per-cycle "
                "and --gain-buffer-size."
            )

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}", flush=True)
    if args.jax_cache_dir is not None:
        print(f"JAX compilation cache: {args.jax_cache_dir}", flush=True)
    print(
        f"preset={args.preset} "
        f"dt={args.dt} h={args.horizon} samples={args.samples} cp={args.control_points} "
        f"gain_method={args.gain_method} gains={args.gains} substeps={args.substeps} "
        f"timing={args.timing_mode} retime={args.retime_initial_state} "
        f"desired={args.desired_state_mode} feedback={args.feedback} "
        f"clip_torque={args.clip_torque} "
        f"clip_velocity={args.clip_velocity} "
        f"gK={args.gain_samples_per_cycle} gM={args.gain_buffer_size} "
        f"phase0_async_probe={args.phase0_async_probe} "
        f"background_gain_worker={args.background_gain_worker} "
        f"gain_refresh_cycles={args.gain_refresh_cycles} mjx={args.mjx_opts}",
        flush=True,
    )
    if args.visual:
        run_lfc_visual(args)
    else:
        result = run_lfc_validation(args)
        _print_summary(result)
        if args.emit_reference is not None:
            _emit_reference(args, result, args.emit_reference)


if __name__ == "__main__":
    main()
