"""MuJoCo-side validation of SB-MPC gains applied through the LFC law.

This intentionally differs from bench_controller.py: it applies

    tau = tau_ff + K_lfc @ (x_desired - x_measured)

inside each control period, with K_lfc using the same sign convention as the
ROS linear_feedback_controller bridge. The simulation only advances after the
controller output is ready, so this isolates gain correctness from ROS/Gazebo
wall-clock scheduling.
"""

from __future__ import annotations

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


def _step_durations(planner, config):
    return planner.step_durations(
        config.MPC.horizon,
        config.MPC.dt,
        config.MPC.dt_schedule,
    )


def _reset_guess(sim, planner, config, state) -> None:
    sim.controller.sampler.optimal_samples = planner.nominal_torque_sequence_from_state(
        state,
        config.MPC.horizon,
        _step_durations(planner, config),
    )


def _ee_error(planner, state: np.ndarray) -> float:
    q = jnp.asarray(state[: planner.nq], dtype=jnp.float32)
    ee_pos = np.asarray(jax.block_until_ready(planner.ee_position(q)))
    return float(np.linalg.norm(ee_pos - np.asarray(planner.goal_pos)))


def run_lfc_validation(args: argparse.Namespace) -> dict[str, object]:
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.dt_schedule = None
    config.MPC.gain_method = args.gain_method
    config.MPC.gain_fd_epsilon = args.gain_fd_epsilon
    config.MPC.gain_fd_scheme = args.gain_fd_scheme
    config.MPC.gain_fd_num_samples = args.gain_fd_samples
    config.MPC.horizon = args.horizon
    config.MPC.num_parallel_computations = args.samples
    config.MPC.num_control_points = args.control_points
    config.MPC.initial_guess = planner.nominal_torque_sequence(
        config.MPC.horizon,
        _step_durations(planner, config),
    )

    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )

    low_dt = config.MPC.dt / args.substeps
    errors: list[float] = []
    gain_norms: list[float] = []
    feedback_peaks: list[float] = []
    plan_times_ms: list[float] = []
    q_history: list[np.ndarray] = []
    v_history: list[np.ndarray] = []

    # Compile before recording metrics.
    state = sim.current_state_vec()
    _reset_guess(sim, planner, config, state)
    warm = sim.controller.command(state, sim.const_reference, num_steps=1)
    jax.block_until_ready(warm)
    jax.block_until_ready(sim.controller.gains)

    torque_limits = np.asarray(planner.torque_limits, dtype=np.float64)
    for _ in range(args.steps):
        desired = np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)
        _reset_guess(sim, planner, config, jnp.asarray(desired, dtype=jnp.float32))

        start = time.perf_counter()
        input_sequence = sim.controller.command(
            jnp.asarray(desired, dtype=jnp.float32),
            sim.const_reference,
            num_steps=1,
        )
        jax.block_until_ready(input_sequence)
        K_sbmpc = np.asarray(jax.block_until_ready(sim.controller.gains), dtype=np.float64)
        tau_ff = np.asarray(input_sequence[0], dtype=np.float64)
        plan_times_ms.append((time.perf_counter() - start) * 1000.0)

        # Same convention as the ROS bridge: SB-MPC exposes du/dx_measured,
        # while LFC multiplies (desired - measured).
        K_lfc = -K_sbmpc
        feedback_peak = 0.0
        for _substep in range(args.substeps):
            measured = np.asarray(
                jax.block_until_ready(sim.current_state_vec()),
                dtype=np.float64,
            )
            feedback = K_lfc @ (desired - measured)
            feedback_peak = max(
                feedback_peak,
                float(np.max(np.abs(feedback), initial=0.0)),
            )
            tau = np.clip(tau_ff + feedback, -torque_limits, torque_limits)
            sim.current_state = sim.model.integrate_sim(
                sim.current_state,
                jnp.asarray(tau, dtype=jnp.float32),
                low_dt,
            )

        state_np = np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)
        errors.append(_ee_error(planner, state_np))
        gain_norms.append(float(np.linalg.norm(K_sbmpc)))
        feedback_peaks.append(feedback_peak)
        q_history.append(state_np[: planner.nq])
        v_history.append(state_np[planner.nq : planner.nq + planner.nv])

    q = np.asarray(q_history, dtype=np.float64)
    v = np.asarray(v_history, dtype=np.float64)
    tail = slice(len(q) // 2, None)
    return {
        "errors": np.asarray(errors, dtype=np.float64),
        "gain_norms": np.asarray(gain_norms, dtype=np.float64),
        "feedback_peaks": np.asarray(feedback_peaks, dtype=np.float64),
        "plan_times_ms": np.asarray(plan_times_ms, dtype=np.float64),
        "tail_joint_spans": np.ptp(q[tail], axis=0),
        "joint_velocity_abs_max": float(np.max(np.abs(v), initial=0.0)),
        "joint_velocity_rms_mean": float(np.mean(np.sqrt(np.mean(v * v, axis=1)))),
    }


def _print_summary(result: dict[str, object]) -> None:
    errors = result["errors"]
    gain_norms = result["gain_norms"]
    feedback_peaks = result["feedback_peaks"]
    plan_times_ms = result["plan_times_ms"]
    tail_joint_spans = result["tail_joint_spans"]
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
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--gain-method", choices=("exact", "finite_difference", "local_lqr"), default="exact")
    parser.add_argument("--gain-fd-epsilon", type=float, default=1e-3)
    parser.add_argument("--gain-fd-scheme", choices=("forward", "central"), default="forward")
    parser.add_argument("--gain-fd-samples", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--control-points", type=int, default=8)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive.")

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(
        f"h={args.horizon} samples={args.samples} cp={args.control_points} "
        f"gain_method={args.gain_method} substeps={args.substeps}"
    )
    result = run_lfc_validation(args)
    _print_summary(result)


if __name__ == "__main__":
    main()
