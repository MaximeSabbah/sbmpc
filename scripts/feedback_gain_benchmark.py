#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc import PandaPickAndPlaceController
from sbmpc.panda_pick_and_place import (
    PandaPickAndPlacePlanner,
    Phase,
    make_panda_pick_and_place_config,
)


@dataclass(frozen=True)
class StageTiming:
    sample_ms: float
    rollout_ms: float
    update_ms: float
    gain_ms: float
    total_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    gains_enabled: bool
    gain_method: str
    gain_fd_scheme: str
    horizon: int
    num_parallel_computations: int
    num_control_points: int
    lambda_mpc: float
    std_mode: str
    std_value: float
    planner_step_ms_avg: float
    planner_step_ms_all: list[float]
    stage_timing_ms: StageTiming
    gain_norms: list[float]
    gain_norm_std_excluding_first: float
    max_diag_delta_vs_first: float
    max_sign_flips_vs_first: int
    spectral_radius: float
    unstable_modes: int
    clip_fraction: float
    max_ratio_to_limit: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Panda SB-MPC feedback-gain runtime and local stability "
            "for a fixed operating point."
        )
    )
    parser.add_argument("--name", default="case", help="Label used in the output.")
    parser.add_argument(
        "--mode",
        choices=("ff", "fd", "exact"),
        default="exact",
        help="Benchmark feedforward only, finite-difference gains, or exact gains.",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--control-points", type=int, default=4)
    parser.add_argument("--lambda-mpc", type=float, default=0.05)
    parser.add_argument(
        "--std-mode",
        choices=("scale", "absolute"),
        default="scale",
        help=(
            "'scale' multiplies Panda torque limits, 'absolute' uses the same "
            "standard deviation for every joint."
        ),
    )
    parser.add_argument("--std-value", type=float, default=0.05)
    parser.add_argument(
        "--gain-fd-scheme",
        choices=("forward", "central"),
        default="central",
    )
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of warm-started steady-state planner steps to measure.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of pretty-printed text.",
    )
    return parser.parse_args()


def _build_controller(args: argparse.Namespace) -> tuple[PandaPickAndPlacePlanner, PandaPickAndPlaceController]:
    planner = PandaPickAndPlacePlanner()
    gains_enabled = args.mode != "ff"
    gain_method = "exact" if args.mode == "exact" else "finite_difference"

    config = make_panda_pick_and_place_config(
        planner,
        visualize=False,
        gains=gains_enabled,
    )
    config.MPC.horizon = args.horizon
    config.MPC.num_parallel_computations = args.samples
    config.MPC.num_control_points = args.control_points
    config.MPC.lambda_mpc = float(args.lambda_mpc)
    if args.std_mode == "scale":
        config.MPC.std_dev_mppi = jnp.asarray(
            float(args.std_value) * planner.torque_limits,
            dtype=jnp.float32,
        )
    else:
        config.MPC.std_dev_mppi = jnp.full(
            (planner.nu,),
            float(args.std_value),
            dtype=jnp.float32,
        )
    config.MPC.gains = gains_enabled
    config.MPC.gain_method = gain_method
    config.MPC.gain_fd_scheme = args.gain_fd_scheme

    controller = PandaPickAndPlaceController(
        planner=planner,
        config=config,
        num_steps=args.num_steps,
    )
    return planner, controller


def _fixed_pregrasp_state(planner: PandaPickAndPlacePlanner) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(planner.phase_goal_q_map[Phase.PREGRASP], dtype=np.float32)
    v = np.zeros(planner.nv, dtype=np.float32)
    return q, v


def _profile_command(
    controller: PandaPickAndPlaceController,
    state: jax.Array,
    reference: jax.Array,
    *,
    num_steps: int,
) -> StageTiming:
    low = controller.controller
    rollout_gen = low.rollout_gen
    sampler = low.sampler
    gains_obj = low.gains_obj

    optimal_samples = sampler.optimal_samples
    gains = gains_obj.cur_gains

    sample_ms = 0.0
    rollout_ms = 0.0
    update_ms = 0.0
    gain_ms = 0.0
    for _ in range(num_steps):
        previous_optimal_samples = optimal_samples

        t0 = time.perf_counter()
        raw_samples_delta = sampler.sample_input_sequence(sampler.master_key)
        jax.block_until_ready(raw_samples_delta)
        t1 = time.perf_counter()

        samples, costs, gradients = rollout_gen.do_rollout(
            state,
            reference,
            previous_optimal_samples,
            raw_samples_delta,
            gains,
        )
        jax.block_until_ready(costs)
        t2 = time.perf_counter()

        optimal_samples = sampler.update(previous_optimal_samples, samples, costs)
        jax.block_until_ready(optimal_samples)
        t3 = time.perf_counter()

        if gains_obj.compute_gains and rollout_gen.gain_method == "finite_difference":
            new_gains = low._finite_difference_gains(
                state,
                reference,
                previous_optimal_samples,
                raw_samples_delta,
                samples,
                costs,
            )
        else:
            new_gains = gains_obj.gains_computation(costs, samples, gradients)
        jax.block_until_ready(new_gains)
        gains_obj.cur_gains = new_gains
        gains = new_gains
        t4 = time.perf_counter()

        sample_ms += 1e3 * (t1 - t0)
        rollout_ms += 1e3 * (t2 - t1)
        update_ms += 1e3 * (t3 - t2)
        gain_ms += 1e3 * (t4 - t3)

    sampler.optimal_samples = low._shift_guess(optimal_samples)
    jax.block_until_ready(sampler.optimal_samples)
    total_ms = sample_ms + rollout_ms + update_ms + gain_ms
    return StageTiming(
        sample_ms=sample_ms / num_steps,
        rollout_ms=rollout_ms / num_steps,
        update_ms=update_ms / num_steps,
        gain_ms=gain_ms / num_steps,
        total_ms=total_ms / num_steps,
    )


def _local_stability(
    controller: PandaPickAndPlaceController,
    state: np.ndarray,
    output_tau_ff: np.ndarray,
    output_gain: np.ndarray,
) -> tuple[float, int]:
    dt = float(controller.config.MPC.dt)
    state_jax = jnp.asarray(state, dtype=jnp.float32)
    tau_jax = jnp.asarray(output_tau_ff, dtype=jnp.float32)
    A, B = jax.jacfwd(
        lambda x, u: controller.model.integrate(x, u, dt),
        argnums=(0, 1),
    )(state_jax, tau_jax)
    A_np = np.asarray(A, dtype=np.float64)
    B_np = np.asarray(B, dtype=np.float64)
    K_np = np.asarray(output_gain, dtype=np.float64)
    eigvals = np.linalg.eigvals(A_np + B_np @ K_np)
    radius = float(np.max(np.abs(eigvals)))
    unstable = int(np.sum(np.abs(eigvals) > 1.0 + 1e-6))
    return radius, unstable


def _clip_stats(controller: PandaPickAndPlaceController, planner: PandaPickAndPlacePlanner, state: np.ndarray) -> tuple[float, float]:
    low = controller.controller
    rollout_gen = low.rollout_gen
    optimal_samples = low.sampler.optimal_samples
    raw_samples_delta = low.sampler.sample_input_sequence(low.sampler.master_key)

    if rollout_gen.config.MPC.smoothing == "Spline":
        control_vars_all = optimal_samples[rollout_gen.control_spline_indices, :] + raw_samples_delta
    else:
        control_vars_all = optimal_samples + raw_samples_delta

    reference = planner.reference_vec
    if reference.ndim == 1:
        reference = jnp.tile(reference, (rollout_gen.horizon + 1, 1))

    state_jax = jnp.asarray(state, dtype=jnp.float32)
    if rollout_gen.compute_exact_gains:
        (_, control_vars_used), _ = rollout_gen.rollout_sens_to_state(
            state_jax,
            reference,
            control_vars_all,
        )
    else:
        _, control_vars_used = rollout_gen.rollout_all(
            state_jax,
            reference,
            control_vars_all,
        )

    used = np.asarray(jax.block_until_ready(control_vars_used), dtype=np.float64)
    upper = np.asarray(rollout_gen.input_max_full_horizon, dtype=np.float64)[None, :, :]
    lower = np.asarray(rollout_gen.input_min_full_horizon, dtype=np.float64)[None, :, :]
    hits_limit = np.isclose(used, upper, atol=1e-5) | np.isclose(used, lower, atol=1e-5)
    clip_fraction = float(np.mean(hits_limit))
    max_ratio = float(np.max(np.abs(used) / np.maximum(np.abs(upper), 1e-6)))
    return clip_fraction, max_ratio


def _benchmark(args: argparse.Namespace) -> BenchmarkResult:
    planner, controller = _build_controller(args)
    q, v = _fixed_pregrasp_state(planner)
    state = np.concatenate([q, v]).astype(np.float32)

    # Warm up and initialize the shifted nominal sequence before taking steady-state timings.
    controller.step(q, v, Phase.PREGRASP, reset_guess=True)

    planner_times: list[float] = []
    gain_mats: list[np.ndarray] = []
    tau_seq: list[np.ndarray] = []
    for _ in range(args.repeats):
        out = controller.step(q, v, Phase.PREGRASP)
        planner_times.append(float(out.diagnostics.planning_time_ms))
        gain_mats.append(np.asarray(out.K, dtype=np.float64))
        tau_seq.append(np.asarray(out.tau_ff, dtype=np.float64))

    controller.step(q, v, Phase.PREGRASP, reset_guess=True)
    controller.planner.set_phase(Phase.PREGRASP)
    reference = controller.planner.reference_vec
    stage_timing = _profile_command(
        controller,
        jnp.asarray(state, dtype=jnp.float32),
        reference,
        num_steps=args.num_steps,
    )

    base_gain = gain_mats[0]
    gain_norms = [float(np.linalg.norm(g)) for g in gain_mats]
    if len(gain_mats) > 1:
        max_diag_delta = max(
            float(np.max(np.abs(np.diag(g - base_gain)))) for g in gain_mats[1:]
        )
        max_sign_flips = max(
            int(np.sum(np.sign(np.diag(g)) != np.sign(np.diag(base_gain))))
            for g in gain_mats[1:]
        )
        gain_norm_std = float(np.std(gain_norms[1:]))
    else:
        max_diag_delta = 0.0
        max_sign_flips = 0
        gain_norm_std = 0.0

    spectral_radius, unstable_modes = _local_stability(
        controller,
        state,
        tau_seq[-1],
        gain_mats[-1],
    )
    clip_fraction, max_ratio_to_limit = _clip_stats(controller, planner, state)
    gains_enabled = args.mode != "ff"
    gain_method = "exact" if args.mode == "exact" else "finite_difference"

    return BenchmarkResult(
        name=args.name,
        gains_enabled=gains_enabled,
        gain_method=gain_method,
        gain_fd_scheme=args.gain_fd_scheme,
        horizon=args.horizon,
        num_parallel_computations=args.samples,
        num_control_points=args.control_points,
        lambda_mpc=float(args.lambda_mpc),
        std_mode=args.std_mode,
        std_value=float(args.std_value),
        planner_step_ms_avg=float(np.mean(planner_times)),
        planner_step_ms_all=planner_times,
        stage_timing_ms=stage_timing,
        gain_norms=gain_norms,
        gain_norm_std_excluding_first=gain_norm_std,
        max_diag_delta_vs_first=max_diag_delta,
        max_sign_flips_vs_first=max_sign_flips,
        spectral_radius=spectral_radius,
        unstable_modes=unstable_modes,
        clip_fraction=clip_fraction,
        max_ratio_to_limit=max_ratio_to_limit,
    )


def _print_text(result: BenchmarkResult) -> None:
    data = asdict(result)
    stage = data.pop("stage_timing_ms")
    print(f"name: {data.pop('name')}")
    for key, value in data.items():
        print(f"{key}: {value}")
    print("stage_timing_ms:")
    for key, value in stage.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = _parse_args()
    result = _benchmark(args)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        _print_text(result)


if __name__ == "__main__":
    main()
