"""Benchmark exact-gain work split by samples and state-gradient directions.

This is intentionally narrower than bench_lfc.py: it reuses the pregrasp setup
but times only the gain sensitivity kernels after warmup. The goal is to decide
whether a live controller can spend leftover 50 Hz budget on resumable gain
microtasks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all


DEFAULT_JAX_CACHE_DIR = ".jax_cache"


@dataclass(frozen=True)
class TimingSummary:
    label: str
    sample_count: int
    direction_count: int | None
    mean_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    max_ms: float
    timings_ms: list[float]


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    if any(item <= 0 for item in result):
        raise ValueError("all values must be positive")
    return result


def configure_jax_cache(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def build_sim(args: argparse.Namespace):
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.dt = args.dt
    config.MPC.horizon = args.horizon
    config.MPC.num_parallel_computations = args.samples
    config.MPC.num_control_points = args.control_points
    config.MPC.gain_samples_per_cycle = args.max_gain_samples
    config.MPC.gain_buffer_size = args.gain_buffer_size
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
        warm_start_gains=False,
    )
    return planner, config, sim


def reset_guess(sim, planner, config, state) -> None:
    sim.controller.sampler.optimal_samples = planner.nominal_torque_sequence_from_state(
        state,
        config.MPC.horizon,
        config.MPC.dt,
    )


def capture_context(sim, planner, config):
    state = sim.current_state_vec()
    reset_guess(sim, planner, config, state)
    output = sim.controller.command(
        state,
        sim.const_reference,
        num_steps=1,
        update_gains=False,
        capture_gain_context=True,
    )
    jax.block_until_ready(output)
    context = sim.controller._phase0_pending_context
    if context is None:
        raise RuntimeError("foreground command did not capture an exact-gain context")
    return context


def sample_indices_for_costs(costs, sample_count: int):
    total = int(costs.shape[0])
    if sample_count >= total:
        return jnp.arange(total, dtype=jnp.int32)
    if sample_count == 1:
        return jnp.zeros((1,), dtype=jnp.int32)
    _, top_non_nominal = jax.lax.top_k(-costs[1:], sample_count - 1)
    return jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), top_non_nominal.astype(jnp.int32) + 1),
        axis=0,
    )


def make_direction_gradient_kernel(rollout_gen):
    def one_sample_gradient(initial_state, reference, control_variables, basis_dirs):
        def cost_from_state(state):
            cost, _ = rollout_gen.rollout_single(state, reference, control_variables)
            return cost

        return jax.vmap(
            lambda tangent: jax.jvp(cost_from_state, (initial_state,), (tangent,))[1]
        )(basis_dirs)

    return jax.jit(
        jax.vmap(one_sample_gradient, in_axes=(None, None, 0, None), out_axes=0),
        device=rollout_gen.device,
    )


def summarize(label: str, sample_count: int, direction_count: int | None, timings):
    values = np.asarray(timings, dtype=np.float64)
    return TimingSummary(
        label=label,
        sample_count=int(sample_count),
        direction_count=None if direction_count is None else int(direction_count),
        mean_ms=float(np.mean(values)),
        min_ms=float(np.min(values)),
        p50_ms=float(np.quantile(values, 0.50)),
        p90_ms=float(np.quantile(values, 0.90)),
        max_ms=float(np.max(values)),
        timings_ms=[float(value) for value in values.tolist()],
    )


def time_call(fn, *, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        jax.block_until_ready(fn())
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        timings.append(1000.0 * (time.perf_counter() - start))
    return timings


def benchmark(args: argparse.Namespace) -> list[TimingSummary]:
    configure_jax_cache(args.jax_cache_dir)
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    planner, config, sim = build_sim(args)
    context = capture_context(sim, planner, config)
    controller = sim.controller
    rollout_gen = controller.rollout_gen
    direction_kernel = make_direction_gradient_kernel(rollout_gen)
    basis = jnp.eye(rollout_gen.model.nx, dtype=rollout_gen.dtype_general)
    results: list[TimingSummary] = []

    for sample_count in args.sample_counts:
        indices = sample_indices_for_costs(context.nominal_costs, sample_count)
        jax.block_until_ready(indices)
        snapshot = controller._pack_exact_gain_snapshot(context, indices)
        jax.block_until_ready(snapshot.control_variables)

        def full_gradient():
            return controller._process_exact_gain_snapshot(snapshot).gradients

        timings = time_call(
            full_gradient,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        results.append(
            summarize("current_full_gradient", sample_count, None, timings)
        )

    for sample_count in args.direction_sample_counts:
        indices = sample_indices_for_costs(context.nominal_costs, sample_count)
        jax.block_until_ready(indices)
        snapshot = controller._pack_exact_gain_snapshot(context, indices)
        jax.block_until_ready(snapshot.control_variables)
        for direction_count in args.direction_counts:
            basis_dirs = basis[:direction_count]

            def direction_gradient():
                return direction_kernel(
                    snapshot.state,
                    snapshot.reference,
                    snapshot.control_variables,
                    basis_dirs,
                )

            timings = time_call(
                direction_gradient,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            results.append(
                summarize(
                    "direction_chunk_gradient",
                    sample_count,
                    direction_count,
                    timings,
                )
            )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--control-points", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--gain-buffer-size", type=int, default=512)
    parser.add_argument("--sample-counts", type=parse_int_list, default=parse_int_list("1,2,4,8,16,32,64,128"))
    parser.add_argument("--direction-sample-counts", type=parse_int_list, default=parse_int_list("1,2,4"))
    parser.add_argument("--direction-counts", type=parse_int_list, default=parse_int_list("1,2,4,7,14"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jax-cache-dir", default=DEFAULT_JAX_CACHE_DIR)
    args = parser.parse_args(argv)
    args.max_gain_samples = max(args.sample_counts + args.direction_sample_counts)

    results = benchmark(args)
    for result in results:
        print(
            f"{result.label:24s} samples={result.sample_count:3d} "
            f"dirs={str(result.direction_count):>4s} "
            f"mean={result.mean_ms:7.3f}ms p50={result.p50_ms:7.3f}ms "
            f"p90={result.p90_ms:7.3f}ms max={result.max_ms:7.3f}ms"
        )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
