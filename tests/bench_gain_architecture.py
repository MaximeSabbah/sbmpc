"""Benchmark the rolling exact-gain architecture for 50 Hz LFC deployment.

This benchmark is deliberately between ``bench_gain_microbatch.py`` and
``bench_lfc.py``:

* foreground SB-MPC planning is measured as the publish-critical path;
* exact-gain work reuses the foreground costs and first-step control deltas;
* a rolling gain window publishes once full, then publishes again after each
  new chunk overwrites the oldest chunk;
* closed-loop MuJoCo/LFC metrics are reported with the same accuracy and
  stability vocabulary used by ``bench_lfc.py``.

The benchmark does not change the production controller. It is a design probe
for choosing the chunk size and deciding whether fixed-size chunks are enough
or a finite-set adaptive scheduler is worth implementing later.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.controller.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all
from sbmpc.solvers import ProcessedGainBatch
from tests.bench_lfc import (
    _apply_lfc_control,
    _desired_state_for_control,
    _ee_error,
    _joint_vel_hf_energy,
    _reset_guess,
)


DEFAULT_JAX_CACHE_DIR = ".jax_cache"

GATE_ERROR_FINAL_M = 1.5e-3
GATE_FEEDBACK_PEAK_NM = 50.0
GATE_JOINT_VEL_MAX = 3.0
GATE_HF_ENERGY = 0.13


@dataclass(frozen=True)
class ForegroundPlan:
    tau_ff: np.ndarray
    context: Any
    reseed_ms: float
    command_ms: float
    foreground_ms: float


@dataclass(frozen=True)
class Candidate:
    label: str
    chunks: tuple[int, ...]
    adaptive: bool = False


def parse_int_list(value: str) -> list[int]:
    if value.strip() == "":
        return []
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item <= 0 for item in result):
        raise ValueError("all chunk sizes must be positive")
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


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


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


def make_full_state_gradient_kernel(rollout_gen):
    basis = jnp.eye(rollout_gen.model.nx, dtype=rollout_gen.dtype_general)

    def one_sample_gradient(initial_state, reference, control_variables):
        def cost_from_state(state):
            cost, _ = rollout_gen.rollout_single(state, reference, control_variables)
            return cost

        return jax.vmap(
            lambda tangent: jax.jvp(cost_from_state, (initial_state,), (tangent,))[1]
        )(basis)

    return jax.jit(
        jax.vmap(one_sample_gradient, in_axes=(None, None, 0), out_axes=0),
        device=rollout_gen.device,
    )


class VariableRollingGainWindow:
    """Rolling exact-gain buffer that can append fixed or adaptive chunk sizes."""

    def __init__(self, capacity: int, nu: int, nx: int, dtype):
        self.capacity = int(capacity)
        self.costs = jnp.zeros((self.capacity,), dtype=dtype)
        self.delta_u0 = jnp.zeros((self.capacity, nu), dtype=dtype)
        self.gradients = jnp.zeros((self.capacity, nx), dtype=dtype)
        self.fill = 0
        self.cursor = 0
        self.append_count = 0
        self.sample_count = 0

    def reset(self) -> None:
        self.costs = jnp.zeros_like(self.costs)
        self.delta_u0 = jnp.zeros_like(self.delta_u0)
        self.gradients = jnp.zeros_like(self.gradients)
        self.fill = 0
        self.cursor = 0
        self.append_count = 0
        self.sample_count = 0

    def _write_ring(self, target, values):
        batch_size = int(values.shape[0])
        start = self.cursor
        end = start + batch_size
        if end <= self.capacity:
            return target.at[start:end].set(values)
        split = self.capacity - start
        target = target.at[start:].set(values[:split])
        remaining = batch_size - split
        return target.at[:remaining].set(values[split:])

    def append(self, batch: ProcessedGainBatch) -> None:
        batch_size = int(batch.costs.shape[0])
        if batch_size <= 0:
            raise ValueError("cannot append an empty gain batch")
        if batch_size > self.capacity:
            raise ValueError(
                f"batch size {batch_size} exceeds gain window capacity {self.capacity}"
            )
        self.costs = self._write_ring(self.costs, batch.costs)
        self.delta_u0 = self._write_ring(self.delta_u0, batch.delta_u0)
        self.gradients = self._write_ring(self.gradients, batch.gradients)
        self.cursor = (self.cursor + batch_size) % self.capacity
        self.fill = min(self.capacity, self.fill + batch_size)
        self.append_count += 1
        self.sample_count += batch_size

    def ready_to_publish(self) -> bool:
        return self.fill >= self.capacity

    def compute_gain(self, gains_obj):
        return gains_obj.gains_computation(
            self.costs,
            self.delta_u0[:, jnp.newaxis, :],
            self.gradients,
        )


class RollingGainProbe:
    def __init__(self, controller, buffer_size: int, chunks: tuple[int, ...]):
        self.controller = controller
        self.rollout_gen = controller.rollout_gen
        self.window = VariableRollingGainWindow(
            buffer_size,
            self.rollout_gen.model.nu,
            self.rollout_gen.model.nx,
            self.rollout_gen.dtype_general,
        )
        self.gradient_kernel = make_full_state_gradient_kernel(self.rollout_gen)
        self.chunks = tuple(sorted(set(int(chunk) for chunk in chunks)))
        self.current_gain = jnp.zeros_like(controller.gains)
        self.last_published_gain = None
        self.publish_cycles: list[int] = []
        self.chunk_estimate_ms: dict[int, float] = {}
        self.synth_estimate_ms = 0.0

    def reset(self) -> None:
        self.window.reset()
        self.current_gain = jnp.zeros_like(self.current_gain)
        self.last_published_gain = None
        self.publish_cycles.clear()
        self.controller._set_current_gains(self.current_gain)

    def _compute_batch(
        self,
        context,
        chunk_size: int,
    ) -> tuple[ProcessedGainBatch, dict[str, float]]:
        t_select = time.perf_counter()
        sample_indices = sample_indices_for_costs(context.nominal_costs, chunk_size)
        jax.block_until_ready(sample_indices)
        subset_select_ms = (time.perf_counter() - t_select) * 1000.0

        t_pack = time.perf_counter()
        snapshot = self.controller._pack_exact_gain_snapshot(context, sample_indices)
        jax.block_until_ready(snapshot.costs)
        jax.block_until_ready(snapshot.delta_u0)
        jax.block_until_ready(snapshot.control_variables)
        snapshot_pack_ms = (time.perf_counter() - t_pack) * 1000.0

        t_grad = time.perf_counter()
        gradients = self.gradient_kernel(
            snapshot.state,
            snapshot.reference,
            snapshot.control_variables,
        )
        jax.block_until_ready(gradients)
        gain_grad_ms = (time.perf_counter() - t_grad) * 1000.0

        batch = ProcessedGainBatch(
            cycle_id=context.cycle_id,
            sample_indices=sample_indices,
            costs=snapshot.costs,
            delta_u0=snapshot.delta_u0,
            gradients=gradients,
        )
        timings = {
            "subset_select_ms": subset_select_ms,
            "snapshot_pack_ms": snapshot_pack_ms,
            "gain_grad_ms": gain_grad_ms,
        }
        return batch, timings

    def warmup(self, context, warmups: int) -> None:
        if warmups <= 0:
            return
        for chunk in self.chunks:
            # First call compiles the fixed-shape chunk. Do not let compilation
            # poison the runtime estimate used by the adaptive scheduler.
            batch, _ = self._compute_batch(context, chunk)
            jax.block_until_ready(batch.gradients)
            timing_samples = []
            for _ in range(warmups):
                batch, timings = self._compute_batch(context, chunk)
                timing_samples.append(
                    timings["subset_select_ms"]
                    + timings["snapshot_pack_ms"]
                    + timings["gain_grad_ms"]
                )

            self.window.reset()
            while self.window.fill < self.window.capacity:
                self.window.append(batch)

            warm_gain = self.window.compute_gain(self.controller.gains_obj)
            jax.block_until_ready(warm_gain)
            synth_samples = []
            for _ in range(warmups):
                t_synth = time.perf_counter()
                gain = self.window.compute_gain(self.controller.gains_obj)
                jax.block_until_ready(gain)
                synth_samples.append((time.perf_counter() - t_synth) * 1000.0)

            self.synth_estimate_ms = max(self.synth_estimate_ms, float(np.mean(synth_samples)))
            self.chunk_estimate_ms[chunk] = (
                float(np.mean(timing_samples)) + float(np.mean(synth_samples))
            )
            self.window.reset()
        self.reset()

    def process_context(self, context, chunk_size: int, step_idx: int) -> dict[str, Any]:
        batch, timings = self._compute_batch(context, chunk_size)
        self.window.append(batch)

        gain_published = False
        gain_synth_ms = 0.0
        gain_delta_norm = float("nan")
        if self.window.ready_to_publish():
            t_synth = time.perf_counter()
            new_gain = self.window.compute_gain(self.controller.gains_obj)
            jax.block_until_ready(new_gain)
            gain_synth_ms = (time.perf_counter() - t_synth) * 1000.0
            if self.last_published_gain is not None:
                gain_delta_norm = float(
                    np.linalg.norm(np.asarray(new_gain - self.last_published_gain))
                )
            self.current_gain = new_gain
            self.last_published_gain = new_gain
            self.controller._set_current_gains(new_gain)
            self.publish_cycles.append(step_idx)
            gain_published = True

        gain_work_ms = (
            timings["subset_select_ms"]
            + timings["snapshot_pack_ms"]
            + timings["gain_grad_ms"]
            + gain_synth_ms
        )
        return {
            **timings,
            "gain_synth_ms": gain_synth_ms,
            "gain_work_ms": gain_work_ms,
            "gain_published": gain_published,
            "gain_delta_norm": gain_delta_norm,
            "window_fill": int(self.window.fill),
            "total_gradient_samples": int(self.window.sample_count),
        }


def _consume_phase0_context(controller):
    context = controller._phase0_pending_context
    controller._phase0_pending_context = None
    if context is None:
        raise RuntimeError("foreground command did not capture an exact-gain context")
    return context


def _build_sim(args: argparse.Namespace):
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.gains = True
    config.MPC.gain_method = "exact"
    config.MPC.dt = args.dt
    config.MPC.horizon = args.horizon
    config.MPC.num_parallel_computations = args.samples
    config.MPC.num_control_points = args.control_points
    config.MPC.gain_samples_per_cycle = 1
    config.MPC.gain_buffer_size = args.gain_buffer_size
    config.robot.mjx_opts = args.mjx_opts
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


def _plan_foreground(
    sim,
    planner,
    config,
    state: np.ndarray,
    *,
    reseed: bool,
) -> ForegroundPlan:
    reseed_ms = 0.0
    state_jax = jnp.asarray(state, dtype=jnp.float32)
    if reseed:
        reseed_start = time.perf_counter()
        _reset_guess(sim, planner, config, state_jax)
        jax.block_until_ready(sim.controller.sampler.optimal_samples)
        reseed_ms = (time.perf_counter() - reseed_start) * 1000.0

    start = time.perf_counter()
    input_sequence = sim.controller.command(
        state_jax,
        sim.const_reference,
        num_steps=1,
        update_gains=False,
        capture_gain_context=True,
    )
    jax.block_until_ready(input_sequence)
    command_ms = (time.perf_counter() - start) * 1000.0
    context = _consume_phase0_context(sim.controller)
    return ForegroundPlan(
        tau_ff=np.asarray(input_sequence[0], dtype=np.float64),
        context=context,
        reseed_ms=reseed_ms,
        command_ms=command_ms,
        foreground_ms=reseed_ms + command_ms,
    )


def _compile_runtime(
    sim,
    planner,
    config,
    args: argparse.Namespace,
    probe: RollingGainProbe,
) -> None:
    state = np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)
    print("[setup] compiling foreground rollout and gain chunks...", flush=True)
    plan = _plan_foreground(sim, planner, config, state, reseed=True)
    probe.warmup(plan.context, args.warmups)

    # The measured loop will eventually run foreground planning after a
    # nonzero gain has been published. Compile that shape/state here so the
    # first rolling-window publish does not pollute the controller timing.
    warm_chunk = max(probe.chunks)
    while probe.window.fill < probe.window.capacity:
        probe.process_context(plan.context, warm_chunk, step_idx=-1)
    _plan_foreground(sim, planner, config, state, reseed=True)

    sim.controller.reset_phase0_exact_gain_probe(reset_published_gain=True)

    print("[setup] compiling LFC/MuJoCo runtime path...", flush=True)
    warm_state = sim.current_state
    zero_tau = np.zeros(probe.current_gain.shape[0], dtype=np.float64)
    zero_control = {
        "tau_ff": zero_tau,
        "K_lfc": np.zeros((probe.current_gain.shape[0], state.size), dtype=np.float64),
        "desired": state,
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
    _desired_state_for_control(sim, args, state, zero_tau, config.MPC.dt)
    jax.block_until_ready(sim.current_state_vec())
    sim.current_state = warm_state
    probe.reset()


def _choose_adaptive_chunk(
    probe: RollingGainProbe,
    chunks: tuple[int, ...],
    budget_ms: float,
    allow_overrun: bool,
) -> int | None:
    for chunk in sorted(chunks, reverse=True):
        if probe.chunk_estimate_ms.get(chunk, float("inf")) <= budget_ms:
            return chunk
    if allow_overrun and chunks:
        return min(chunks)
    return None


def _empty_gain_result() -> dict[str, Any]:
    return {
        "subset_select_ms": 0.0,
        "snapshot_pack_ms": 0.0,
        "gain_grad_ms": 0.0,
        "gain_synth_ms": 0.0,
        "gain_work_ms": 0.0,
        "gain_published": False,
        "gain_delta_norm": float("nan"),
        "window_fill": 0,
        "total_gradient_samples": 0,
    }


def _reseed_this_step(args: argparse.Namespace, step_idx: int) -> bool:
    if args.reseed_policy == "every":
        return True
    if args.reseed_policy == "periodic":
        return step_idx % max(1, int(args.reseed_period)) == 0
    if args.reseed_policy == "initial":
        return step_idx == 0
    if args.reseed_policy == "none":
        return False
    raise ValueError(f"unsupported reseed policy: {args.reseed_policy!r}")


def _run_candidate(args: argparse.Namespace, candidate: Candidate) -> dict[str, Any]:
    planner, config, sim = _build_sim(args)
    probe = RollingGainProbe(sim.controller, args.gain_buffer_size, candidate.chunks)
    _compile_runtime(sim, planner, config, args, probe)

    previous_control: dict[str, np.ndarray] | None = None
    previous_publish_ms = 0.0
    last_publish_cycle: int | None = None
    first_gain_ready_cycle: int | None = None

    errors: list[float] = []
    feedback_peaks: list[float] = []
    foreground_ms: list[float] = []
    foreground_command_ms: list[float] = []
    reseed_ms: list[float] = []
    subset_select_ms: list[float] = []
    snapshot_pack_ms: list[float] = []
    gain_work_ms: list[float] = []
    gain_grad_ms: list[float] = []
    gain_synth_ms: list[float] = []
    cycle_compute_ms: list[float] = []
    gain_norms_used: list[float] = []
    published_gain_norms: list[float] = []
    gain_delta_norms: list[float] = []
    gain_age_cycles: list[float] = []
    chosen_chunks: list[int] = []
    window_fill: list[int] = []
    total_gradient_samples: list[int] = []
    q_history: list[np.ndarray] = []
    v_history: list[np.ndarray] = []

    target_ms = float(args.target_ms)
    for step_idx in range(args.steps):
        if args.timing_mode == "gazebo":
            pre_timer_duration = max(0.0, config.MPC.dt - previous_publish_ms * 1e-3)
            _apply_lfc_control(
                sim,
                planner,
                previous_control,
                pre_timer_duration,
                args.substeps,
                config.MPC.dt,
                clip_torque=args.clip_torque,
                clip_velocity=args.clip_velocity,
            )

        planning_state = np.asarray(
            jax.block_until_ready(sim.current_state_vec()),
            dtype=np.float64,
        )
        plan = _plan_foreground(
            sim,
            planner,
            config,
            planning_state,
            reseed=_reseed_this_step(args, step_idx),
        )
        gain_for_publish = np.asarray(jax.block_until_ready(probe.current_gain), dtype=np.float64)

        feedback_peak = 0.0
        if args.timing_mode == "gazebo":
            feedback_peak = _apply_lfc_control(
                sim,
                planner,
                previous_control,
                plan.foreground_ms * 1e-3,
                args.substeps,
                config.MPC.dt,
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
            plan.tau_ff,
            config.MPC.dt,
        )
        current_control = {
            "tau_ff": plan.tau_ff,
            "K_lfc": -gain_for_publish if args.feedback else np.zeros_like(gain_for_publish),
            "desired": desired,
        }

        if args.timing_mode == "immediate":
            feedback_peak = _apply_lfc_control(
                sim,
                planner,
                current_control,
                config.MPC.dt,
                args.substeps,
                config.MPC.dt,
                clip_torque=args.clip_torque,
                clip_velocity=args.clip_velocity,
            )

        if candidate.adaptive:
            budget_ms = target_ms - plan.foreground_ms - args.safety_margin_ms
            chunk = _choose_adaptive_chunk(
                probe,
                candidate.chunks,
                budget_ms,
                args.adaptive_allow_overrun,
            )
        else:
            chunk = candidate.chunks[0]

        if chunk is None:
            gain_result = _empty_gain_result()
            chosen_chunks.append(0)
        else:
            gain_result = probe.process_context(plan.context, chunk, step_idx)
            chosen_chunks.append(int(chunk))

        if gain_result["gain_published"]:
            last_publish_cycle = step_idx
            if first_gain_ready_cycle is None:
                first_gain_ready_cycle = step_idx
            published_gain_norms.append(float(np.linalg.norm(np.asarray(probe.current_gain))))
            if np.isfinite(gain_result["gain_delta_norm"]):
                gain_delta_norms.append(float(gain_result["gain_delta_norm"]))

        age = float("nan") if last_publish_cycle is None else float(step_idx - last_publish_cycle)
        state_np = np.asarray(jax.block_until_ready(sim.current_state_vec()), dtype=np.float64)

        errors.append(_ee_error(planner, state_np))
        feedback_peaks.append(feedback_peak)
        foreground_ms.append(plan.foreground_ms)
        foreground_command_ms.append(plan.command_ms)
        reseed_ms.append(plan.reseed_ms)
        subset_select_ms.append(gain_result["subset_select_ms"])
        snapshot_pack_ms.append(gain_result["snapshot_pack_ms"])
        gain_work_ms.append(gain_result["gain_work_ms"])
        gain_grad_ms.append(gain_result["gain_grad_ms"])
        gain_synth_ms.append(gain_result["gain_synth_ms"])
        cycle_compute_ms.append(plan.foreground_ms + gain_result["gain_work_ms"])
        gain_norms_used.append(float(np.linalg.norm(gain_for_publish)))
        gain_age_cycles.append(age)
        window_fill.append(int(gain_result["window_fill"]))
        total_gradient_samples.append(int(gain_result["total_gradient_samples"]))
        q_history.append(state_np[: planner.nq])
        v_history.append(state_np[planner.nq : planner.nq + planner.nv])

        previous_control = current_control
        previous_publish_ms = plan.foreground_ms

        if args.print_every > 0 and step_idx % args.print_every == 0:
            chunk_label = "skip" if chunk is None else str(chunk)
            print(
                f"{candidate.label} step={step_idx:04d} "
                f"err={errors[-1]:.4f}m "
                f"fg={plan.foreground_ms:.2f}ms "
                f"seed={plan.reseed_ms:.2f}ms "
                f"gain={gain_result['gain_work_ms']:.2f}ms "
                f"cycle={cycle_compute_ms[-1]:.2f}ms "
                f"chunk={chunk_label} "
                f"fill={window_fill[-1]} "
                f"|K|={gain_norms_used[-1]:.3f}",
                flush=True,
            )

    return _summarize_candidate(
        args,
        candidate,
        {
            "errors": errors,
            "feedback_peaks": feedback_peaks,
            "foreground_ms": foreground_ms,
            "foreground_command_ms": foreground_command_ms,
            "reseed_ms": reseed_ms,
            "subset_select_ms": subset_select_ms,
            "snapshot_pack_ms": snapshot_pack_ms,
            "gain_work_ms": gain_work_ms,
            "gain_grad_ms": gain_grad_ms,
            "gain_synth_ms": gain_synth_ms,
            "cycle_compute_ms": cycle_compute_ms,
            "gain_norms_used": gain_norms_used,
            "published_gain_norms": published_gain_norms,
            "gain_delta_norms": gain_delta_norms,
            "gain_age_cycles": gain_age_cycles,
            "chosen_chunks": chosen_chunks,
            "window_fill": window_fill,
            "total_gradient_samples": total_gradient_samples,
            "q_history": q_history,
            "v_history": v_history,
            "first_gain_ready_cycle": first_gain_ready_cycle,
            "publish_cycles": list(probe.publish_cycles),
            "chunk_estimate_ms": dict(probe.chunk_estimate_ms),
            "synth_estimate_ms": probe.synth_estimate_ms,
        },
    )


def _summarize_candidate(
    args: argparse.Namespace,
    candidate: Candidate,
    series: dict[str, Any],
) -> dict[str, Any]:
    errors = np.asarray(series["errors"], dtype=np.float64)
    feedback = np.asarray(series["feedback_peaks"], dtype=np.float64)
    q = np.asarray(series["q_history"], dtype=np.float64)
    v = np.asarray(series["v_history"], dtype=np.float64)
    gain_norms_used = np.asarray(series["gain_norms_used"], dtype=np.float64)
    published_gain_norms = np.asarray(series["published_gain_norms"], dtype=np.float64)
    gain_delta_norms = np.asarray(series["gain_delta_norms"], dtype=np.float64)
    chosen_chunks = np.asarray(series["chosen_chunks"], dtype=np.float64)
    publish_cycles = np.asarray(series["publish_cycles"], dtype=np.float64)
    tail = slice(len(errors) // 2, None)
    finite_age = np.asarray(series["gain_age_cycles"], dtype=np.float64)
    finite_age = finite_age[np.isfinite(finite_age)]
    target_ms = float(args.target_ms)

    update_period = np.diff(publish_cycles) if publish_cycles.size >= 2 else np.asarray([])
    summary = {
        "error_initial_m": float(errors[0]),
        "error_final_m": float(errors[-1]),
        "error_min_m": float(np.min(errors)),
        "tail_error_mean_m": float(np.mean(errors[tail])),
        "tail_error_std_m": float(np.std(errors[tail])),
        "foreground_ms": _stats(series["foreground_ms"]),
        "foreground_command_ms": _stats(series["foreground_command_ms"]),
        "reseed_ms": _stats(series["reseed_ms"]),
        "subset_select_ms": _stats(series["subset_select_ms"]),
        "snapshot_pack_ms": _stats(series["snapshot_pack_ms"]),
        "gain_work_ms": _stats(series["gain_work_ms"]),
        "gain_grad_ms": _stats(series["gain_grad_ms"]),
        "gain_synth_ms": _stats([value for value in series["gain_synth_ms"] if value > 0.0]),
        "cycle_compute_ms": _stats(series["cycle_compute_ms"]),
        "foreground_budget_miss_count": int(
            np.sum(np.asarray(series["foreground_ms"]) > target_ms)
        ),
        "cycle_budget_miss_count": int(np.sum(np.asarray(series["cycle_compute_ms"]) > target_ms)),
        "cycle_budget_miss_rate": float(
            np.mean(np.asarray(series["cycle_compute_ms"]) > target_ms)
        ),
        "first_gain_ready_cycle": series["first_gain_ready_cycle"],
        "first_gain_ready_sec": (
            None
            if series["first_gain_ready_cycle"] is None
            else float(series["first_gain_ready_cycle"] * args.dt)
        ),
        "gain_update_count": int(len(series["publish_cycles"])),
        "gain_update_period_cycles_mean": (
            float(np.mean(update_period)) if update_period.size else float("nan")
        ),
        "gain_update_period_cycles_max": (
            float(np.max(update_period)) if update_period.size else float("nan")
        ),
        "gain_age_cycles_mean": float(np.mean(finite_age)) if finite_age.size else float("nan"),
        "gain_norm_used_final": float(gain_norms_used[-1]),
        "gain_norm_used_max": float(np.max(gain_norms_used)),
        "published_gain_norm_final": (
            float(published_gain_norms[-1]) if published_gain_norms.size else 0.0
        ),
        "published_gain_norm_max": (
            float(np.max(published_gain_norms)) if published_gain_norms.size else 0.0
        ),
        "published_gain_norm_tail_std": (
            float(np.std(published_gain_norms[len(published_gain_norms) // 2 :]))
            if published_gain_norms.size
            else 0.0
        ),
        "gain_delta_norm_mean": (
            float(np.mean(gain_delta_norms)) if gain_delta_norms.size else 0.0
        ),
        "gain_delta_norm_max": (
            float(np.max(gain_delta_norms)) if gain_delta_norms.size else 0.0
        ),
        "chosen_chunk_mean": float(np.mean(chosen_chunks)),
        "chosen_chunk_min_nonzero": (
            int(np.min(chosen_chunks[chosen_chunks > 0])) if np.any(chosen_chunks > 0) else 0
        ),
        "chosen_chunk_max": int(np.max(chosen_chunks)) if chosen_chunks.size else 0,
        "skipped_gain_cycles": int(np.sum(chosen_chunks == 0)),
        "window_fill_final": int(series["window_fill"][-1]),
        "total_gradient_samples": int(series["total_gradient_samples"][-1]),
        "feedback_peak_max_nm": float(np.max(feedback)),
        "feedback_peak_tail_mean_nm": float(np.mean(feedback[tail])),
        "joint_velocity_abs_max": float(np.max(np.abs(v), initial=0.0)),
        "joint_velocity_rms_mean": float(np.mean(np.sqrt(np.mean(v * v, axis=1)))),
        "joint_vel_hf_energy": _joint_vel_hf_energy(v[tail], args.dt),
        "tail_joint_spans": np.ptp(q[tail], axis=0).tolist(),
    }
    summary["gain_timing_by_chunk"] = _gain_timing_by_chunk(series)
    summary["gate_failures"] = _gate_failures(summary)
    return {
        "label": candidate.label,
        "candidate": {
            "chunks": list(candidate.chunks),
            "adaptive": bool(candidate.adaptive),
        },
        "summary": summary,
        "series": _jsonify_series(series),
    }


def _gate_failures(summary: dict[str, Any]) -> list[str]:
    failures = []
    if summary["error_final_m"] > GATE_ERROR_FINAL_M:
        failures.append(f"error_final={summary['error_final_m'] * 1000.0:.2f}mm")
    if summary["feedback_peak_max_nm"] >= GATE_FEEDBACK_PEAK_NM:
        failures.append(f"feedback_peak={summary['feedback_peak_max_nm']:.1f}Nm")
    if summary["joint_velocity_abs_max"] >= GATE_JOINT_VEL_MAX:
        failures.append(f"joint_velocity_abs_max={summary['joint_velocity_abs_max']:.2f}")
    if summary["joint_vel_hf_energy"] > GATE_HF_ENERGY:
        failures.append(f"joint_vel_hf_energy={summary['joint_vel_hf_energy']:.3f}")
    if summary["gain_update_count"] <= 0:
        failures.append("no_gain_update")
    if not np.isfinite(summary["published_gain_norm_final"]):
        failures.append("nonfinite_gain")
    return failures


def _jsonify_series(series: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in series.items():
        if key in {"q_history", "v_history"}:
            result[key] = np.asarray(value, dtype=np.float64).tolist()
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, list):
            result[key] = [
                item.tolist() if isinstance(item, np.ndarray) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _gain_timing_by_chunk(series: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    chunks = np.asarray(series["chosen_chunks"], dtype=np.int64)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for chunk in sorted(int(value) for value in np.unique(chunks) if value > 0):
        mask = chunks == chunk
        result[str(chunk)] = {
            "subset_select_ms": _stats(np.asarray(series["subset_select_ms"])[mask]),
            "snapshot_pack_ms": _stats(np.asarray(series["snapshot_pack_ms"])[mask]),
            "gain_grad_ms": _stats(np.asarray(series["gain_grad_ms"])[mask]),
            "gain_work_ms": _stats(np.asarray(series["gain_work_ms"])[mask]),
        }
    return result


def _print_candidate_summary(result: dict[str, Any]) -> None:
    s = result["summary"]
    fails = s["gate_failures"]
    mark = "PASS" if not fails else "CHECK"
    first_gain = (
        "none"
        if s["first_gain_ready_sec"] is None
        else f"{s['first_gain_ready_sec']:.3f}s"
    )
    print(
        f"{mark:5s} {result['label']:18s} "
        f"err_f={s['error_final_m'] * 1000.0:6.2f}mm "
        f"fg_p99={s['foreground_ms']['p99']:6.2f}ms "
        f"seed_p99={s['reseed_ms']['p99']:5.2f}ms "
        f"gain_p99={s['gain_work_ms']['p99']:6.2f}ms "
        f"cycle_p99={s['cycle_compute_ms']['p99']:6.2f}ms "
        f"miss={s['cycle_budget_miss_count']:3d}/{len(result['series']['errors'])} "
        f"firstK={first_gain:>7s} "
        f"updates={s['gain_update_count']:3d} "
        f"|K|={s['published_gain_norm_final']:6.2f} "
        f"dKmax={s['gain_delta_norm_max']:6.2f}",
        flush=True,
    )
    if fails:
        print(f"      gate notes: {', '.join(fails)}", flush=True)


def _make_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = [
        Candidate(label=f"fixed-{chunk}", chunks=(chunk,), adaptive=False)
        for chunk in args.chunk_sizes
    ]
    if args.adaptive_chunks:
        chunks = tuple(sorted(set(args.adaptive_chunks)))
        candidates.append(
            Candidate(
                label="adaptive-" + "-".join(str(chunk) for chunk in sorted(chunks, reverse=True)),
                chunks=chunks,
                adaptive=True,
            )
        )
    return candidates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--control-points", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--timing-mode", choices=("immediate", "gazebo"), default="gazebo")
    parser.add_argument(
        "--retime-initial-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--desired-state-mode",
        choices=("base", "midpoint", "next"),
        default="base",
    )
    parser.add_argument("--feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-torque", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-velocity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reseed-policy",
        choices=("every", "periodic", "initial", "none"),
        default="every",
        help=(
            "Nominal trajectory reseeding policy for the foreground critical path. "
            "'every' matches the current ROS default; 'initial' warm-starts from "
            "the solver's shifted previous solution after the first measured step."
        ),
    )
    parser.add_argument(
        "--reseed-period",
        type=int,
        default=2,
        help="Cycle period used only when --reseed-policy=periodic.",
    )
    parser.add_argument("--gain-buffer-size", type=int, default=512)
    parser.add_argument("--chunk-sizes", type=parse_int_list, default=parse_int_list("32,64,128"))
    parser.add_argument("--adaptive-chunks", type=parse_int_list, default=parse_int_list(""))
    parser.add_argument("--adaptive-allow-overrun", action="store_true")
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--safety-margin-ms", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--mjx-iterations", type=int, default=None)
    parser.add_argument("--mjx-ls-iterations", type=int, default=None)
    parser.add_argument("--mjx-tolerance", type=float, default=None)
    parser.add_argument("--jax-cache-dir", type=str, default=DEFAULT_JAX_CACHE_DIR)
    args = parser.parse_args(argv)

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive")
    if args.reseed_period <= 0:
        raise ValueError("--reseed-period must be positive")
    if args.gain_buffer_size <= 0:
        raise ValueError("--gain-buffer-size must be positive")
    if not args.chunk_sizes and not args.adaptive_chunks:
        raise ValueError("provide at least one fixed or adaptive chunk size")
    all_chunks = args.chunk_sizes + args.adaptive_chunks
    if any(chunk > args.samples for chunk in all_chunks):
        raise ValueError("chunk size cannot exceed --samples")
    if any(chunk > args.gain_buffer_size for chunk in all_chunks):
        raise ValueError("chunk size cannot exceed --gain-buffer-size")

    mjx_opts = {}
    if args.mjx_iterations is not None:
        mjx_opts["iterations"] = args.mjx_iterations
    if args.mjx_ls_iterations is not None:
        mjx_opts["ls_iterations"] = args.mjx_ls_iterations
    if args.mjx_tolerance is not None:
        mjx_opts["tolerance"] = args.mjx_tolerance
    args.mjx_opts = mjx_opts or None
    args.jax_cache_dir = configure_jax_cache(args.jax_cache_dir)

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}", flush=True)
    if args.jax_cache_dir is not None:
        print(f"JAX compilation cache: {args.jax_cache_dir}", flush=True)
    print(
        f"steps={args.steps} timing={args.timing_mode} dt={args.dt} "
        f"samples={args.samples} h={args.horizon} cp={args.control_points} "
        f"buffer={args.gain_buffer_size} chunks={args.chunk_sizes} "
        f"adaptive={args.adaptive_chunks} target={args.target_ms}ms "
        f"safety={args.safety_margin_ms}ms reseed={args.reseed_policy} "
        f"reseed_period={args.reseed_period} "
        f"mjx={args.mjx_opts}",
        flush=True,
    )

    results = []
    for candidate in _make_candidates(args):
        print(f"\n=== {candidate.label} ===", flush=True)
        result = _run_candidate(args, candidate)
        _print_candidate_summary(result)
        results.append(result)

    payload = {
        "config": {
            "steps": args.steps,
            "samples": args.samples,
            "horizon": args.horizon,
            "control_points": args.control_points,
            "dt": args.dt,
            "substeps": args.substeps,
            "timing_mode": args.timing_mode,
            "reseed_policy": args.reseed_policy,
            "reseed_period": args.reseed_period,
            "gain_buffer_size": args.gain_buffer_size,
            "chunk_sizes": args.chunk_sizes,
            "adaptive_chunks": args.adaptive_chunks,
            "adaptive_allow_overrun": args.adaptive_allow_overrun,
            "target_ms": args.target_ms,
            "safety_margin_ms": args.safety_margin_ms,
            "warmups": args.warmups,
            "backend": jax.default_backend(),
            "jax_cache_dir": args.jax_cache_dir,
            "mjx_opts": args.mjx_opts,
        },
        "gates": {
            "error_final_m": GATE_ERROR_FINAL_M,
            "feedback_peak_nm": GATE_FEEDBACK_PEAK_NM,
            "joint_velocity_abs_max": GATE_JOINT_VEL_MAX,
            "joint_vel_hf_energy": GATE_HF_ENERGY,
        },
        "results": results,
    }
    if args.output_json is not None:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote benchmark detail -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
