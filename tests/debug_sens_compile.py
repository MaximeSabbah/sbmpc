"""Isolate the XLA `hlo_lexer` / giant-int-literal issue.

Calls `rollout_sens_to_state` directly with a range of K values (leading vmap
axis sizes). Each call is timed and its exception/stderr captured. The goal:

  1. Know whether the error depends on K (e.g. fires only when K != N_mppi)
     or appears for every fresh compile.
  2. See how long each K legitimately takes to compile, without the
     surrounding controller logic muddying the picture.

Outputs one line per K with either OK + compile+exec time, or FAIL with
the exception type. Watch stderr for `hlo_lexer.cc:443 Failed to parse
int literal` messages alongside — those are XLA-side and won't raise a
Python exception.

Run:
  PYTHONPATH=/home/msabbah/Desktop/sbmpc \\
    /home/msabbah/Desktop/sbmpc/.pixi/envs/cuda/bin/python \\
    tests/debug_sens_compile.py

The script aborts only on a Python exception. If it appears to HANG on a
particular K, that K is the one driving the pathological compile — note
the value and Ctrl+C.
"""
from __future__ import annotations

import time
import traceback

import jax
import jax.numpy as jnp

from sbmpc.controller.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all


def main() -> None:
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"JAX {jax.__version__}", flush=True)

    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.gain_method = "exact"
    config.MPC.dt = 0.02
    config.MPC.horizon = 8
    config.MPC.num_parallel_computations = 1024
    config.MPC.num_control_points = 8
    config.MPC.initial_guess = planner.nominal_torque_sequence(
        config.MPC.horizon,
        config.MPC.dt,
    )

    print("[setup] build_all ...", flush=True)
    t0 = time.perf_counter()
    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    print(f"[setup] build_all done in {time.perf_counter() - t0:.1f}s", flush=True)

    rollout_gen = sim.controller.rollout_gen
    sampler = sim.controller.sampler
    state = sim.current_state_vec()
    reference = sim.const_reference
    if reference.ndim == 1:
        reference_tiled = jnp.tile(reference, (rollout_gen.horizon + 1, 1))
    else:
        reference_tiled = reference

    # Full-size raw samples to subslice from; same tensor shape the real
    # controller sees every cycle.
    raw_full = sampler.sample_input_sequence(sampler.master_key)
    if config.MPC.smoothing == "Spline":
        ctrl_full = sampler.optimal_samples[rollout_gen.control_spline_indices, :] + raw_full
    else:
        ctrl_full = sampler.optimal_samples + raw_full
    jax.block_until_ready(ctrl_full)
    print(
        f"[setup] ctrl_full.shape={ctrl_full.shape}  dtype={ctrl_full.dtype}  "
        f"state.shape={state.shape}  reference_tiled.shape={reference_tiled.shape}",
        flush=True,
    )

    # Ascending K so small compiles come first. If any K hangs, user Ctrl+C'ing
    # tells us exactly which one. 1024 last because it's the known-good case.
    K_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for K in K_values:
        print(f"\n[K={K}] calling rollout_sens_to_state ...", flush=True)
        ctrl_sub = ctrl_full[:K]
        t0 = time.perf_counter()
        try:
            (costs_sub, ctrl_ret), grads_sub = rollout_gen.rollout_sens_to_state(
                state, reference_tiled, ctrl_sub
            )
            jax.block_until_ready(grads_sub)
            dt = time.perf_counter() - t0
            print(
                f"[K={K}] OK  {dt:6.2f}s  costs={costs_sub.shape}  "
                f"ctrl_ret={ctrl_ret.shape}  grads={grads_sub.shape}",
                flush=True,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"[K={K}] FAIL elapsed={dt:.1f}s  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
