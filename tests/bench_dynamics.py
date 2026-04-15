"""
Benchmark: Cross-library GPU-parallel forward dynamics for Feedback-MPPI.

Tests the full dynamics pipeline relevant to sbmpc's Feedback-MPPI controller
(Belvedere et al., 2026, RA-L) on a 7-DoF Franka Panda arm with torque control.

The Feedback-MPPI gains require:
    jax.vmap(jax.value_and_grad(rollout_single, argnums=0))
which backpropagates through the integrator and dynamics. Therefore ALL backends
must be JAX-differentiable. This eliminates CusADi (CUDA kernels), MJX-Warp (no
autodiff), and GRiD (C++/CUDA).

Backends tested:
  A) JaxSim ABA — build() inside JIT            [YOUR CURRENT CODE]
  B) JaxSim ABA — template .replace()            [QUICK FIX]
  C) JaxSim CRBA + Cholesky solve                [ALTERNATIVE ALGO]
  D) MJX-JAX forward (not mjx.step)              [DIFFERENT ENGINE]
  E) Pinocchio + CasADi + Jaxadi                 [SYMBOLIC -> JAX]

Each backend is measured for:
  - Forward pass:  vmap(dynamics)(q, qd, tau)       — raw MPPI rollout
  - Gradient pass: vmap(grad(si_euler_step))         — what F-MPPI gains need
  - JIT compile time

Plus:
  - si_euler 2x-call vs 1x-call overhead test

Usage:
    pixi run -e cuda python benchmarks/bench_dynamics.py
    pixi run -e cuda python benchmarks/bench_dynamics.py --batch-sizes 14 32 512 2048
    pixi run -e cuda python benchmarks/bench_dynamics.py --skip-grad
    pixi run -e cuda python benchmarks/bench_dynamics.py --skip-jaxadi

Dependencies beyond sbmpc:
    pip install jaxsim robot-descriptions pinocchio mujoco-mjx
    pip install jaxadi casadi   # for Method E
"""
from __future__ import annotations

import argparse
import dataclasses
from pyexpat import model
import time
import traceback
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZES = [32, 512, 2048]
N_WARMUP = 10
N_TRIALS = 300
NQ = 7
DT = 0.02

PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


class BenchResult(NamedTuple):
    name: str
    jit_time_ms: float
    single_us: float
    batch_ms: dict[int, float]


# ── Timing ─────────────────────────────────────────────────────────────────

def time_jit(fn, *args):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) * 1000.0


def time_exec(fn, *args, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n_trials):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / n_trials * 1e6  # us


def make_inputs(key, K=None):
    shape = (NQ,) if K is None else (K, NQ)
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.uniform(k1, shape, minval=-2.0, maxval=2.0),
        jax.random.uniform(k2, shape, minval=-1.0, maxval=1.0),
        jax.random.uniform(k3, shape, minval=-10.0, maxval=10.0),
    )


def make_state_and_tau(key, K=None):
    shape = (2 * NQ,) if K is None else (K, 2 * NQ)
    tau_shape = (NQ,) if K is None else (K, NQ)
    k1, k2 = jax.random.split(key)
    return (
        jax.random.uniform(k1, shape, minval=-1.0, maxval=1.0),
        jax.random.uniform(k2, tau_shape, minval=-10.0, maxval=10.0),
    )


# ══════════════════════════════════════════════════════════════════════════
# JaxSim shared helpers
# ══════════════════════════════════════════════════════════════════════════

def _build_jaxsim_model():
    import jaxsim.api as js
    import jaxsim.parsers.rod as rodp
    from robot_descriptions.panda_description import URDF_PATH

    desc = rodp.build_model_description(
        URDF_PATH, is_urdf=True,
    ).reduce(considered_joints=PANDA_ARM_JOINT_NAMES)
    desc = dataclasses.replace(desc, fixed_base=True)
    return js.model.JaxSimModel.build(model_description=desc), URDF_PATH


def _js_reorder(js_model):
    names = tuple(js_model.joint_names())
    e2j = jnp.array([names.index(n) for n in PANDA_ARM_JOINT_NAMES], dtype=jnp.int32)
    j2e = jnp.array(np.argsort(np.array(e2j)), dtype=jnp.int32)
    return e2j, j2e


# ══════════════════════════════════════════════════════════════════════════
# A: JaxSim — build() inside JIT  (your current code pattern)
# ══════════════════════════════════════════════════════════════════════════

def build_A(js_model):
    import jaxsim.api as js
    e2j, j2e = _js_reorder(js_model)

    def aba(q, qd, tau):
        data = js.data.JaxSimModelData.build(
            model=js_model,
            joint_positions=jnp.take(q, j2e, axis=-1),
            joint_velocities=jnp.take(qd, j2e, axis=-1),
        )
        _, ddq = js.model.forward_dynamics_aba(
            model=js_model, data=data,
            joint_forces=jnp.take(tau, j2e, axis=-1),
        )
        return jnp.take(ddq, e2j, axis=-1)
    return jax.jit(aba)


# ══════════════════════════════════════════════════════════════════════════
# B: JaxSim — template .replace()
# ══════════════════════════════════════════════════════════════════════════

def build_B(js_model):
    import jaxsim.api as js
    e2j, j2e = _js_reorder(js_model)
    tpl = js.data.JaxSimModelData.build(
        model=js_model,
        joint_positions=jnp.zeros(NQ),
        joint_velocities=jnp.zeros(NQ),
    )

    def aba(q, qd, tau):
        data = tpl.replace(
            model=js_model,
            joint_positions=jnp.take(q, j2e, axis=-1),
            joint_velocities=jnp.take(qd, j2e, axis=-1),
        )
        _, ddq = js.model.forward_dynamics_aba(
            model=js_model, data=data,
            joint_forces=jnp.take(tau, j2e, axis=-1),
        )
        return jnp.take(ddq, e2j, axis=-1)
    return jax.jit(aba)


# ══════════════════════════════════════════════════════════════════════════
# C: JaxSim CRBA + Cholesky
# ══════════════════════════════════════════════════════════════════════════

def build_C(js_model):
    import jaxsim.api as js
    e2j, j2e = _js_reorder(js_model)
    tpl = js.data.JaxSimModelData.build(
        model=js_model,
        joint_positions=jnp.zeros(NQ),
        joint_velocities=jnp.zeros(NQ),
    )

    def minv(q, qd, tau):
        data = tpl.replace(
            model=js_model,
            joint_positions=jnp.take(q, j2e, axis=-1),
            joint_velocities=jnp.take(qd, j2e, axis=-1),
        )
        M_full = js.model.free_floating_mass_matrix(model=js_model, data=data)
        M = M_full[-NQ:, -NQ:]
        _, bias = js.model.inverse_dynamics(
            model=js_model, data=data, joint_accelerations=jnp.zeros(NQ),
        )
        rhs = jnp.take(tau, j2e, axis=-1) - bias
        L = jax.lax.linalg.cholesky(M)
        y = jax.lax.linalg.triangular_solve(L, rhs, left_side=True, lower=True)
        ddq = jax.lax.linalg.triangular_solve(L.T, y, left_side=True, lower=False)
        return jnp.take(ddq, e2j, axis=-1)
    return jax.jit(minv)


# ══════════════════════════════════════════════════════════════════════════
# D: MJX-JAX — mjx.forward only (no step, no contacts)
# ══════════════════════════════════════════════════════════════════════════

def build_D(scene_path: str):
    import mujoco
    from mujoco import mjx

    mj_model = mujoco.MjModel.from_xml_path(scene_path)
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    mjx_model = mjx.put_model(mj_model)
    mjx_data_tpl = mjx.make_data(mj_model)

    def fwd(q, qd, tau):
        data = mjx_data_tpl.replace(qpos=q, qvel=qd, qfrc_applied=tau)
        data = mjx.forward(mjx_model, data)
        return data.qacc

    return jax.jit(fwd)


# ══════════════════════════════════════════════════════════════════════════
# E: Pinocchio + CasADi + Jaxadi
# ══════════════════════════════════════════════════════════════════════════

def build_E(urdf_path: str):
    import casadi as cs
    import pinocchio as pin
    import pinocchio.casadi as cpin
    from jaxadi import convert

    full_model = pin.buildModelFromUrdf(urdf_path)
    finger_ids = [
        full_model.getJointId(n)
        for n in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    model = pin.buildReducedModel(full_model, finger_ids, np.zeros(full_model.nq))

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    q_sym = cs.SX.sym("q", model.nq)
    v_sym = cs.SX.sym("v", model.nv)
    tau_sym = cs.SX.sym("tau", model.nv)

    ddq_sym = cpin.aba(cmodel, cdata, q_sym, v_sym, tau_sym)
    aba_cs = cs.Function("aba", [q_sym, v_sym, tau_sym], [ddq_sym])
    aba_jax = convert(aba_cs, compile=True)

    def aba(q, qd, tau):
        return aba_jax(q, qd, tau)

    return jax.jit(aba)


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATOR WRAPPERS
# ══════════════════════════════════════════════════════════════════════════

def make_si_euler_fixed(aba_fn):
    """Correct si_euler: ONE dynamics call."""
    def step(state, tau):
        q, v = state[:NQ], state[NQ:]
        qdd = aba_fn(q, v, tau)
        v_kp1 = v + DT * qdd
        q_kp1 = q + DT * v_kp1
        return jnp.concatenate([q_kp1, v_kp1])
    return step


def make_si_euler_current(aba_fn):
    """Current sbmpc pattern: TWO dynamics calls (redundant)."""
    def step(state, tau):
        q, v = state[:NQ], state[NQ:]
        qdd = aba_fn(q, v, tau)
        v_kp1 = v + DT * qdd
        _ = aba_fn(q, v_kp1, tau)  # wasteful second call
        q_kp1 = q + DT * v_kp1
        return jnp.concatenate([q_kp1, v_kp1])
    return step


def make_grad_step(integrator_fn):
    """grad(cost_of_one_step) w.r.t. state — Feedback-MPPI atomic unit."""
    def loss(state, tau):
        return jnp.sum(jnp.square(integrator_fn(state, tau)))
    return jax.jit(jax.grad(loss, argnums=0))


# ══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════

def bench(name, fn, make_args, key, batch_sizes, n_trials=N_TRIALS):
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")

    args = make_args(key)
    jit_ms = time_jit(fn, *args)
    print(f"  JIT compile:     {jit_ms:9.1f} ms")
    single_us = time_exec(fn, *args, n_trials=n_trials)
    print(f"  Single call:     {single_us:9.1f} us")

    fn_batch = jax.jit(jax.vmap(fn))
    batch_ms = {}
    for K in batch_sizes:
        args_b = make_args(key, K)
        jax.block_until_ready(fn_batch(*args_b))
        us = time_exec(fn_batch, *args_b, n_trials=n_trials)
        ms = us / 1000.0
        batch_ms[K] = ms
        print(f"  K={K:<5d}:       {ms:9.3f} ms   ({us/K:.2f} us/eval)")

    return BenchResult(name, jit_ms, single_us, batch_ms)


def print_summary(title, results, batch_sizes):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    header = f"  {'Method':<42} {'JIT':>7}"
    for K in batch_sizes:
        header += f" K={K:<6}"
    print(header)
    print(f"  {'─'*42} {'─'*7}" + " ───────" * len(batch_sizes))
    for r in results:
        row = f"  {r.name:<42} {r.jit_time_ms:6.0f}s"
        for K in batch_sizes:
            row += f" {r.batch_ms.get(K, float('nan')):6.3f}"
        print(row)
    print()
    for K in batch_sizes:
        valid = [r for r in results if K in r.batch_ms]
        if len(valid) < 2:
            continue
        best = min(valid, key=lambda r: r.batch_ms[K])
        worst = max(valid, key=lambda r: r.batch_ms[K])
        ratio = worst.batch_ms[K] / max(best.batch_ms[K], 1e-9)
        print(f"  K={K}: WINNER = {best.name}  ({ratio:.1f}x over slowest)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-library dynamics benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--skip-grad", action="store_true")
    parser.add_argument("--skip-mjx", action="store_true")
    parser.add_argument("--skip-jaxadi", action="store_true")
    args = parser.parse_args()

    n_trials = args.trials
    bs = args.batch_sizes

    scene_path = str(Path(__file__).resolve().parents[1]
                     / "examples" / "franka_emika_panda" / "panda_nohand.xml")

    print("=" * 75)
    print("  CROSS-LIBRARY DYNAMICS BENCHMARK FOR FEEDBACK-MPPI")
    print("  Franka Panda 7-DoF | torque control | no contacts")
    print("=" * 75)
    backend = jax.default_backend()
    print(f"  JAX backend:    {backend}")
    if backend == "gpu":
        print(f"  GPU:            {jax.devices('gpu')[0]}")
    print(f"  Precision:      {'f64' if jax.config.jax_enable_x64 else 'f32'}")
    print(f"  Batch sizes:    {bs}")
    print(f"  Gradient test:  {'OFF' if args.skip_grad else 'ON'}")

    key = jax.random.PRNGKey(42)
    fwd_results, grad_results = [], []
    aba_fns = {}

    # ── Build JaxSim model ──
    print(f"\n  Building JaxSim model...")
    js_model, urdf_path = _build_jaxsim_model()
    print(f"  Joints: {list(js_model.joint_names())}")

    # ═══════════════════════════════════════════════════════
    #  PART 1: FORWARD DYNAMICS
    # ═══════════════════════════════════════════════════════

    for label, builder, build_args in [
        ("A", build_A, (js_model,)),
        ("B", build_B, (js_model,)),
        ("C", build_C, (js_model,)),
    ]:
        try:
            fn = builder(*build_args)
            aba_fns[label] = fn
            names = {
                "A": "A: JaxSim ABA build()-in-jit [CURRENT]",
                "B": "B: JaxSim ABA .replace()      [FIX]",
                "C": "C: JaxSim CRBA+Cholesky",
            }
            fwd_results.append(bench(names[label], fn, make_inputs, key, bs, n_trials))
        except Exception:
            print(f"  {label} FAILED:\n{traceback.format_exc()}")

    if not args.skip_mjx:
        try:
            print(f"\n  Building MJX-JAX model (contacts disabled)...")
            fn_d = build_D(scene_path)
            aba_fns["D"] = fn_d
            fwd_results.append(bench("D: MJX-JAX forward (no contacts)", fn_d, make_inputs, key, bs, n_trials))
        except Exception:
            print(f"  D FAILED:\n{traceback.format_exc()}")

    if not args.skip_jaxadi:
        try:
            print(f"\n  Building Pinocchio + CasADi + Jaxadi ABA...")
            fn_e = build_E(urdf_path)
            aba_fns["E"] = fn_e
            fwd_results.append(bench("E: Pinocchio+CasADi+Jaxadi", fn_e, make_inputs, key, bs, n_trials))
        except Exception:
            print(f"  E FAILED:\n{traceback.format_exc()}")

    if fwd_results:
        print_summary("FORWARD DYNAMICS — vmap(aba)(q, qd, tau)", fwd_results, bs)

    # ═══════════════════════════════════════════════════════
    #  PART 2: GRADIENT (Feedback-MPPI gains)
    # ═══════════════════════════════════════════════════════

    if not args.skip_grad:
        print(f"\n\n{'='*75}")
        print("  GRADIENT THROUGHPUT — vmap(grad(si_euler_step)) w.r.t. state")
        print("  This is the backward pass that Feedback-MPPI needs for gains.")
        print(f"{'='*75}")

        name_map = {
            "A": "A: JaxSim build() GRAD",
            "B": "B: JaxSim .replace() GRAD",
            "C": "C: CRBA+Chol GRAD",
            "D": "D: MJX-JAX GRAD",
            "E": "E: Pin+Jaxadi GRAD",
        }
        for label, aba_fn in aba_fns.items():
            try:
                step_fn = make_si_euler_fixed(aba_fn)
                grad_fn = make_grad_step(step_fn)
                grad_results.append(bench(name_map.get(label, f"{label} GRAD"),
                                          grad_fn, make_state_and_tau, key, bs, n_trials))
            except Exception:
                print(f"  {label} GRAD FAILED:\n{traceback.format_exc()}")

        if grad_results:
            print_summary("GRADIENT — backward through dynamics", grad_results, bs)

    # ═══════════════════════════════════════════════════════
    #  PART 3: SI_EULER DOUBLE-CALL OVERHEAD
    # ═══════════════════════════════════════════════════════

    if "B" in aba_fns:
        print(f"\n\n{'='*75}")
        print("  SI_EULER: 2x dynamics calls (current) vs 1x (fixed)")
        print("  model.py:94 calls dynamics() twice. Second call is redundant.")
        print(f"{'='*75}")

        fn = aba_fns["B"]
        int_res = []
        int_res.append(bench("si_euler CURRENT (2x)", jax.jit(make_si_euler_current(fn)),
                             make_state_and_tau, key, bs, n_trials))
        int_res.append(bench("si_euler FIXED   (1x)", jax.jit(make_si_euler_fixed(fn)),
                             make_state_and_tau, key, bs, n_trials))
        print_summary("INTEGRATOR OVERHEAD", int_res, bs)
        for K in bs:
            t2, t1 = int_res[0].batch_ms.get(K, 0), int_res[1].batch_ms.get(K, 0)
            if t2 > 0 and t1 > 0:
                print(f"  K={K}: fixing si_euler saves {t2-t1:.3f}ms ({(t2-t1)/t2*100:.0f}%)")

    # ═══════════════════════════════════════════════════════
    #  INTERPRETATION GUIDE
    # ═══════════════════════════════════════════════════════

    print(f"""

{'='*75}
  HOW TO INTERPRET
{'='*75}

  FORWARD TABLE:
    A vs B  -> Cost of JaxSimModelData.build() inside JIT.
               B faster? Apply .replace() fix immediately.
    B vs D  -> JaxSim vs MJX-JAX as a dynamics engine.
               If D wins, MJX-JAX is the better backend.
    B vs E  -> JaxSim vs Pinocchio symbolic pipeline.
               E slow? Confirms CasADi expression graphs
               compile poorly in XLA (known Jaxadi limit).
    B vs C  -> ABA vs M^-1 Cholesky for n=7 on GPU.

  GRADIENT TABLE:
    This is THE number for Feedback-MPPI.
    Some backends have much worse fwd/bwd ratios.
    MJX-JAX grad may fail if contacts are on (CG solver
    uses while_loop, no reverse-mode AD).

  SI_EULER:
    2x ~ 2x slower than 1x? Fix halves dynamics cost.
    Change model.py:94 to call dynamics once.

  TOTAL COST per Feedback-MPPI iteration:
    total_ms ~ (fwd_K_time + grad_K_time) x horizon
    Example: K=32, H=8, fwd=0.2ms, grad=0.4ms
             -> (0.2+0.4)*8 = 4.8ms -> ~200Hz
""")


if __name__ == "__main__":
    main()
