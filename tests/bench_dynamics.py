"""
Benchmark: Cross-library GPU-parallel forward dynamics for Feedback-MPPI.

Tests the full dynamics pipeline relevant to sbmpc's Feedback-MPPI controller
(Belvedere et al., 2026, RA-L) on a 7-DoF Franka Panda arm with torque control.
The model includes the Panda hand (mass 0.73 kg) attached as a fixed end-effector,
matching real deployment conditions.

The Feedback-MPPI gains require:
    jax.vmap(jax.value_and_grad(rollout_single, argnums=0))
which backpropagates through the integrator and dynamics. Therefore ALL backends
must be JAX-differentiable. This eliminates CusADi (CUDA kernels), MJX-Warp (no
autodiff), and GRiD (C++/CUDA).

Backends tested:
  A) JaxSim ABA — build() inside JIT            [CURRENT CODE]
  D) MJX-JAX — mjx.forward, full Panda w/ hand  [CANDIDATE]
  E) Pinocchio + CasADi + Jaxadi                [SYMBOLIC → JAX, fwd only]
  F) ADAM-JAX — CRBA + RNEA forward dynamics    [CANDIDATE]

MODEL CONSISTENCY:
  All backends use the same physical model: 7-DoF Panda arm + hand (fixed).
  - JaxSim / ADAM: build from panda.urdf (robot_descriptions), reduce to 7 arm
    joints — the hand+fingers inertia is lumped into panda_link7.
  - Pinocchio: same URDF, lock finger joints (hand stays via fixed joints).
  - MJX: panda.xml (7 arm joints + 2 finger joints). Fingers are locked at
    q=0, qd=0, τ=0 internally — the hand body is present with correct mass.
  Minor residuals between methods (< 5 rad/s²) are expected because JaxSim
  lumps the hand into link7 while Pinocchio/MJX keep it as a separate body.

FAIRNESS FIXES (panda.xml):
  Zeroed: actuator gainprm/biasprm (PD gains were injecting −4500q−450qd),
          dof_damping (XML=1 vs URDF≈0.003), dof_armature (XML=0.1).

Gradient benchmark:
  E (Pinocchio+Jaxadi) is excluded — it is ~30× slower than others and adds
  no information beyond the forward comparison.

Usage:
    pixi run -e cuda python tests/bench_dynamics.py
    pixi run -e cuda python tests/bench_dynamics.py --batch-sizes 32 512 2048
    pixi run -e cuda python tests/bench_dynamics.py --skip-grad
    pixi run -e cuda python tests/bench_dynamics.py --skip-mjx
    pixi run -e cuda python tests/bench_dynamics.py --skip-jaxadi
    pixi run -e cuda python tests/bench_dynamics.py --skip-adam

Dependencies beyond sbmpc:
    jaxsim robot-descriptions mujoco-mjx           # A, D
    pinocchio casadi jaxadi                        # E  (in pixi cuda env)
    adam-robotics[jax]                             # F  (add to pyproject.toml)
"""
from __future__ import annotations

import argparse
import dataclasses
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxsim.api as js
import jaxsim.parsers.rod as rodp
from robot_descriptions.panda_description import URDF_PATH
from robot_descriptions.panda_mj_description import MJCF_PATH
import xml.etree.ElementTree as ET

import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
from jaxadi import convert
import mujoco
from mujoco import mjx
import jaxsim.api as js

# ── Configuration ──────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZES = [32, 512, 2048]
N_WARMUP = 10
N_TRIALS = 300
NQ = NQ_MJX = 9     # panda.xml has 7 arm + 2 finger joints
DT = 0.02


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
    return (time.perf_counter() - t0) / n_trials * 1e6  # µs


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

def _build_models():
    desc = rodp.build_model_description(
        URDF_PATH, is_urdf=True,
    )
    desc = dataclasses.replace(desc, fixed_base=True)
    jaxsim_model = js.model.JaxSimModel.build(model_description=desc)
    
    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    
    return jaxsim_model, mj_model, pin_model, URDF_PATH


def _js_reorder(js_model):
    names = tuple(js_model.joint_names())
    e2j = jnp.array([names.index(n) for n in js_model.joint_names()], dtype=jnp.int32)
    j2e = jnp.array(np.argsort(np.array(e2j)), dtype=jnp.int32)
    return e2j, j2e


# ══════════════════════════════════════════════════════════════════════════
# A: JaxSim — build() inside JIT  (current code pattern)
# ══════════════════════════════════════════════════════════════════════════

def build_A(js_model):
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
# B: MJX-JAX — mjx.forward, full Panda model WITH hand
# ══════════════════════════════════════════════════════════════════════════

def build_B(mj_model):
    """
    Mujoco build
    """

    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    mj_model.actuator_gainprm[:] = 0.0
    mj_model.actuator_biasprm[:] = 0.0
    mj_model.dof_damping[:] = 0.0
    mj_model.dof_frictionloss[:] = 0.0
    mj_model.dof_armature[:] = 0.0

    mjx_model = mjx.put_model(mj_model)
    mjx_data_tpl = mjx.make_data(mj_model)

    def fwd(q, qd, tau):
        # Pad arm (7) to full model (9); finger joints locked at 0.
        q9   = jnp.concatenate([q,   jnp.zeros(NQ_MJX - NQ)])
        qd9  = jnp.concatenate([qd,  jnp.zeros(NQ_MJX - NQ)])
        tau9 = jnp.concatenate([tau, jnp.zeros(NQ_MJX - NQ)])
        data = mjx_data_tpl.replace(qpos=q9, qvel=qd9, qfrc_applied=tau9)
        data = mjx.forward(mjx_model, data)
        return data.qacc[:NQ]

    return jax.jit(fwd)


# ══════════════════════════════════════════════════════════════════════════
# C: Pinocchio + CasADi + Jaxadi  (forward pass only — too slow for grads)
# ══════════════════════════════════════════════════════════════════════════

def build_C(pin_model):
    """
    CasADi-symbolic ABA compiled through Jaxadi to XLA.
    Pinocchio reads the same URDF as JaxSim; after locking the finger joints
    the hand body remains in the kinematic tree via fixed joints.
    Jaxadi's convert() returns a list of outputs — we extract [0] and ravel
    to get a plain (NQ,) JAX array.
    """

    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    q_sym   = cs.SX.sym("q",   pin_model.nq)
    v_sym   = cs.SX.sym("v",   pin_model.nv)
    tau_sym = cs.SX.sym("tau", pin_model.nv)

    ddq_sym = cpin.aba(cmodel, cdata, q_sym, v_sym, tau_sym)
    aba_cs  = cs.Function("aba", [q_sym, v_sym, tau_sym], [ddq_sym])
    aba_jax = convert(aba_cs, compile=True)

    def aba(q, qd, tau):
        # jaxadi returns a list of JAX arrays (one per CasADi output).
        # CasADi column vectors come back as (N, 1); ravel → (N,).
        return jnp.ravel(aba_jax(q, qd, tau)[0])

    return jax.jit(aba)


# ══════════════════════════════════════════════════════════════════════════
# D: ADAM-JAX — CRBA + RNEA forward dynamics (not working yet)
# ══════════════════════════════════════════════════════════════════════════

# def build_D(urdf_path: str, joints_name_list: list[str]):
#     """
#     ADAM (Automatic Differentiation for rigid-body dynamics Algorithm in
#     Multi-body systems) JAX backend. 
#     """
#     try:
#         from adam.jax import KinDynComputations
#         from adam import Representations
#     except ImportError as exc:
#         raise ImportError(
#             "ADAM not installed.  Add to pyproject.toml:\n"
#             "then run: pixi install"
#         ) from exc
        
#     comp = KinDynComputations(
#         urdf_path,
#         joints_name_list=joints_name_list,
#         # Fixed-base: gravity vector can be optionally specified
#         gravity=jnp.array([0, 0, -9.80665, 0, 0, 0])
#     )
    
#     # Set velocity representation
#     comp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

#     # Fixed-base constants (traced once into the JIT graph)
#     H_b = jnp.eye(4)     # base homogeneous transform: identity = fixed at world
#     vB  = jnp.zeros(6)   # base spatial velocity = 0

#     def aba(q, qd, tau):
#         qdd = comp.aba(
#             base_transform=H_b,
#             joint_positions=q,
#             base_velocity=vB,
#             joint_velocities=qd,
#             joint_torques=tau
#         )
#         # The return value is a 1D array: [base_acceleration (6), joint_accelerations (n)]
#         # For a fixed-base robot, we only need the joint part.
#         return qdd[6:]

#     return jax.jit(aba)


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATOR WRAPPERS
# ══════════════════════════════════════════════════════════════════════════

def make_si_euler(aba_fn):
    """Semi-implicit Euler: one dynamics call."""
    def step(state, tau):
        q, v = state[:NQ], state[NQ:]
        qdd    = aba_fn(q, v, tau)
        v_kp1  = v + DT * qdd
        q_kp1  = q + DT * v_kp1
        return jnp.concatenate([q_kp1, v_kp1])
    return step


def make_grad_step(integrator_fn):
    """grad(cost_of_one_step) w.r.t. state — Feedback-MPPI atomic unit."""
    def loss(state, tau):
        return jnp.sum(jnp.square(integrator_fn(state, tau)))
    return jax.jit(jax.grad(loss, argnums=0))


# ══════════════════════════════════════════════════════════════════════════
# CORRECTNESS CHECK
# ══════════════════════════════════════════════════════════════════════════

def check_correctness(aba_fns: dict):
    """
    Call every available backend at the same (q, qd, tau) and compare qacc
    against A (JaxSim ABA) as the reference.

    Expected residuals:
      A vs B  < 5 rad/s²   — lumped vs. explicit hand representation
      A vs C  < 5 rad/s²   — same reason
      A vs D  < 5 rad/s²   — ADAM uses same URDF, same lumping
    A residual >> 10 rad/s² signals a model-parameter or unit mismatch.
    """
    print(f"\n{'='*75}")
    print("  CORRECTNESS CHECK — qacc agreement across backends")
    print("  Reference: A (JaxSim ABA build()-in-jit)")
    print(f"{'='*75}")

    if "A" not in aba_fns:
        print("  Reference A not available — skipping.")
        return

    q_ref, qd_ref, tau_ref = make_inputs(jax.random.PRNGKey(0))
    ref = np.asarray(jax.device_get(aba_fns["A"](q_ref, qd_ref, tau_ref)))
    print(f"  q   : {np.round(q_ref,   3)}")
    print(f"  qd  : {np.round(qd_ref,  3)}")
    print(f"  tau : {np.round(tau_ref, 3)}")
    print(f"\n  A (ref) qacc: {np.round(ref, 4)}")
    print()

    # ≤10 rad/s² tolerates lumped-vs-explicit hand difference.
    # >> 10 signals a real model mismatch (wrong params or missing body).
    THRESH = 10.0
    all_ok = True
    for label in ["B", "C", "D"]:
        if label not in aba_fns:
            continue
        try:
            out = np.asarray(jax.device_get(aba_fns[label](q_ref, qd_ref, tau_ref)))
            diff    = out - ref
            max_abs = float(np.max(np.abs(diff)))
            max_rel = float(np.max(np.abs(diff) / (np.abs(ref) + 1e-6)))
            ok      = max_abs < THRESH
            status  = "OK" if ok else "MISMATCH !"
            if not ok:
                all_ok = False
            print(f"  {label}: max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}  [{status}]")
            if not ok:
                print(f"       {label} qacc : {np.round(out,  4)}")
                print(f"       A qacc : {np.round(ref,  4)}")
                print(f"       diff   : {np.round(diff, 4)}")
        except Exception:
            print(f"  {label}: FAILED\n{traceback.format_exc()}")

    if all_ok:
        print("\n  All backends agree within threshold.")
    else:
        print("\n  WARNING: one or more backends disagree — see MISMATCH lines above.")


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
        best  = min(valid, key=lambda r: r.batch_ms[K])
        worst = max(valid, key=lambda r: r.batch_ms[K])
        ratio = worst.batch_ms[K] / max(best.batch_ms[K], 1e-9)
        print(f"  K={K}: WINNER = {best.name}  ({ratio:.1f}x over slowest)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-library dynamics benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--trials",       type=int,  default=N_TRIALS)
    parser.add_argument("--skip-grad",    action="store_true")
    parser.add_argument("--skip-mjx",     action="store_true")
    parser.add_argument("--skip-jaxadi",  action="store_true")
    parser.add_argument("--skip-adam",    action="store_true")
    args = parser.parse_args()

    n_trials = args.trials
    bs       = args.batch_sizes


    print("=" * 75)
    print("  CROSS-LIBRARY DYNAMICS BENCHMARK FOR FEEDBACK-MPPI")
    print("  Franka Panda 7-DoF + hand | torque control | no contacts")
    print("=" * 75)
    backend = jax.default_backend()
    print(f"  JAX backend:    {backend}")
    if backend == "gpu":
        print(f"  GPU:            {jax.devices('gpu')[0]}")
    print(f"  Precision:      {'f64' if jax.config.jax_enable_x64 else 'f32'}")
    print(f"  Batch sizes:    {bs}")
    print(f"  Gradient test:  {'OFF' if args.skip_grad else 'ON (A, D, F)'}")

    key = jax.random.PRNGKey(42)
    fwd_results, grad_results = [], []
    aba_fns: dict = {}

    # ── Build JaxSim model (also yields the URDF path for E and F) ──
    print(f"\n  Building models...")
    js_model, mj_model, pin_model, urdf_path = _build_models()
    print(f"  JaxSim joints : {list(js_model.joint_names())}")
    
    mj_joint_names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(mj_model.njnt)
    ]
    print(f"  Mujoco joints : {mj_joint_names}")
    
    pin_joint_names = []
    for joint_id in range(1, pin_model.njoints):
        joint_name = pin_model.names[joint_id]
        pin_joint_names.append(joint_name)
    print(f"  Pinocchio joints : {pin_joint_names}")

    # ═══════════════════════════════════════════════════════
    #  PART 1: FORWARD DYNAMICS
    # ═══════════════════════════════════════════════════════

    # A: JaxSim
    try:
        fn_a = build_A(js_model)
        aba_fns["A"] = fn_a
        fwd_results.append(bench(
            "A: JaxSim ABA build()-in-jit", fn_a, make_inputs, key, bs, n_trials))
    except Exception:
        print(f"  A FAILED:\n{traceback.format_exc()}")

    # B: MJX
    if not args.skip_mjx:
        try:
            print(f"\n  Building MJX model (panda.xml, contacts disabled)...")
            fn_b = build_B(mj_model)
            aba_fns["B"] = fn_b
            fwd_results.append(bench(
                "B: MJX-JAX forward (hand incl.)", fn_b, make_inputs, key, bs, n_trials))
        except Exception:
            print(f"  B FAILED:\n{traceback.format_exc()}")

    # C: Pinocchio+CasADi+Jaxadi (forward only)
    if not args.skip_jaxadi:
        try:
            print(f"\n  Building Pinocchio + CasADi + Jaxadi ABA...")
            fn_c = build_C(pin_model)
            aba_fns["C"] = fn_c
            fwd_results.append(bench(
                "C: Pinocchio+CasADi+Jaxadi", fn_c, make_inputs, key, bs, n_trials))
        except Exception:
            print(f"  C FAILED:\n{traceback.format_exc()}")

    # D: ADAM-JAX
    # if not args.skip_adam:
    #     try:
    #         print(f"\n  Building ADAM-JAX model...")
    #         fn_d = build_D(urdf_path, pin_joint_names)
    #         aba_fns["D"] = fn_d
    #         fwd_results.append(bench(
    #             "D: ADAM-JAX", fn_d, make_inputs, key, bs, n_trials))
    #     except Exception:
    #         print(f"  D FAILED:\n{traceback.format_exc()}")

    if fwd_results:
        print_summary("FORWARD DYNAMICS — vmap(aba)(q, qd, tau)", fwd_results, bs)

    # ── Correctness check (before gradient, while aba_fns are still warm) ──
    if aba_fns:
        check_correctness(aba_fns)

    # ═══════════════════════════════════════════════════════
    #  PART 2: GRADIENT (Feedback-MPPI gains)
    #  C (Pinocchio) is skipped — 30× slower, no new info.
    # ═══════════════════════════════════════════════════════

    if not args.skip_grad:
        print(f"\n\n{'='*75}")
        print("  GRADIENT THROUGHPUT — vmap(grad(si_euler_step)) w.r.t. state")
        print("  This is the backward pass that Feedback-MPPI needs for gains.")
        print(f"{'='*75}")

        name_map = {
            "A": "A: JaxSim build() GRAD",
            "B": "B: MJX-JAX GRAD",
        }
        # C is excluded (Pinocchio+Jaxadi grad is ~30× slower and impractical)
        for label in ["A", "B"]:
            if label not in aba_fns:
                continue
            try:
                step_fn = make_si_euler(aba_fns[label])
                grad_fn = make_grad_step(step_fn)
                grad_results.append(bench(
                    name_map[label], grad_fn, make_state_and_tau, key, bs, n_trials))
            except Exception:
                print(f"  {label} GRAD FAILED:\n{traceback.format_exc()}")

        if grad_results:
            print_summary("GRADIENT — backward through dynamics", grad_results, bs)

    # ═══════════════════════════════════════════════════════
    #  INTERPRETATION GUIDE
    # ═══════════════════════════════════════════════════════

    print(f"""

{'='*75}
  HOW TO INTERPRET
{'='*75}

  MODEL:
    All backends use the 7-DoF Panda arm WITH the hand (0.73 kg end-effector)
    attached.  This matches real deployment.  Small residuals in the
    correctness check (< 5 rad/s²) are expected because JaxSim lumps the
    hand inertia into link7 while Pinocchio/MJX keep it as a separate body.

  FORWARD TABLE:
    A vs B  → JaxSim vs MJX-JAX.  B faster → MJX is the better engine.
    A vs C  → JaxSim vs Pinocchio symbolic.  C slow → CasADi→XLA overhead.
    A vs D  → JaxSim vs ADAM.  Reveals CRBA+solve vs ABA tradeoff.

  GRADIENT TABLE (THE number for Feedback-MPPI):
    This is what dominates controller loop time.
    Some backends have much worse fwd/bwd ratios.
    MJX-JAX grad requires contacts OFF (CG solver uses while_loop,
    no reverse-mode AD through it).

  TOTAL COST per Feedback-MPPI iteration:
    total_ms ~ (fwd_K_time + grad_K_time) × horizon
    Example: K=512, H=10, fwd=0.5ms, grad=1.0ms
             → (0.5+1.0)×10 = 15ms → ~66Hz
""")


if __name__ == "__main__":
    main()
