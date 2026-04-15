"""
Diagnostic: pinpoint *why* JaxSim / MJX / Pinocchio disagree on Panda qacc.

Run after `bench_dynamics.py` flags a mismatch.  Each section targets a
specific possible source of disagreement and prints a side-by-side table:

  SECTION 1 — Model structure: joint/body counts, names, ordering.
  SECTION 2 — Inertial parameters: mass / CoM / inertia per link (URDF vs MJCF).
  SECTION 3 — Gravity vector actually used by each backend.
  SECTION 4 — MJX vs MuJoCo-C (the gold standard) — if these disagree, the
              MJX implementation itself is wrong.  If they AGREE, any
              remaining difference is a URDF-vs-MJCF *model* difference, not
              an MJX bug.
  SECTION 5 — Gravity-only test (q=0, qd=0, tau=0).  Isolates g + M^-1.
  SECTION 6 — Unit-torque sweep (q=0, qd=0, tau=e_i).  Probes M^-1 column-by-column.
  SECTION 7 — Finger-zeroed test.  Same (q, qd, tau) as the bench, but with
              finger entries forced to zero.  If the residuals shrink, the
              mismatch is dominated by the way each backend treats the
              (unphysical, |q|>>joint-limit) finger configuration.

Usage:
    pixi run -e cuda python tests/debug_dynamics_diff.py
"""
from __future__ import annotations

import dataclasses
import numpy as np

import jax
import jax.numpy as jnp

import jaxsim.api as js
import jaxsim.parsers.rod as rodp
from robot_descriptions.panda_description import URDF_PATH
from robot_descriptions.panda_mj_description import MJCF_PATH

import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cpin
from jaxadi import convert

import mujoco
from mujoco import mjx


# ── Model builders (match bench_dynamics.py) ──────────────────────────────

def build_jaxsim():
    desc = rodp.build_model_description(URDF_PATH, is_urdf=True)
    desc = dataclasses.replace(desc, fixed_base=True)
    return js.model.JaxSimModel.build(model_description=desc)


def build_pinocchio():
    return pin.buildModelFromUrdf(URDF_PATH)


def build_mujoco():
    mj = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # Same "fairness fixes" as bench_dynamics.py
    mj.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mj.actuator_gainprm[:] = 0.0
    mj.actuator_biasprm[:]  = 0.0
    mj.dof_damping[:]       = 0.0
    mj.dof_frictionloss[:]  = 0.0
    mj.dof_armature[:]      = 0.0
    return mj


def make_jaxsim_fn(js_model):
    def aba(q, qd, tau):
        data = js.data.JaxSimModelData.build(
            model=js_model,
            joint_positions=q,
            joint_velocities=qd,
        )
        _, ddq = js.model.forward_dynamics_aba(
            model=js_model, data=data, joint_forces=tau,
        )
        return ddq
    return jax.jit(aba)


def make_pinocchio_fn(pin_model):
    """Pure-Pinocchio ABA (no jaxadi).  Reference implementation."""
    data = pin_model.createData()

    def aba_np(q, qd, tau):
        return np.array(pin.aba(pin_model, data, q, qd, tau))
    return aba_np


def make_mjx_fn(mj_model):
    mjx_model = mjx.put_model(mj_model)
    mjx_data_tpl = mjx.make_data(mj_model)

    def fwd(q, qd, tau):
        data = mjx_data_tpl.replace(qpos=q, qvel=qd, qfrc_applied=tau)
        data = mjx.forward(mjx_model, data)
        return data.qacc
    return jax.jit(fwd)


def make_mj_native_fn(mj_model):
    """Native MuJoCo (CPU, C++).  The *canonical* implementation —
    if MJX disagrees with this, MJX itself is wrong."""
    mj_data = mujoco.MjData(mj_model)

    def fwd(q, qd, tau):
        mj_data.qpos[:] = q
        mj_data.qvel[:] = qd
        mj_data.qfrc_applied[:] = tau
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        return np.array(mj_data.qacc)
    return fwd


# ── SECTION 1 ─────────────────────────────────────────────────────────────

def section1_structure(js_model, pin_model, mj_model):
    print("\n" + "=" * 78)
    print(" SECTION 1 — MODEL STRUCTURE")
    print("=" * 78)

    js_joints = list(js_model.joint_names())
    pin_joints = [pin_model.names[i] for i in range(1, pin_model.njoints)]
    mj_joints = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(mj_model.njnt)]

    print(f"\n  JaxSim    : nq={len(js_joints):2d}   joints={js_joints}")
    print(f"  Pinocchio : nq={pin_model.nq:2d}   joints={pin_joints}")
    print(f"  MuJoCo    : nq={mj_model.nq:2d}   joints={mj_joints}")

    if not (len(js_joints) == pin_model.nq == mj_model.nq):
        print("\n  WARNING: DOF counts differ — inputs will not align!")
    # Warn if finger joint counts differ (mimic expansion)
    finger_like = lambda n: n and ("finger" in n.lower())
    nf_js  = sum(finger_like(n) for n in js_joints)
    nf_pin = sum(finger_like(n) for n in pin_joints)
    nf_mj  = sum(finger_like(n) for n in mj_joints)
    print(f"\n  Finger joints: JaxSim={nf_js}  Pinocchio={nf_pin}  MuJoCo={nf_mj}")
    if not (nf_js == nf_pin == nf_mj):
        print("  → Finger-joint counts differ.  Likely due to URDF <mimic> "
              "being expanded by one parser but not another.")


# ── SECTION 2 ─────────────────────────────────────────────────────────────

def section2_inertial_params(pin_model, mj_model):
    print("\n" + "=" * 78)
    print(" SECTION 2 — INERTIAL PARAMETERS (URDF vs MJCF)")
    print("=" * 78)

    print(f"\n  URDF (Pinocchio):")
    print(f"  {'body':<22s} {'mass':>10s} {'com_x':>10s} {'com_y':>10s} {'com_z':>10s}")
    for i in range(1, pin_model.njoints):
        name = pin_model.names[i]
        I = pin_model.inertias[i]
        c = np.array(I.lever)
        print(f"  {name:<22s} {I.mass:10.5f} {c[0]:10.5f} {c[1]:10.5f} {c[2]:10.5f}")

    print(f"\n  MJCF (MuJoCo):")
    print(f"  {'body':<22s} {'mass':>10s} {'com_x':>10s} {'com_y':>10s} {'com_z':>10s}")
    for i in range(1, mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        m = mj_model.body_mass[i]
        c = mj_model.body_ipos[i]
        print(f"  {name:<22s} {m:10.5f} {c[0]:10.5f} {c[1]:10.5f} {c[2]:10.5f}")

    total_pin = sum(pin_model.inertias[i].mass for i in range(1, pin_model.njoints))
    total_mj  = float(np.sum(mj_model.body_mass[1:]))
    print(f"\n  Total mass  URDF={total_pin:.5f}   MJCF={total_mj:.5f}   "
          f"Δ={total_pin - total_mj:+.5f} kg")


# ── SECTION 3 ─────────────────────────────────────────────────────────────

def section3_gravity(pin_model, mj_model):
    import jaxsim.math as jsm
    print("\n" + "=" * 78)
    print(" SECTION 3 — GRAVITY")
    print("=" * 78)
    print(f"\n  JaxSim    STANDARD_GRAVITY : {jsm.STANDARD_GRAVITY}")
    print(f"  Pinocchio model.gravity    : {np.array(pin_model.gravity.linear)}")
    print(f"  MuJoCo    opt.gravity      : {np.array(mj_model.opt.gravity)}")


# ── SECTION 4 ─────────────────────────────────────────────────────────────

def section4_mjx_vs_native(mj_model, mjx_fn, mjn_fn, q, qd, tau):
    """MJX vs native MuJoCo C++ — this is the MJX correctness acid test."""
    print("\n" + "=" * 78)
    print(" SECTION 4 — MJX vs native MuJoCo (same MJCF)")
    print("=" * 78)
    a_mjx = np.asarray(jax.device_get(mjx_fn(q, qd, tau)))
    a_mj  = mjn_fn(np.asarray(q), np.asarray(qd), np.asarray(tau))
    err   = a_mjx - a_mj
    print(f"\n  qacc (MJX)      : {np.round(a_mjx, 5)}")
    print(f"  qacc (mj_forward): {np.round(a_mj,  5)}")
    print(f"  max |err|        : {np.max(np.abs(err)):.3e}")
    if np.max(np.abs(err)) < 1e-4:
        print("  → MJX matches reference MuJoCo.  MJX implementation is correct.")
        print("    Any residual vs JaxSim/Pinocchio is a MODEL (URDF vs MJCF) issue,")
        print("    not an MJX bug.")
    else:
        print("  → MJX disagrees with reference MuJoCo.  MJX usage is wrong")
        print("    (likely: stale data, missing smooth-dyn call, or disabled flag).")


# ── SECTION 5 ─────────────────────────────────────────────────────────────

def section5_gravity_only(fns, nq):
    """q=0, qd=0, tau=0  =>  qacc = -M(0)^{-1} g.
    Nullifies everything except gravity and inertia matrix.  Very clean."""
    print("\n" + "=" * 78)
    print(" SECTION 5 — GRAVITY-ONLY TEST (q=0, qd=0, tau=0)")
    print("=" * 78)
    q = jnp.zeros(nq); qd = jnp.zeros(nq); tau = jnp.zeros(nq)
    out = {}
    for name, fn in fns.items():
        a = fn(q, qd, tau) if name != "Pinocchio" else fn(
            np.zeros(nq), np.zeros(nq), np.zeros(nq))
        out[name] = np.asarray(jax.device_get(a))
    print(f"\n  {'backend':<14s} " + " ".join(f"q[{i}]".rjust(10) for i in range(nq)))
    for name, a in out.items():
        print(f"  {name:<14s} " + " ".join(f"{v:10.4f}" for v in a))
    if len(out) >= 2:
        ref = next(iter(out.values()))
        for name, a in out.items():
            err = float(np.max(np.abs(a - ref)))
            print(f"  max |Δ vs {next(iter(out)):s}|  {name:<12s} {err:.3e}")


# ── SECTION 6 ─────────────────────────────────────────────────────────────

def section6_unit_torques(fns, nq):
    """q=0, qd=0, tau=e_i  =>  column i of M(0)^{-1} (plus gravity offset).
    Subtract off the gravity-only qacc to isolate M^{-1}."""
    print("\n" + "=" * 78)
    print(" SECTION 6 — M(0)^{-1} PROBE (tau = e_i, q=0, qd=0)")
    print("=" * 78)
    q  = jnp.zeros(nq); qd = jnp.zeros(nq)
    q_np, qd_np = np.zeros(nq), np.zeros(nq)

    def call(name, fn, t):
        if name == "Pinocchio":
            return fn(q_np, qd_np, np.asarray(t))
        return np.asarray(jax.device_get(fn(q, qd, t)))

    # Gravity-only offset
    offsets = {n: call(n, f, jnp.zeros(nq)) for n, f in fns.items()}

    Minv = {n: np.zeros((nq, nq)) for n in fns}
    for i in range(nq):
        e = jnp.zeros(nq).at[i].set(1.0)
        for n, f in fns.items():
            Minv[n][:, i] = call(n, f, e) - offsets[n]

    # Compare Minv between backends
    names = list(fns.keys())
    for a, b in [(names[i], names[j]) for i in range(len(names))
                 for j in range(i + 1, len(names))]:
        err = np.max(np.abs(Minv[a] - Minv[b]))
        print(f"  max |M^-1[{a}] - M^-1[{b}]| = {err:.3e}")

    # Print arm-block (top-left 7x7) of the first backend's M^-1 for sanity
    n0 = names[0]
    print(f"\n  {n0} M^-1 (arm 7×7 block):")
    for row in Minv[n0][:7, :7]:
        print("   " + "  ".join(f"{v:+8.4f}" for v in row))


# ── SECTION 7 ─────────────────────────────────────────────────────────────

def section7_finger_zeroed(fns, nq, seed=0):
    """Same as bench random input, but zero the finger DoFs (indices 7, 8).
    If this shrinks the residuals, the mismatch is dominated by how each
    backend handles out-of-range finger configurations."""
    print("\n" + "=" * 78)
    print(" SECTION 7 — SAME RANDOM INPUT, FINGERS ZEROED")
    print("=" * 78)
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    q   = jax.random.uniform(k1, (nq,), minval=-2.0, maxval=2.0)
    qd  = jax.random.uniform(k2, (nq,), minval=-1.0, maxval=1.0)
    tau = jax.random.uniform(k3, (nq,), minval=-10.0, maxval=10.0)
    # Zero fingers (indices 7, 8 — last two if nq == 9)
    if nq >= 9:
        q   = q.at[7:].set(0.0)
        qd  = qd.at[7:].set(0.0)
        tau = tau.at[7:].set(0.0)

    def call(name, fn):
        if name == "Pinocchio":
            return fn(np.asarray(q), np.asarray(qd), np.asarray(tau))
        return np.asarray(jax.device_get(fn(q, qd, tau)))

    out = {n: call(n, f) for n, f in fns.items()}
    print(f"\n  Input q  : {np.round(q,  3)}")
    print(f"  Input qd : {np.round(qd, 3)}")
    print(f"  Input tau: {np.round(tau, 3)}")
    for n, a in out.items():
        print(f"  {n:<12s} qacc: {np.round(a, 4)}")
    names = list(out)
    for a, b in [(names[i], names[j]) for i in range(len(names))
                 for j in range(i + 1, len(names))]:
        err = float(np.max(np.abs(out[a] - out[b])))
        print(f"  max |Δ {a} vs {b}| = {err:.3e}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Building models...")
    js_model  = build_jaxsim()
    pin_model = build_pinocchio()
    mj_model  = build_mujoco()

    # Sanity: we only proceed if the three models agree on DoF count.
    nq_js, nq_pin, nq_mj = (len(js_model.joint_names()),
                            pin_model.nq, mj_model.nq)

    fns_jax = {
        "JaxSim":    make_jaxsim_fn(js_model),
        "MJX":       make_mjx_fn(mj_model),
        "Pinocchio": make_pinocchio_fn(pin_model),
    }
    mjx_fn  = fns_jax["MJX"]
    mjn_fn  = make_mj_native_fn(mj_model)

    section1_structure(js_model, pin_model, mj_model)
    section2_inertial_params(pin_model, mj_model)
    section3_gravity(pin_model, mj_model)

    # Use the same random input the bench uses for reproducibility
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    NQ = nq_mj
    q   = jax.random.uniform(k1, (NQ,), minval=-2.0, maxval=2.0)
    qd  = jax.random.uniform(k2, (NQ,), minval=-1.0, maxval=1.0)
    tau = jax.random.uniform(k3, (NQ,), minval=-10.0, maxval=10.0)
    section4_mjx_vs_native(mj_model, mjx_fn, mjn_fn, q, qd, tau)

    # Sections 5-7 require matching DoF count across all three backends.
    if nq_js == nq_pin == nq_mj:
        section5_gravity_only(fns_jax, NQ)
        section6_unit_torques(fns_jax, NQ)
        section7_finger_zeroed(fns_jax, NQ)
    else:
        print("\n  [sections 5-7 skipped: DoF counts differ across backends]")

    print("\n" + "=" * 78)
    print(" SUMMARY — where the mismatch comes from")
    print("=" * 78)
    print("""
  Read the sections top-to-bottom:
    • SECTION 1 identifies structural mismatches (nq/nv, finger mimic).
    • SECTION 2 identifies parameter mismatches (masses, CoMs).
    • SECTION 3 identifies gravity mismatches (9.81 vs 9.80665 etc.).
    • SECTION 4 tells you whether MJX itself is correct:
        - If MJX == native MuJoCo within 1e-5, MJX is trustworthy and any
          residual vs JaxSim/Pinocchio is a URDF-vs-MJCF model issue.
        - If not, the MJX glue code is wrong (usually a missed disableflag
          or a stale field in the mjx.Data we replace).
    • SECTION 5 is the cleanest inter-library test (no velocities, no
      torques, no unphysical inputs).  Any non-trivial residual here is
      pure parameter difference between URDF and MJCF.
    • SECTION 6 isolates the mass matrix.
    • SECTION 7 checks whether the default random input (which pushes
      fingers to |q|=2 m, far beyond the 0.04 m joint limit) inflates the
      residual — because finger mass at a huge lever arm changes the
      effective hand inertia and any per-body parameter difference gets
      amplified.
""")


if __name__ == "__main__":
    main()
