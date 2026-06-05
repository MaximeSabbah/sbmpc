"""Fast, faithful offline reach harness for iterating on the PREGRASP cost/OCP.

Runs the real PandaPregrasp controller (GPU) against a *damped* MuJoCo plant
(CPU ``mj_step`` with the torque applied as ``qfrc_applied``, keeping the scene's
native joint damping/armature/gravity). This is the key fidelity fix over a naive
open-loop test on the solver's undamped model, which diverges.

It reports the same metrics as the ROS limit gate (per-joint peak torque/velocity/
power vs FR3 limits + EE-error convergence), so cost designs can be compared in
seconds before the ROS gate (``validate_sbmpc_sim``) confirms the winner.

Examples::

    # current default OCP (reproduces the gate's saturation)
    python examples/panda_pregrasp_reach_sweep.py
    # try a variant yaml or override weights inline
    python examples/panda_pregrasp_reach_sweep.py --ocp pregrasp \
        --weight joint_velocity=5.0 --weight mechanical_power=0.01
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
import jax.numpy as jnp

from sbmpc.examples.franka_emika_panda.panda_pregrasp import PANDA_SCENE_PATH
from sbmpc.examples.franka_emika_panda.planner_api import PandaPregraspController
from sbmpc.ocp import load_ocp_config, with_weight_overrides

FR3_TAU = np.array([87, 87, 87, 87, 12, 12, 12.0])
FR3_VEL = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])


def _arm_joint_addresses(model):
    """Find the 7 arm joints by name suffix (joint1.., panda_joint1.., fer_joint1..)."""
    qadr, vadr = [], []
    for i in range(1, 8):
        jid = next(
            (
                j
                for j in range(model.njnt)
                if model.joint(j).name.endswith(f"joint{i}")
                and "finger" not in model.joint(j).name
            ),
            None,
        )
        if jid is None:
            raise ValueError(f"could not find arm joint ...joint{i} in scene")
        qadr.append(int(model.jnt_qposadr[jid]))
        vadr.append(int(model.jnt_dofadr[jid]))
    return np.array(qadr), np.array(vadr)


def build_plant(scene_path: str):
    """Load the scene as a torque-controlled, damped plant (actuators neutralized)."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    # Neutralize built-in actuators so only our qfrc_applied torque acts; keep the
    # native dof_damping / dof_armature (this is what makes the plant faithful).
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    data = mujoco.MjData(model)
    arm_qadr, arm_vadr = _arm_joint_addresses(model)
    return model, data, arm_qadr, arm_vadr


def run_reach(ctrl, model, data, arm_qadr, arm_vadr, *, control_dt, n_steps):
    planner = ctrl.planner
    goal = np.asarray(planner.goal_pos, dtype=float)
    substeps = max(1, int(round(control_dt / model.opt.timestep)))

    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    mujoco.mj_forward(model, data)
    q = data.qpos[arm_qadr].astype(np.float32).copy()
    v = data.qvel[arm_vadr].astype(np.float32).copy()

    taus, vs, errs = [], [], []
    for k in range(n_steps):
        out = ctrl.step(q, v, reset_guess=(k == 0))
        tau = np.asarray(out.tau_ff, dtype=float)
        for _ in range(substeps):
            data.qfrc_applied[arm_vadr] = tau
            mujoco.mj_step(model, data)
        q = data.qpos[arm_qadr].astype(np.float32).copy()
        v = data.qvel[arm_vadr].astype(np.float32).copy()
        ee = np.asarray(planner.ee_features(jnp.asarray(q))[0], dtype=float)
        taus.append(tau)
        vs.append(v.copy())
        errs.append(float(np.linalg.norm(ee - goal)))
    data.qfrc_applied[:] = 0.0
    return np.array(taus), np.array(vs), np.array(errs)


def report(label, taus, vs, errs):
    peak_tau = np.max(np.abs(taus), axis=0)
    peak_vel = np.max(np.abs(vs), axis=0)
    power = taus * vs
    converged = errs < 0.02
    first = int(np.argmax(converged)) if converged.any() else -1
    print(f"\n===== {label} =====")
    print("joint:            j1     j2     j3     j4     j5     j6     j7")
    print("peak|tau| Nm :  " + " ".join(f"{x:6.1f}" for x in peak_tau))
    print("  %tau limit :  " + " ".join(f"{x*100:5.0f}" for x in peak_tau / FR3_TAU))
    print("peak|vel|    :  " + " ".join(f"{x:6.2f}" for x in peak_vel))
    print("  %vel limit :  " + " ".join(f"{x*100:5.0f}" for x in peak_vel / FR3_VEL))
    print(
        f"worst torque={np.max(peak_tau/FR3_TAU)*100:.0f}%  "
        f"worst velocity={np.max(peak_vel/FR3_VEL)*100:.0f}%  "
        f"peak|sum power|={np.max(np.abs(power.sum(axis=1))):.0f} W"
    )
    print(
        f"EE err: start={errs[0]:.3f} min={errs.min():.4f} final={errs[-1]:.4f}  "
        f"<2cm at step {first}"
        + ("" if first < 0 else f" (~{first*0.025:.2f}s)")
    )
    within = (np.max(peak_tau / FR3_TAU) <= 0.9) and (np.max(peak_vel / FR3_VEL) <= 0.9)
    reached = errs[-1] < 0.02
    print(f"VERDICT: {'PASS' if (within and reached) else 'FAIL'} "
          f"(within_limits={within}, reached={reached})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ocp", default="pregrasp", help="OCP config name or yaml path")
    p.add_argument("--weight", action="append", default=[], metavar="name=value",
                   help="override a term weight, repeatable")
    p.add_argument("--plant-scene", default=str(PANDA_SCENE_PATH))
    p.add_argument("--control-dt", type=float, default=0.025)
    p.add_argument("--steps", type=int, default=160)
    args = p.parse_args()

    ocp = load_ocp_config(args.ocp)
    overrides = {}
    for item in args.weight:
        name, _, value = item.partition("=")
        overrides[name.strip()] = float(value)
    if overrides:
        ocp = with_weight_overrides(ocp, overrides)

    ctrl = PandaPregraspController(
        gains=True,
        gain_mode="exact_async_feedback",
        reseed_every_step=True,
        compute_running_cost=False,
        compute_task_diagnostics=False,
        ocp_config=ocp,
    )
    print(f"warming up controller (ocp={ocp.name}, overrides={overrides or 'none'}) ...",
          flush=True)
    ctrl.warmup()

    model, data, qa, va = build_plant(args.plant_scene)
    taus, vs, errs = run_reach(
        ctrl, model, data, qa, va, control_dt=args.control_dt, n_steps=args.steps
    )
    report(f"ocp={ocp.name} overrides={overrides or 'none'}", taus, vs, errs)


if __name__ == "__main__":
    main()
