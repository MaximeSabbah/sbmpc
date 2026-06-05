"""Offline PREGRASP MPPI tuning and MuJoCo validation.

This script is intentionally headless and metrics-first. It uses the sbmpc MPPI
controller with a cheap joint-space torque surrogate for rollouts, then applies
only the controller's torque output to the native MuJoCo Panda plant. It is meant
for cost/controller design before ROS timing constraints enter the loop.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
    PANDA_SCENE_PATH,
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.ocp import load_ocp_config, with_weight_overrides
from sbmpc.simulation import build_model_and_solver

FR3_TAU = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=float)
FR3_VEL = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=float)
JOINT_LABELS = tuple(f"j{i}" for i in range(1, 8))


@dataclass(frozen=True)
class PlantHandles:
    model: mujoco.MjModel
    data: mujoco.MjData
    arm_qadr: np.ndarray
    arm_vadr: np.ndarray


@dataclass(frozen=True)
class RunMetrics:
    rows: list[dict[str, float]]
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    tau_preclip: np.ndarray
    tau_ff: np.ndarray
    ee_error: np.ndarray
    ori_error: np.ndarray
    planning_ms: np.ndarray
    gain_norm: np.ndarray


def _arm_joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
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
            raise ValueError(f"could not find arm joint ending with joint{i}")
        qadr.append(int(model.jnt_qposadr[jid]))
        vadr.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qadr, dtype=int), np.asarray(vadr, dtype=int)


def build_torque_plant(scene_path: str | Path) -> PlantHandles:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    # The arm is driven by qfrc_applied below. Keep gravity/damping/armature, but
    # neutralize any XML actuators so they cannot add hidden control effort.
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    data = mujoco.MjData(model)
    arm_qadr, arm_vadr = _arm_joint_addresses(model)
    return PlantHandles(model=model, data=data, arm_qadr=arm_qadr, arm_vadr=arm_vadr)


def plant_home_state(plant: PlantHandles) -> tuple[np.ndarray, np.ndarray]:
    model, data = plant.model, plant.data
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    mujoco.mj_forward(model, data)
    return (
        data.qpos[plant.arm_qadr].astype(np.float32).copy(),
        data.qvel[plant.arm_vadr].astype(np.float32).copy(),
    )


def _mass_damping_at_home(scene_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    arm_qadr, arm_vadr = _arm_joint_addresses(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass, data.qM)
    inertia = np.maximum(np.diag(mass)[arm_vadr], 1e-3)
    damping = np.asarray(model.dof_damping[arm_vadr], dtype=float)
    del arm_qadr
    return inertia.astype(np.float32), damping.astype(np.float32)


def _gravity_linearization(
    planner: PandaPregraspPlanner,
    q0: np.ndarray,
    *,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    q0 = np.asarray(q0, dtype=np.float32)
    g0 = np.asarray(planner.gravity_torques(jnp.asarray(q0)), dtype=np.float32)
    jac = np.zeros((planner.nu, planner.nq), dtype=np.float32)
    for i in range(planner.nq):
        dq = np.zeros(planner.nq, dtype=np.float32)
        dq[i] = eps
        gp = np.asarray(planner.gravity_torques(jnp.asarray(q0 + dq)), dtype=np.float32)
        gm = np.asarray(planner.gravity_torques(jnp.asarray(q0 - dq)), dtype=np.float32)
        jac[:, i] = (gp - gm) / (2.0 * eps)
    return g0, jac


def make_fast_torque_dynamics(
    planner: PandaPregraspPlanner,
    scene_path: str | Path,
    *,
    inertia_scale: float,
    damping_scale: float,
):
    inertia, damping = _mass_damping_at_home(scene_path)
    g0, gravity_jac = _gravity_linearization(planner, np.asarray(planner.home_q))
    q_home = np.asarray(planner.home_q, dtype=np.float32)

    inertia_j = jnp.asarray(inertia_scale * inertia, dtype=jnp.float32)
    damping_j = jnp.asarray(damping_scale * damping, dtype=jnp.float32)
    g0_j = jnp.asarray(g0, dtype=jnp.float32)
    gravity_jac_j = jnp.asarray(gravity_jac, dtype=jnp.float32)
    q_home_j = jnp.asarray(q_home, dtype=jnp.float32)

    def dynamics(state, inputs, params):
        del params
        q = state[: planner.nq]
        v = state[planner.nq :]
        gravity = g0_j + gravity_jac_j @ (q - q_home_j)
        qdd = (inputs - gravity - damping_j * v) / inertia_j
        return jnp.concatenate([v, qdd]).astype(jnp.float32)

    return dynamics


def pose_errors(planner: PandaPregraspPlanner, q: np.ndarray) -> tuple[np.ndarray, float, float]:
    ee_pos, ee_rot = planner.forward_kinematics(np.asarray(q, dtype=np.float32))
    ee_pos = np.asarray(ee_pos, dtype=float)
    goal_pos = np.asarray(planner.goal_pos, dtype=float)
    goal_rot = np.asarray(planner.goal_rotation, dtype=float)
    position_error = float(np.linalg.norm(ee_pos - goal_pos))
    orientation_error = float(
        (1.0 - np.clip(np.dot(ee_rot[:, 2], goal_rot[:, 2]), -1.0, 1.0))
        + 0.5 * (1.0 - np.clip(np.dot(ee_rot[:, 0], goal_rot[:, 0]), -1.0, 1.0))
    )
    return ee_pos, position_error, orientation_error


def gravity_seed(planner: PandaPregraspPlanner, q: np.ndarray, horizon: int) -> jax.Array:
    tau = planner.gravity_torques(jnp.asarray(q, dtype=jnp.float32))
    tau = jnp.clip(tau, -planner.torque_limits, planner.torque_limits)
    return jnp.tile(tau.astype(jnp.float32), (horizon, 1))




def pd_seed(
    planner: PandaPregraspPlanner,
    q: np.ndarray,
    v: np.ndarray,
    horizon: int,
    *,
    kp_scale: float,
    kd_scale: float,
) -> jax.Array:
    q_j = jnp.asarray(q, dtype=jnp.float32)
    v_j = jnp.asarray(v, dtype=jnp.float32)
    torque_limits = planner.torque_limits
    travel = jnp.maximum(jnp.abs(planner.goal_q - planner.home_q), 0.25)
    kp = kp_scale * 0.6 * torque_limits / travel
    kd = kd_scale * 0.15 * torque_limits / planner.velocity_limits
    tau = planner.gravity_torques(q_j) + kp * (planner.goal_q - q_j) - kd * v_j
    tau = jnp.clip(tau, -torque_limits, torque_limits)
    return jnp.tile(tau.astype(jnp.float32), (horizon, 1))

def parse_weight_overrides(items: list[str]) -> dict[str, float]:
    overrides = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"weight override must be name=value, got {item!r}")
        overrides[name.strip()] = float(value)
    return overrides


def configure_controller(args: argparse.Namespace):
    ocp = load_ocp_config(args.ocp)
    overrides = parse_weight_overrides(args.weight)
    if overrides:
        ocp = with_weight_overrides(ocp, overrides)

    planner = PandaPregraspPlanner(goal_pos=None)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=args.gains)
    config.MPC.dt = args.mpc_dt
    config.MPC.horizon = args.horizon
    config.MPC.num_parallel_computations = args.samples
    config.MPC.num_control_points = args.control_points
    config.MPC.lambda_mpc = args.lambda_mpc
    config.MPC.std_dev_mppi = args.std_dev_scale * planner.torque_limits
    config.MPC.initial_guess = jnp.zeros((args.horizon, planner.nu), dtype=jnp.float32)
    if args.gains:
        config.MPC.gain_method = "exact"
        config.MPC.gain_samples_per_cycle = args.gain_samples
        config.MPC.gain_buffer_size = args.gain_buffer
    objective = PandaPregraspObjective(planner, ocp_config=ocp)
    dynamics = make_fast_torque_dynamics(
        planner,
        args.plant_scene,
        inertia_scale=args.inertia_scale,
        damping_scale=args.damping_scale,
    )
    model, controller = build_model_and_solver(config, objective, custom_dynamics_fn=dynamics)
    del model
    return planner, config, controller, ocp, overrides


def run(args: argparse.Namespace) -> RunMetrics:
    planner, config, controller, ocp, overrides = configure_controller(args)
    plant = build_torque_plant(args.plant_scene)
    q, v = plant_home_state(plant)
    substeps = max(1, int(round(args.control_dt / plant.model.opt.timestep)))
    controller.sampler.optimal_samples = gravity_seed(planner, q, config.MPC.horizon)

    rows: list[dict[str, float]] = []
    qs, vs, taus, tau_preclips, tau_ffs, ee_errors, ori_errors, planning_ms, gain_norms = [], [], [], [], [], [], [], [], []
    goal_q = np.asarray(planner.goal_q, dtype=float)
    desired_state = np.concatenate([goal_q, np.zeros(planner.nv, dtype=float)])

    print(
        f"offline pregrasp: ocp={ocp.name} overrides={overrides or 'none'} "
        f"samples={config.MPC.num_parallel_computations} horizon={config.MPC.horizon} "
        f"ctrl_pts={config.MPC.num_control_points} lambda={config.MPC.lambda_mpc} "
        f"std={args.std_dev_scale} gains={args.gains}",
        flush=True,
    )
    print(f"goal_pos={np.asarray(planner.goal_pos, dtype=float)}", flush=True)
    print(f"goal_q={goal_q}", flush=True)

    for k in range(args.steps):
        if args.warm_start_policy == "gravity_each_step":
            controller.sampler.optimal_samples = gravity_seed(planner, q, config.MPC.horizon)
        elif args.warm_start_policy == "pd_each_step":
            controller.sampler.optimal_samples = pd_seed(
                planner,
                q,
                v,
                config.MPC.horizon,
                kp_scale=args.pd_kp_scale,
                kd_scale=args.pd_kd_scale,
            )
        elif args.warm_start_policy != "gravity_once":
            raise ValueError(f"unsupported warm_start_policy={args.warm_start_policy!r}")

        state = jnp.concatenate(
            [jnp.asarray(q, dtype=jnp.float32), jnp.asarray(v, dtype=jnp.float32)]
        )
        tic = time.perf_counter()
        seq = controller.command(
            state,
            planner.reference_vec,
            shift_guess=True,
            num_steps=args.mppi_steps,
            update_gains=args.gains,
        )
        seq = jax.block_until_ready(seq)
        if args.terminal_hold != "shift":
            if args.terminal_hold == "current_gravity":
                hold_tau = planner.gravity_torques(jnp.asarray(q, dtype=jnp.float32))
            elif args.terminal_hold == "goal_gravity":
                hold_tau = planner.goal_tau
            else:
                raise ValueError(f"unsupported terminal_hold={args.terminal_hold!r}")
            hold_tau = jnp.clip(hold_tau, -planner.torque_limits, planner.torque_limits)
            tail = max(1, min(args.hold_tail_steps, config.MPC.horizon))
            hold_block = jnp.tile(hold_tau.astype(jnp.float32), (tail, 1))
            controller.sampler.optimal_samples = controller.sampler.optimal_samples.at[-tail:, :].set(hold_block)
        elapsed_ms = 1000.0 * (time.perf_counter() - tic)
        tau_ff = np.asarray(seq[0], dtype=float)
        tau_preclip = tau_ff.copy()
        gains = np.asarray(controller.gains_obj.cur_gains, dtype=float)
        if args.apply_feedback:
            measured_state = np.concatenate([q.astype(float), v.astype(float)])
            tau_preclip = tau_ff + args.feedback_scale * (gains @ (measured_state - desired_state))
        tau = np.clip(tau_preclip, -FR3_TAU, FR3_TAU)

        for _ in range(substeps):
            plant.data.qfrc_applied[plant.arm_vadr] = tau
            mujoco.mj_step(plant.model, plant.data)
        q = plant.data.qpos[plant.arm_qadr].astype(np.float32).copy()
        v = plant.data.qvel[plant.arm_vadr].astype(np.float32).copy()

        ee_pos, ee_err, ori_err = pose_errors(planner, q)
        q_err = float(np.linalg.norm(q.astype(float) - goal_q))
        tau_ratio = float(np.max(np.abs(tau) / FR3_TAU))
        vel_ratio = float(np.max(np.abs(v.astype(float)) / FR3_VEL))
        tau_ff_ratio = float(np.max(np.abs(tau_ff) / FR3_TAU))
        tau_preclip_ratio = float(np.max(np.abs(tau_preclip) / FR3_TAU))
        power = float(np.sum(tau * v.astype(float)))
        gain_norm = float(np.linalg.norm(gains))

        row = {
            "step": float(k),
            "time_s": float((k + 1) * args.control_dt),
            "ee_error_m": ee_err,
            "orientation_error": ori_err,
            "q_error_rad": q_err,
            "tau_ratio": tau_ratio,
            "tau_preclip_ratio": tau_preclip_ratio,
            "tau_ff_ratio": tau_ff_ratio,
            "vel_ratio": vel_ratio,
            "sum_power_w": power,
            "planning_ms": elapsed_ms,
            "gain_norm": gain_norm,
        }
        for idx, label in enumerate(JOINT_LABELS):
            row[f"q_{label}"] = float(q[idx])
            row[f"v_{label}"] = float(v[idx])
            row[f"tau_{label}"] = float(tau[idx])
            row[f"tau_preclip_{label}"] = float(tau_preclip[idx])
            row[f"tau_ff_{label}"] = float(tau_ff[idx])
        for idx, axis in enumerate("xyz"):
            row[f"ee_{axis}"] = float(ee_pos[idx])
        rows.append(row)
        qs.append(q.copy())
        vs.append(v.copy())
        taus.append(tau.copy())
        tau_preclips.append(tau_preclip.copy())
        tau_ffs.append(tau_ff.copy())
        ee_errors.append(ee_err)
        ori_errors.append(ori_err)
        planning_ms.append(elapsed_ms)
        gain_norms.append(gain_norm)

        if k == 0 or (k + 1) % args.report_every == 0 or k == args.steps - 1:
            print(
                f"step={k + 1:04d} t={(k + 1) * args.control_dt:5.2f}s "
                f"ee={ee_err:7.4f}m ori={ori_err:7.4f} qerr={q_err:7.4f} "
                f"tau={100*tau_ratio:5.1f}% pre={100*tau_preclip_ratio:5.1f}% "
                f"vel={100*vel_ratio:5.1f}% "
                f"plan={elapsed_ms:7.1f}ms |K|={gain_norm:7.3f}",
                flush=True,
            )

    plant.data.qfrc_applied[:] = 0.0
    return RunMetrics(
        rows=rows,
        q=np.asarray(qs),
        v=np.asarray(vs),
        tau=np.asarray(taus),
        tau_preclip=np.asarray(tau_preclips),
        tau_ff=np.asarray(tau_ffs),
        ee_error=np.asarray(ee_errors),
        ori_error=np.asarray(ori_errors),
        planning_ms=np.asarray(planning_ms),
        gain_norm=np.asarray(gain_norms),
    )


def summarize(metrics: RunMetrics) -> dict[str, float]:
    peak_tau = np.max(np.abs(metrics.tau), axis=0)
    peak_tau_preclip = np.max(np.abs(metrics.tau_preclip), axis=0)
    peak_tau_ff = np.max(np.abs(metrics.tau_ff), axis=0)
    peak_vel = np.max(np.abs(metrics.v), axis=0)
    tau_ratio = peak_tau / FR3_TAU
    tau_preclip_ratio = peak_tau_preclip / FR3_TAU
    tau_ff_ratio = peak_tau_ff / FR3_TAU
    vel_ratio = peak_vel / FR3_VEL
    below_2cm = np.flatnonzero(metrics.ee_error < 0.02)
    first_2cm = int(below_2cm[0]) if below_2cm.size else -1
    summary = {
        "start_ee_error_m": float(metrics.ee_error[0]),
        "min_ee_error_m": float(np.min(metrics.ee_error)),
        "final_ee_error_m": float(metrics.ee_error[-1]),
        "final_orientation_error": float(metrics.ori_error[-1]),
        "peak_tau_ratio": float(np.max(tau_ratio)),
        "peak_tau_preclip_ratio": float(np.max(tau_preclip_ratio)),
        "peak_tau_ff_ratio": float(np.max(tau_ff_ratio)),
        "peak_vel_ratio": float(np.max(vel_ratio)),
        "peak_abs_power_w": float(np.max(np.abs(np.sum(metrics.tau * metrics.v, axis=1)))),
        "median_planning_ms_after_first": float(np.median(metrics.planning_ms[1:])) if metrics.planning_ms.size > 1 else float(metrics.planning_ms[0]),
        "max_gain_norm": float(np.max(metrics.gain_norm)),
        "first_2cm_step": float(first_2cm),
    }
    print("\n===== offline pregrasp summary =====")
    print("joint:            " + " ".join(f"{j:>6s}" for j in JOINT_LABELS))
    print("peak|tau| Nm :  " + " ".join(f"{x:6.1f}" for x in peak_tau))
    print("  %tau limit :  " + " ".join(f"{100*x:6.1f}" for x in tau_ratio))
    print("%preclip tau :  " + " ".join(f"{100*x:6.1f}" for x in tau_preclip_ratio))
    print("    %ff tau :  " + " ".join(f"{100*x:6.1f}" for x in tau_ff_ratio))
    print("peak|vel|    :  " + " ".join(f"{x:6.2f}" for x in peak_vel))
    print("  %vel limit :  " + " ".join(f"{100*x:6.1f}" for x in vel_ratio))
    print(
        f"EE error: start={summary['start_ee_error_m']:.4f}m "
        f"min={summary['min_ee_error_m']:.4f}m "
        f"final={summary['final_ee_error_m']:.4f}m "
        f"<2cm step={first_2cm}"
    )
    print(
        f"limits: peak_tau={100*summary['peak_tau_ratio']:.1f}% "
        f"preclip_tau={100*summary['peak_tau_preclip_ratio']:.1f}% "
        f"ff_tau={100*summary['peak_tau_ff_ratio']:.1f}% "
        f"peak_vel={100*summary['peak_vel_ratio']:.1f}% "
        f"peak|sum_power|={summary['peak_abs_power_w']:.1f}W"
    )
    print(
        f"timing: median_plan_after_first={summary['median_planning_ms_after_first']:.1f}ms "
        f"max|K|={summary['max_gain_norm']:.3f}"
    )
    torque_ok = summary["peak_tau_ratio"] <= 1.0 + 1e-6
    preclip_torque_ok = summary["peak_tau_preclip_ratio"] <= 1.0 + 1e-6
    reached = summary["final_ee_error_m"] < 0.02
    print(
        f"VERDICT: {'PASS' if torque_ok and preclip_torque_ok and reached else 'FAIL'} "
        f"(torque_ok={torque_ok}, preclip_torque_ok={preclip_torque_ok}, reached={reached})"
    )
    return summary


def write_csv(metrics: RunMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics.rows)


def write_plot(metrics: RunMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray([row["time_s"] for row in metrics.rows])
    tau_ratio = np.max(np.abs(metrics.tau) / FR3_TAU, axis=1)
    tau_preclip_ratio = np.max(np.abs(metrics.tau_preclip) / FR3_TAU, axis=1)
    vel_ratio = np.max(np.abs(metrics.v) / FR3_VEL, axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, metrics.ee_error, label="EE position error")
    axes[0].axhline(0.02, color="tab:red", linestyle="--", linewidth=1, label="2 cm")
    axes[0].set_ylabel("m")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, 100.0 * tau_ratio, label="applied max |tau| / limit")
    axes[1].plot(t, 100.0 * tau_preclip_ratio, label="preclip max |tau| / limit", alpha=0.7)
    axes[1].axhline(100.0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("%")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, 100.0 * vel_ratio, label="max |velocity| / limit")
    axes[2].axhline(100.0, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("%")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocp", default="pregrasp_joint_space")
    parser.add_argument("--weight", action="append", default=[], metavar="name=value")
    parser.add_argument("--plant-scene", default=str(PANDA_SCENE_PATH))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--control-dt", type=float, default=0.025)
    parser.add_argument("--mpc-dt", type=float, default=0.025)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--control-points", type=int, default=16)
    parser.add_argument("--mppi-steps", type=int, default=1)
    parser.add_argument("--warm-start-policy", choices=("gravity_once", "gravity_each_step", "pd_each_step"), default="gravity_once")
    parser.add_argument("--pd-kp-scale", type=float, default=1.0)
    parser.add_argument("--pd-kd-scale", type=float, default=1.0)
    parser.add_argument("--terminal-hold", choices=("shift", "current_gravity", "goal_gravity"), default="goal_gravity")
    parser.add_argument("--hold-tail-steps", type=int, default=8)
    parser.add_argument("--lambda", dest="lambda_mpc", type=float, default=0.12)
    parser.add_argument("--std-dev-scale", type=float, default=0.65)
    parser.add_argument("--inertia-scale", type=float, default=1.0)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    parser.add_argument("--gains", action="store_true")
    parser.add_argument("--apply-feedback", action="store_true")
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--gain-samples", type=int, default=64)
    parser.add_argument("--gain-buffer", type=int, default=256)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--csv", default="/tmp/sbmpc_pregrasp_offline.csv")
    parser.add_argument("--plot", default="/tmp/sbmpc_pregrasp_offline.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(args)
    summary = summarize(metrics)
    if args.csv:
        write_csv(metrics, Path(args.csv))
        print(f"csv: {args.csv}")
    if args.plot:
        write_plot(metrics, Path(args.plot))
        print(f"plot: {args.plot}")
    del summary


if __name__ == "__main__":
    main()
