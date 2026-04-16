from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pinocchio as pin
from mujoco import mjx
from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH

from sbmpc.settings import Config, DynamicsModel, RobotConfig
from sbmpc.solvers import BaseObjective


ROOT = Path(__file__).resolve().parents[1]
PANDA_SCENE_PATH = ROOT / "examples" / "panda_pick_place" / "scene.xml"
PANDA_XML_PATH = ROOT / "examples" / "panda_pick_place" / "panda.xml"
PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
PANDA_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
PANDA_TCP_FRAME_NAME = "panda_hand_tcp"

_DESIRED_X = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
_DESIRED_Z = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
_ARM_TORQUE_LIMITS = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=jnp.float32)
PREGRASP_CLEARANCE = 0.05


@dataclass(frozen=True)
class PandaPregraspReference:
    goal_pos: jax.Array
    goal_q: jax.Array
    goal_x_axis: jax.Array
    goal_z_axis: jax.Array
    goal_tau: jax.Array

    def as_vector(self) -> jax.Array:
        return jnp.concatenate(
            [
                self.goal_pos,
                self.goal_q,
                self.goal_x_axis,
                self.goal_z_axis,
                self.goal_tau,
            ]
        )


class PandaPregraspPlanner:
    """MJX + Pinocchio model for a 7-DoF Panda pregrasp phase."""

    def __init__(
        self,
        scene_path: str | Path = PANDA_SCENE_PATH,
        urdf_path: str | Path = PANDA_URDF_PATH,
        goal_pos: jax.Array | None = None,
    ) -> None:
        self.scene_path = str(scene_path)
        self.urdf_path = str(urdf_path)
        self.joint_names = PANDA_ARM_JOINT_NAMES
        self.frame_name = PANDA_TCP_FRAME_NAME

        self.home_q_full, self.object_pos, self.target_pos, self.object_half_height = (
            self._load_mujoco_defaults()
        )

        self.pin_model, self.pin_data = self._build_pinocchio_model()
        self.model = self.pin_model
        self.data = self.pin_data
        self.frame_id = self.pin_model.getFrameId(self.frame_name)

        self.nq = len(self.joint_names)
        self.nv = self.nq
        self.nu = self.nv
        self.nx = self.nq + self.nv

        self.home_q = jnp.asarray(
            np.asarray(self.home_q_full[: self.nq]), dtype=jnp.float32
        )
        self.torque_limits = _ARM_TORQUE_LIMITS

        self._dynamics_jax, self._ee_features_jax = self._build_mjx_functions(
            str(PANDA_XML_PATH)
        )

        home_pos, home_rot = self.forward_kinematics(self.home_q)
        self.home_pos = jnp.asarray(home_pos, dtype=jnp.float32)
        self.home_rotation = jnp.asarray(home_rot, dtype=jnp.float32)

        self.pregrasp_offset = jnp.array(
            [0.0, 0.0, self.object_half_height + PREGRASP_CLEARANCE],
            dtype=jnp.float32,
        )
        self.goal_pos = (
            self.object_pos + self.pregrasp_offset
            if goal_pos is None
            else jnp.asarray(goal_pos, dtype=jnp.float32)
        )
        self.goal_rotation = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        self.goal_q = jnp.asarray(
            self.solve_ik(np.asarray(self.goal_pos), np.asarray(self.goal_rotation)),
            dtype=jnp.float32,
        )
        self.goal_tau = self.gravity_torques(self.goal_q)

        self.reference = PandaPregraspReference(
            goal_pos=self.goal_pos,
            goal_q=self.goal_q,
            goal_x_axis=self.goal_rotation[:, 0],
            goal_z_axis=self.goal_rotation[:, 2],
            goal_tau=self.goal_tau,
        )
        self.reference_vec = self.reference.as_vector()

    def _load_mujoco_defaults(self) -> tuple[jax.Array, jax.Array, jax.Array, float]:
        mj_model = mujoco.MjModel.from_xml_path(self.scene_path)
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home_q = jnp.asarray(mj_model.key_qpos[key_id][:9], dtype=jnp.float32)
        object_body_idx = mj_model.body("object").id
        target_body_idx = mj_model.body("target").id
        object_geom_idx = mj_model.geom("object_geom").id
        object_pos = jnp.asarray(mj_model.body_pos[object_body_idx], dtype=jnp.float32)
        target_pos = jnp.asarray(mj_model.body_pos[target_body_idx], dtype=jnp.float32)
        object_half_height = float(mj_model.geom_size[object_geom_idx, 2])
        return home_q, object_pos, target_pos, object_half_height

    def _build_pinocchio_model(self) -> tuple[pin.Model, pin.Data]:
        full_model = pin.buildModelFromUrdf(self.urdf_path)
        finger_joint_ids = [
            full_model.getJointId(name) for name in PANDA_FINGER_JOINT_NAMES
        ]
        reduced_model = pin.buildReducedModel(
            full_model,
            finger_joint_ids,
            np.asarray(self.home_q_full, dtype=np.float64),
        )
        return reduced_model, reduced_model.createData()

    def _build_mjx_functions(self, panda_xml_path: str) -> tuple[callable, callable]:
        mj_model = mujoco.MjModel.from_xml_path(panda_xml_path)
        torque_limits_np = np.asarray(self.torque_limits)
        mj_model.actuator_gainprm[:7, 0] = 1.0
        mj_model.actuator_gainprm[:7, 1:] = 0.0
        mj_model.actuator_biasprm[:7, :] = 0.0
        mj_model.actuator_gainprm[7:, :] = 0.0
        mj_model.actuator_biasprm[7:, :] = 0.0
        mj_model.actuator_ctrlrange[:7, 0] = -torque_limits_np
        mj_model.actuator_ctrlrange[:7, 1] = torque_limits_np
        mj_model.dof_damping[:] = 0.0
        mj_model.dof_armature[:] = 0.0
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mjx_model = mjx.put_model(mj_model)
        mjx_data_template = mjx.put_data(mj_model, mujoco.MjData(mj_model))
        gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        nq_arm = self.nq
        n_pad = mj_model.nq - nq_arm
        n_ctrl_pad = mj_model.nu - nq_arm
        finger_pad = jnp.zeros(n_pad, dtype=jnp.float32)
        ctrl_pad = jnp.zeros(n_ctrl_pad, dtype=jnp.float32)

        @jax.jit
        def dynamics_fn(state: jax.Array, inputs: jax.Array) -> jax.Array:
            q = state[:nq_arm].astype(jnp.float32)
            v = state[nq_arm:].astype(jnp.float32)
            q_full = jnp.concatenate([q, finger_pad])
            v_full = jnp.concatenate([v, finger_pad])
            ctrl_full = jnp.concatenate([inputs.astype(jnp.float32), ctrl_pad])
            data = mjx_data_template.replace(qpos=q_full, qvel=v_full, ctrl=ctrl_full)
            data = mjx.forward(mjx_model, data)
            return jnp.concatenate([v, data.qacc[:nq_arm]]).astype(jnp.float32)

        @jax.jit
        def ee_features_fn(q: jax.Array) -> jax.Array:
            q_full = jnp.concatenate([q.astype(jnp.float32), finger_pad])
            data = mjx_data_template.replace(qpos=q_full)
            data = mjx.forward(mjx_model, data)
            ee_xmat = data.site_xmat[gripper_site_id].reshape(3, 3)
            return jnp.concatenate([
                data.site_xpos[gripper_site_id],
                ee_xmat[:, 0],
                ee_xmat[:, 2],
            ]).astype(jnp.float32)

        return dynamics_fn, ee_features_fn

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_np = np.asarray(q, dtype=np.float64)
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q_np)
        frame = self.pin_data.oMf[self.frame_id]
        return frame.translation.copy(), frame.rotation.copy()

    def solve_ik(
        self,
        goal_pos: np.ndarray,
        goal_rot: np.ndarray,
        max_iters: int = 100,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """Solve the nominal PREGRASP pose with the same site-Jacobian IK as Hydrax."""
        mj_model = mujoco.MjModel.from_xml_path(self.scene_path)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mujoco.mj_forward(mj_model, mj_data)

        gripper_site_idx = mj_model.site("gripper").id
        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))

        for _ in range(max_iters):
            mujoco.mj_forward(mj_model, mj_data)
            site_rot = mj_data.site_xmat[gripper_site_idx].reshape(3, 3)
            ee_pos = mj_data.site_xpos[gripper_site_idx]
            pos_err = np.asarray(goal_pos) - ee_pos
            ori_err = 0.5 * (
                np.cross(site_rot[:, 0], goal_rot[:, 0])
                + np.cross(site_rot[:, 1], goal_rot[:, 1])
                + np.cross(site_rot[:, 2], goal_rot[:, 2])
            )
            if np.linalg.norm(pos_err) < tol and np.linalg.norm(ori_err) < tol:
                break

            mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, gripper_site_idx)
            jacobian = np.vstack([jacp[:, : self.nq], jacr[:, : self.nq]])
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-3 * np.eye(6),
                np.concatenate([pos_err, ori_err]),
            )
            mj_data.qpos[: self.nq] += np.clip(dq, -0.05, 0.05)
            mj_data.qpos[: self.nq] = np.clip(
                mj_data.qpos[: self.nq],
                mj_model.jnt_range[: self.nq, 0],
                mj_model.jnt_range[: self.nq, 1],
            )

        return mj_data.qpos[: self.nq].astype(np.float32).copy()

    def gravity_torques(self, q: jax.Array) -> jax.Array:
        q_np = np.asarray(q, dtype=np.float64)
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_np)
        return jnp.asarray(self.pin_data.g, dtype=jnp.float32)

    def inverse_dynamics(
        self, q: jax.Array, v: jax.Array, ddq: jax.Array
    ) -> jax.Array:
        tau = pin.rnea(
            self.pin_model, self.pin_data,
            np.asarray(q, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            np.asarray(ddq, dtype=np.float64),
        )
        return jnp.asarray(tau, dtype=jnp.float32)

    def _cubic_joint_trajectory(
        self,
        q_start: jax.Array,
        v_start: jax.Array,
        q_goal: jax.Array,
        horizon: int,
        dt: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Cubic joint trajectory with current velocity and zero terminal velocity."""
        q_start = jnp.asarray(q_start, dtype=jnp.float32)
        v_start = jnp.asarray(v_start, dtype=jnp.float32)
        q_goal = jnp.asarray(q_goal, dtype=jnp.float32)
        dt = jnp.asarray(dt, dtype=jnp.float32)
        t_final = jnp.maximum(jnp.asarray((horizon - 1), dtype=jnp.float32) * dt, 1e-6)
        time = (jnp.arange(horizon, dtype=jnp.float32) * dt)[:, jnp.newaxis]

        v_goal = jnp.zeros_like(v_start)
        delta_q = q_goal - q_start
        c0 = q_start
        c1 = v_start
        c2 = (3.0 * delta_q - (2.0 * v_start + v_goal) * t_final) / (t_final**2)
        c3 = (-2.0 * delta_q + (v_start + v_goal) * t_final) / (t_final**3)

        q = c0 + c1 * time + c2 * time**2 + c3 * time**3
        v = c1 + 2.0 * c2 * time + 3.0 * c3 * time**2
        ddq = 2.0 * c2 + 6.0 * c3 * time
        return q.astype(jnp.float32), v.astype(jnp.float32), ddq.astype(jnp.float32)

    def nominal_torque_sequence_from_state(
        self, state: jax.Array, horizon: int, dt: float
    ) -> jax.Array:
        """Receding inverse-dynamics seed from the current arm state to PREGRASP."""
        return self.nominal_torque_sequence_to_goal(state, self.goal_q, horizon, dt)

    def nominal_torque_sequence_to_goal(
        self,
        state: jax.Array,
        goal_q: jax.Array,
        horizon: int,
        dt: float,
    ) -> jax.Array:
        """Receding inverse-dynamics seed from the current arm state to a goal pose."""
        state = jnp.asarray(state, dtype=jnp.float32)
        q, v, ddq = self._cubic_joint_trajectory(
            state[: self.nq],
            state[self.nq : self.nq + self.nv],
            jnp.asarray(goal_q, dtype=jnp.float32),
            horizon,
            dt,
        )
        q_np = np.asarray(q, dtype=np.float64)
        v_np = np.asarray(v, dtype=np.float64)
        ddq_np = np.asarray(ddq, dtype=np.float64)
        tau = np.stack([
            pin.rnea(self.pin_model, self.pin_data, q_np[i], v_np[i], ddq_np[i])
            for i in range(horizon)
        ])
        tau = jnp.asarray(tau, dtype=jnp.float32)
        return jnp.clip(tau, -self.torque_limits, self.torque_limits).astype(jnp.float32)

    def nominal_torque_sequence(self, horizon: int, dt: float) -> jax.Array:
        """Smooth inverse-dynamics seed from home to the PREGRASP IK pose."""
        state = jnp.concatenate(
            [self.home_q, jnp.zeros(self.nv, dtype=jnp.float32)],
            axis=0,
        )
        return self.nominal_torque_sequence_from_state(state, horizon, dt)

    def dynamics(
        self, state: jax.Array, inputs: jax.Array, params: jax.Array
    ) -> jax.Array:
        del params
        return self._dynamics_jax(state, inputs)

    def ee_position(self, q: jax.Array) -> jax.Array:
        return self._ee_features_jax(q)[:3]

    def ee_x_axis(self, q: jax.Array) -> jax.Array:
        return self._ee_features_jax(q)[3:6]

    def ee_z_axis(self, q: jax.Array) -> jax.Array:
        return self._ee_features_jax(q)[6:9]

    def ee_features(self, q: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        features = self._ee_features_jax(q)
        return features[:3], features[3:6], features[6:9]


class PandaPregraspObjective(BaseObjective):
    """Pregrasp objective shaped after Hydrax's first pick-and-place phase."""

    def __init__(self, planner: PandaPregraspPlanner):
        super().__init__()
        self.planner = planner
        self.nq = planner.nq
        self.nv = planner.nv

    def reference_vector(self) -> jax.Array:
        return self.planner.reference_vec

    def _goal_pos(self, reference: jax.Array) -> jax.Array:
        return reference[:3]

    def _goal_q(self, reference: jax.Array) -> jax.Array:
        return reference[3 : 3 + self.nq]

    def _goal_x_axis(self, reference: jax.Array) -> jax.Array:
        start = 3 + self.nq
        return reference[start : start + 3]

    def _goal_z_axis(self, reference: jax.Array) -> jax.Array:
        start = 6 + self.nq
        return reference[start : start + 3]

    def _goal_tau(self, reference: jax.Array) -> jax.Array:
        start = 9 + self.nq
        return reference[start : start + self.nv]

    def _axis_alignment_cost(
        self, axis: jax.Array, target_axis: jax.Array
    ) -> jax.Array:
        return 1.0 - jnp.clip(jnp.dot(axis, target_axis), -1.0, 1.0)

    def _smooth_norm(self, vec: jax.Array) -> jax.Array:
        return jnp.sqrt(jnp.sum(jnp.square(vec)) + 1e-8)

    def running_cost(
        self, state: jax.Array, inputs: jax.Array, reference: jax.Array
    ) -> jax.Array:
        q = state[: self.nq]
        v = state[self.nq :]

        goal_pos = self._goal_pos(reference)
        goal_q = self._goal_q(reference)
        goal_x = self._goal_x_axis(reference)
        goal_z = self._goal_z_axis(reference)
        goal_tau = self._goal_tau(reference)

        ee_pos, ee_x, ee_z = self.planner.ee_features(q)

        pos_err = goal_pos - ee_pos
        xy_err = self._smooth_norm(pos_err[:2])
        z_err = jnp.abs(pos_err[2])
        orientation_cost = self._axis_alignment_cost(
            ee_z, goal_z
        ) + 0.5 * self._axis_alignment_cost(ee_x, goal_x)
        posture_cost = jnp.sum(jnp.square(q - goal_q))
        torque_scale = jnp.maximum(self.planner.torque_limits, 1.0)
        control_cost = jnp.sum(jnp.square((inputs - goal_tau) / torque_scale))
        velocity_cost = jnp.sum(jnp.square(v))

        return jnp.asarray(
            120.0 * xy_err
            + 90.0 * z_err
            + 70.0 * orientation_cost
            + 12.0 * posture_cost
            + 0.25 * control_cost
            + 0.1 * velocity_cost,
            dtype=jnp.float32,
        )

    def final_cost(self, state: jax.Array, reference: jax.Array) -> jax.Array:
        q = state[: self.nq]
        v = state[self.nq :]

        goal_pos = self._goal_pos(reference)
        goal_q = self._goal_q(reference)
        goal_x = self._goal_x_axis(reference)
        goal_z = self._goal_z_axis(reference)

        ee_pos, ee_x, ee_z = self.planner.ee_features(q)

        orientation_cost = self._axis_alignment_cost(
            ee_z, goal_z
        ) + 0.5 * self._axis_alignment_cost(ee_x, goal_x)

        return jnp.asarray(
            1500.0 * jnp.sum(jnp.square(goal_pos - ee_pos))
            + 220.0 * orientation_cost
            + 90.0 * jnp.sum(jnp.square(q - goal_q))
            + 15.0 * jnp.sum(jnp.square(v)),
            dtype=jnp.float32,
        )


def make_panda_pregrasp_config(
    planner: PandaPregraspPlanner,
    visualize: bool = True,
    gains: bool = True,
) -> Config:
    robot_config = RobotConfig()
    robot_config.robot_scene_path = planner.scene_path
    robot_config.mjx_kinematic = False
    robot_config.nq = planner.nq
    robot_config.nv = planner.nv
    robot_config.nu = planner.nu
    robot_config.input_min = -planner.torque_limits
    robot_config.input_max = planner.torque_limits
    robot_config.q_init = planner.home_q

    config = Config(robot_config)
    config.general.visualize = visualize
    config.general.verbose = False
    config.general.integrator_type = "si_euler"

    config.sim.dt = 0.02
    config.sim_iterations = 400

    config.MPC.dt = 0.02
    config.MPC.lambda_mpc = 0.05
    config.MPC.std_dev_mppi = 0.05 * planner.torque_limits
    config.MPC.smoothing = "Spline"
    config.MPC.gains = gains
    if gains:
        # Full-state finite-difference gains need a shorter rollout budget to keep
        # the end-to-end controller near 50 Hz. The no-gain behavior keeps the
        # larger planning budget used for behavior checks.
        config.MPC.horizon = 8
        config.MPC.num_parallel_computations = 14
        config.MPC.num_control_points = 4
    else:
        config.MPC.horizon = 16
        config.MPC.num_parallel_computations = 32
        config.MPC.num_control_points = 4
    config.MPC.initial_guess = planner.nominal_torque_sequence(
        config.MPC.horizon,
        config.MPC.dt,
    )
    config.MPC.gain_method = "finite_difference"
    config.MPC.gain_fd_scheme = "forward"
    config.MPC.gain_fd_epsilon = 1e-3

    config.solver_dynamics = DynamicsModel.CUSTOM
    config.sim_dynamics = DynamicsModel.CUSTOM

    return config
