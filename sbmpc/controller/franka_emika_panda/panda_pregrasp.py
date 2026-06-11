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
from sbmpc.costs import FactoryObjective
from sbmpc.ocp import build_cost_model, load_ocp_config


ROOT = Path(__file__).resolve().parents[3]
PANDA_SCENE_PATH = ROOT / "models" / "panda_pick_place" / "scene.xml"
PANDA_XML_PATH = ROOT / "models" / "panda_pick_place" / "panda.xml"
PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
PANDA_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
PANDA_TCP_FRAME_NAME = "panda_hand_tcp"

_DESIRED_X = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
_DESIRED_Z = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
_ARM_TORQUE_LIMITS = jnp.array(
    [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=jnp.float32
)
_ARM_VELOCITY_LIMITS = jnp.array(
    [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=jnp.float32
)  # FR3 ("fer") joint velocity limits, used by the validation criteria.
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

        (
            self.home_q_full,
            self.object_pos,
            self.target_pos,
            self.object_half_height,
            self.joint_position_min,
            self.joint_position_max,
        ) = self._load_mujoco_defaults()

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
        self.velocity_limits = _ARM_VELOCITY_LIMITS

        (self._dynamics_jax, self._step_jax, self._ee_features_jax) = (
            self._build_mjx_functions(str(PANDA_XML_PATH))
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

    def _load_mujoco_defaults(
        self,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        float,
        jax.Array,
        jax.Array,
    ]:
        mj_model = mujoco.MjModel.from_xml_path(self.scene_path)
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home_q = jnp.asarray(mj_model.key_qpos[key_id][:9], dtype=jnp.float32)
        object_body_idx = mj_model.body("object").id
        target_body_idx = mj_model.body("target").id
        object_geom_idx = mj_model.geom("object_geom").id
        object_pos = jnp.asarray(mj_model.body_pos[object_body_idx], dtype=jnp.float32)
        target_pos = jnp.asarray(mj_model.body_pos[target_body_idx], dtype=jnp.float32)
        object_half_height = float(mj_model.geom_size[object_geom_idx, 2])
        arm_joint_ids = [mj_model.joint(f"joint{i}").id for i in range(1, 8)]
        joint_ranges = jnp.asarray(mj_model.jnt_range[arm_joint_ids], dtype=jnp.float32)
        return (
            home_q,
            object_pos,
            target_pos,
            object_half_height,
            joint_ranges[:, 0],
            joint_ranges[:, 1],
        )

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

    def _build_mjx_functions(
        self, panda_xml_path: str
    ) -> tuple[callable, callable, callable]:
        mj_model = mujoco.MjModel.from_xml_path(panda_xml_path)
        torque_limits_np = np.asarray(self.torque_limits)
        mj_model.actuator_gainprm[:7, 0] = 1.0
        mj_model.actuator_gainprm[:7, 1:] = 0.0
        mj_model.actuator_biasprm[:7, :] = 0.0
        mj_model.actuator_gainprm[7:, :] = 0.0
        mj_model.actuator_biasprm[7:, :] = 0.0
        mj_model.actuator_ctrlrange[:7, 0] = -torque_limits_np
        mj_model.actuator_ctrlrange[:7, 1] = torque_limits_np
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mjx_model = mjx.put_model(mj_model)
        mjx_data_template = mjx.put_data(mj_model, mujoco.MjData(mj_model))
        gripper_site_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )
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
        def step_fn(state: jax.Array, inputs: jax.Array, dt: jax.Array) -> jax.Array:
            q = state[:nq_arm].astype(jnp.float32)
            v = state[nq_arm:].astype(jnp.float32)
            q_full = jnp.concatenate([q, finger_pad])
            v_full = jnp.concatenate([v, finger_pad])
            ctrl_full = jnp.concatenate([inputs.astype(jnp.float32), ctrl_pad])
            model = mjx_model.replace(opt=mjx_model.opt.replace(timestep=dt))
            data = mjx_data_template.replace(qpos=q_full, qvel=v_full, ctrl=ctrl_full)
            data = mjx.step(model, data)
            return jnp.concatenate([data.qpos[:nq_arm], data.qvel[:nq_arm]]).astype(
                jnp.float32
            )

        @jax.jit
        def ee_features_fn(q: jax.Array) -> jax.Array:
            q_full = jnp.concatenate([q.astype(jnp.float32), finger_pad])
            data = mjx_data_template.replace(qpos=q_full)
            data = mjx.forward(mjx_model, data)
            ee_xmat = data.site_xmat[gripper_site_id].reshape(3, 3)
            return jnp.concatenate(
                [
                    data.site_xpos[gripper_site_id],
                    ee_xmat[:, 0],
                    ee_xmat[:, 2],
                ]
            ).astype(jnp.float32)

        return dynamics_fn, step_fn, ee_features_fn

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

    def dynamics(
        self,
        state: jax.Array,
        inputs: jax.Array,
        params: jax.Array,
        dt: jax.Array | None = None,
    ) -> jax.Array:
        del params
        if dt is None:
            return self._dynamics_jax(state, inputs)
        return self._step_jax(state, inputs, dt)

    def ee_position(self, q: jax.Array) -> jax.Array:
        return self._ee_features_jax(q)[:3]

    def ee_x_axis(self, q: jax.Array) -> jax.Array:
        return self._ee_features_jax(q)[3:6]

    def ee_z_axis(self, q: jax.Array) -> jax.Array:
        return self._ee_features_jax(q)[6:9]

    def ee_features(self, q: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        features = self._ee_features_jax(q)
        return features[:3], features[3:6], features[6:9]


class PandaPregraspObjective(FactoryObjective):
    """Pregrasp objective assembled from the declarative ``pregrasp`` OCP.

    Defaults reproduce the original hand-rolled weights (see
    ``sbmpc/ocp_configs/pregrasp.yaml``). Pass ``ocp_config`` to retune the cost
    without changing code.
    """

    def __init__(self, planner: PandaPregraspPlanner, ocp_config=None):
        self.planner = planner
        self.nq = planner.nq
        self.nv = planner.nv
        if ocp_config is None:
            ocp_config = load_ocp_config("pregrasp")
        self.ocp_config = ocp_config
        super().__init__(build_cost_model(ocp_config, planner))

    def reference_vector(self) -> jax.Array:
        return self.planner.reference_vec


def _initial_guess(planner: PandaPregraspPlanner, mpc) -> jax.Array:
    """Warm start for the MPPI control sequence (shape (horizon, nu)).

    ``zeros`` = apply nothing (the optimizer must find everything, incl. gravity).
    ``gravity`` = hold against gravity at the home pose — a stable starting point,
    NOT a trajectory to the goal; the optimizer still resolves the reach.
    """
    if mpc.initial_guess == "gravity":
        g = jnp.asarray(planner.gravity_torques(planner.home_q), dtype=jnp.float32)
        return jnp.tile(g, (mpc.horizon, 1))
    if mpc.initial_guess == "zeros":
        return jnp.zeros((mpc.horizon, planner.nu), dtype=jnp.float32)
    raise ValueError(
        f"unknown initial_guess '{mpc.initial_guess}' (use 'zeros' or 'gravity')."
    )


def make_panda_pregrasp_config(
    planner: PandaPregraspPlanner,
    visualize: bool = True,
    gains: bool | None = None,
    ocp=None,
) -> Config:
    """Build the sbmpc Config from the OCP yaml (``mpc:`` / ``sim:`` sections).

    All MPPI/solver/sim knobs come from the OCP (default: ``pregrasp.yaml``), so the
    controller is tuned by editing the yaml. ``gains`` overrides the yaml's
    ``mpc.gains`` when given (the ROS/controller API uses this); ``None`` = use yaml.
    """
    if ocp is None:
        ocp = load_ocp_config("pregrasp")
    mpc, sim = ocp.mpc, ocp.sim
    use_gains = mpc.gains if gains is None else gains

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
    config.general.integrator_type = sim.integrator

    config.sim.dt = sim.dt
    config.sim_iterations = sim.iterations

    config.MPC.dt = mpc.dt
    config.MPC.horizon = mpc.horizon
    config.MPC.num_parallel_computations = mpc.num_samples
    config.MPC.num_control_points = mpc.num_control_points
    config.MPC.lambda_mpc = mpc.lambda_mpc
    config.MPC.std_dev_mppi = mpc.std_dev_scale * planner.torque_limits
    config.MPC.smoothing = mpc.smoothing
    config.MPC.gains = use_gains
    if use_gains:
        config.MPC.gain_method = "exact"
        config.MPC.num_gain_samples = mpc.num_gain_samples
    config.MPC.initial_guess = _initial_guess(planner, mpc)

    config.solver_dynamics = DynamicsModel.CUSTOM
    config.sim_dynamics = DynamicsModel.CUSTOM

    return config
