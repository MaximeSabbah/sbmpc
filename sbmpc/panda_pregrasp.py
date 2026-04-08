from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import casadi as cs
import jax
import jax.numpy as jnp
import jaxadi
import mujoco
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from sbmpc.settings import Config, DynamicsModel, RobotConfig
from sbmpc.solvers import BaseObjective


ROOT = Path(__file__).resolve().parents[1]
PANDA_SCENE_PATH = ROOT / "examples" / "franka_emika_panda" / "scene.xml"
PANDA_MJCF_PATH = ROOT / "examples" / "franka_emika_panda" / "panda_nohand.xml"


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
    """Pinocchio + JaxADi model for a 7-DoF Panda pregrasp phase."""

    def __init__(
        self,
        scene_path: str | Path = PANDA_SCENE_PATH,
        mjcf_path: str | Path = PANDA_MJCF_PATH,
        goal_pos: jax.Array | None = None,
    ) -> None:
        self.scene_path = str(scene_path)
        self.mjcf_path = str(mjcf_path)

        self.model = pin.buildModelFromMJCF(self.mjcf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("attachment")

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nv
        self.nx = self.nq + self.nv

        self.home_q, self.torque_limits = self._load_mujoco_defaults()
        self.damping = jnp.asarray(
            np.asarray(self.model.damping).reshape(-1), dtype=jnp.float32
        )

        home_pos, home_rot = self.forward_kinematics(self.home_q)
        self.home_pos = jnp.asarray(home_pos, dtype=jnp.float32)
        self.home_rotation = jnp.asarray(home_rot, dtype=jnp.float32)

        self.goal_pos = (
            jnp.array([0.45, 0.0, 0.35], dtype=jnp.float32)
            if goal_pos is None
            else jnp.asarray(goal_pos, dtype=jnp.float32)
        )
        self.goal_rotation = self.home_rotation
        self.goal_q = jnp.asarray(
            self.solve_ik(
                np.asarray(self.goal_pos), np.asarray(self.goal_rotation)
            ),
            dtype=jnp.float32,
        )
        self.goal_tau = jnp.asarray(
            pin.computeGeneralizedGravity(
                self.model,
                self.data,
                np.asarray(self.goal_q, dtype=np.float64),
            ),
            dtype=jnp.float32,
        )

        self.reference = PandaPregraspReference(
            goal_pos=self.goal_pos,
            goal_q=self.goal_q,
            goal_x_axis=self.goal_rotation[:, 0],
            goal_z_axis=self.goal_rotation[:, 2],
            goal_tau=self.goal_tau,
        )
        self.reference_vec = self.reference.as_vector()

        self._dynamics_jax, self._ee_features_jax = self._build_symbolic_functions()

    def _load_mujoco_defaults(self) -> tuple[jax.Array, jax.Array]:
        mj_model = mujoco.MjModel.from_xml_path(self.scene_path)
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home_q = jnp.asarray(mj_model.key_qpos[key_id], dtype=jnp.float32)
        torque_limits = jnp.asarray(
            mj_model.actuator_forcerange[:, 1], dtype=jnp.float32
        )
        return home_q, torque_limits

    def _build_symbolic_functions(self) -> tuple[callable, callable]:
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()

        q_sym = cs.SX.sym("q", self.nq, 1)
        v_sym = cs.SX.sym("v", self.nv, 1)
        tau_sym = cs.SX.sym("tau", self.nu, 1)

        damping = cs.DM(np.asarray(self.damping).reshape(-1, 1))
        ddq = cpin.aba(cmodel, cdata, q_sym, v_sym, tau_sym - damping * v_sym)
        xdot = cs.vertcat(v_sym, ddq)

        cpin.framesForwardKinematics(cmodel, cdata, q_sym)
        frame = cdata.oMf[self.frame_id]
        ee_features = cs.vertcat(
            frame.translation,
            frame.rotation[:, 0],
            frame.rotation[:, 2],
        )

        dynamics_fn = jaxadi.convert(
            cs.Function("panda_aba_dynamics", [q_sym, v_sym, tau_sym], [xdot])
        )
        ee_features_fn = jaxadi.convert(
            cs.Function("panda_attachment_features", [q_sym], [ee_features])
        )

        def _single_output(fn, *args, size: int) -> jax.Array:
            return jnp.asarray(fn(*args)[0], dtype=jnp.float32).reshape(size)

        return (
            jax.jit(lambda q, v, tau: _single_output(dynamics_fn, q, v, tau, size=self.nx)),
            jax.jit(lambda q: _single_output(ee_features_fn, q, size=9)),
        )

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_np = np.asarray(q, dtype=np.float64)
        pin.framesForwardKinematics(self.model, self.data, q_np)
        frame = self.data.oMf[self.frame_id]
        return frame.translation.copy(), frame.rotation.copy()

    def solve_ik(
        self,
        goal_pos: np.ndarray,
        goal_rot: np.ndarray,
        max_iters: int = 250,
        tol: float = 1e-4,
    ) -> np.ndarray:
        q = np.asarray(self.home_q, dtype=np.float64).copy()
        target = pin.SE3(goal_rot, goal_pos)
        lower = np.asarray(self.model.lowerPositionLimit).reshape(-1)
        upper = np.asarray(self.model.upperPositionLimit).reshape(-1)

        for _ in range(max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.frame_id)
            current = self.data.oMf[self.frame_id]
            delta = current.actInv(target)
            err = pin.log6(delta).vector
            if np.linalg.norm(err) < tol:
                break

            jacobian = pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, pin.LOCAL
            )
            step = -jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(6), err
            )
            q = pin.integrate(self.model, q, 0.4 * step)
            q = np.clip(q, lower, upper)

        return q.astype(np.float32)

    def dynamics(
        self, state: jax.Array, inputs: jax.Array, params: jax.Array
    ) -> jax.Array:
        del params
        q = state[: self.nq]
        v = state[self.nq :]
        return self._dynamics_jax(q, v, inputs)

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

        return (
            120.0 * xy_err
            + 90.0 * z_err
            + 70.0 * orientation_cost
            + 12.0 * posture_cost
            + 0.25 * control_cost
            + 0.1 * velocity_cost
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

        return (
            1500.0 * jnp.sum(jnp.square(goal_pos - ee_pos))
            + 220.0 * orientation_cost
            + 90.0 * jnp.sum(jnp.square(q - goal_q))
            + 15.0 * jnp.sum(jnp.square(v))
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
    config.general.integrator_type = "si_euler"

    config.sim.dt = 0.02
    config.sim_iterations = 400

    config.MPC.dt = 0.02
    config.MPC.horizon = 25
    config.MPC.num_parallel_computations = 128
    config.MPC.lambda_mpc = 0.05
    config.MPC.std_dev_mppi = 0.05 * planner.torque_limits
    config.MPC.initial_guess = planner.goal_tau
    config.MPC.smoothing = "Spline"
    config.MPC.num_control_points = 6
    config.MPC.gains = gains

    config.solver_dynamics = DynamicsModel.CUSTOM
    config.sim_dynamics = DynamicsModel.CUSTOM

    return config
