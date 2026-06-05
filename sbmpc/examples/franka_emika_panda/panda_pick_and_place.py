from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.settings import Config, DynamicsModel, RobotConfig
from sbmpc.costs import FactoryObjective
from sbmpc.ocp import build_cost_model, load_ocp_config

from .panda_pregrasp import PandaPregraspPlanner


class Phase(IntEnum):
    """Minimal Panda pick-and-place sequence."""

    PREGRASP = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    OPEN = 6
    RETREAT = 7
    DONE = 8


@dataclass(frozen=True)
class PandaPickAndPlaceReference:
    goal_pos: jax.Array
    goal_q: jax.Array
    goal_x_axis: jax.Array
    goal_z_axis: jax.Array
    goal_tau: jax.Array
    weights: jax.Array

    def as_vector(self) -> jax.Array:
        return jnp.concatenate(
            [
                self.goal_pos,
                self.goal_q,
                self.goal_x_axis,
                self.goal_z_axis,
                self.goal_tau,
                self.weights,
            ]
        )


class PandaPickAndPlacePlanner(PandaPregraspPlanner):
    """Phase-aware torque planner for scripted Panda pick-and-place."""

    PREGRASP_CLEARANCE = 0.05
    CARRY_CLEARANCE = 0.12
    PLACE_CLEARANCE = 0.005
    RETREAT_CLEARANCE = 0.05

    PREGRASP_POS_TOL = 0.02
    DESCEND_POS_TOL = 0.01
    CARRY_OBJ_TOL = 0.025
    PLACE_OBJ_TOL = 0.02
    RETREAT_POS_TOL = 0.03
    SUCCESS_ORI_TOL = 0.08

    PREGRASP_HOLD_STEPS = 4
    DESCEND_HOLD_STEPS = 1
    CLOSE_HOLD_STEPS = 4
    LIFT_HOLD_STEPS = 2
    TRANSPORT_HOLD_STEPS = 2
    PLACE_HOLD_STEPS = 2
    OPEN_HOLD_STEPS = 2
    RETREAT_HOLD_STEPS = 2

    CLOSED_FINGER_WIDTH = 0.022
    OPEN_FINGER_WIDTH = 0.035
    GRIPPER_OPEN = 0.04
    GRIPPER_CLOSE = 0.0

    def __init__(self, scene_path: str | Path | None = None) -> None:
        kwargs = {} if scene_path is None else {"scene_path": scene_path}
        super().__init__(**kwargs)

        self.initial_object_pos = jnp.asarray(self.object_pos, dtype=jnp.float32)
        self.place_height = float(self.initial_object_pos[2])
        self.default_target_pos = jnp.asarray(
            self.target_pos.at[2].set(self.place_height), dtype=jnp.float32
        )
        self.carry_offset = jnp.array([0.0, 0.0, self.CARRY_CLEARANCE], dtype=jnp.float32)
        self.place_offset = jnp.array([0.0, 0.0, self.PLACE_CLEARANCE], dtype=jnp.float32)
        self.retreat_offset = jnp.array(
            [0.0, 0.0, self.object_half_height + self.RETREAT_CLEARANCE],
            dtype=jnp.float32,
        )

        self.phase = Phase.PREGRASP
        self._hold_count = 0
        self._build_phase_tables()
        self.set_phase(Phase.PREGRASP)

    def _build_phase_tables(self) -> None:
        self.phase_goal_pos_map = {
            Phase.PREGRASP: self.initial_object_pos + self.pregrasp_offset,
            Phase.DESCEND: self.initial_object_pos,
            Phase.CLOSE: self.initial_object_pos,
            Phase.LIFT: self.initial_object_pos + self.carry_offset,
            Phase.TRANSPORT: self.default_target_pos + self.carry_offset,
            Phase.PLACE: self.default_target_pos + self.place_offset,
            Phase.OPEN: self.default_target_pos + self.place_offset,
            Phase.RETREAT: self.default_target_pos + self.retreat_offset,
            Phase.DONE: self.default_target_pos + self.retreat_offset,
        }
        self.phase_goal_q_map = {
            phase: jnp.asarray(
                self.solve_ik(np.asarray(goal), np.asarray(self.goal_rotation)),
                dtype=jnp.float32,
            )
            for phase, goal in self.phase_goal_pos_map.items()
        }
        self.phase_goal_tau_map = {
            phase: self.gravity_torques(goal_q)
            for phase, goal_q in self.phase_goal_q_map.items()
        }
        self.phase_next_map = {
            Phase.PREGRASP: Phase.DESCEND,
            Phase.DESCEND: Phase.CLOSE,
            Phase.CLOSE: Phase.LIFT,
            Phase.LIFT: Phase.TRANSPORT,
            Phase.TRANSPORT: Phase.PLACE,
            Phase.PLACE: Phase.OPEN,
            Phase.OPEN: Phase.RETREAT,
            Phase.RETREAT: Phase.DONE,
            Phase.DONE: Phase.DONE,
        }
        self.phase_hold_steps_map = {
            Phase.PREGRASP: self.PREGRASP_HOLD_STEPS,
            Phase.DESCEND: self.DESCEND_HOLD_STEPS,
            Phase.CLOSE: self.CLOSE_HOLD_STEPS,
            Phase.LIFT: self.LIFT_HOLD_STEPS,
            Phase.TRANSPORT: self.TRANSPORT_HOLD_STEPS,
            Phase.PLACE: self.PLACE_HOLD_STEPS,
            Phase.OPEN: self.OPEN_HOLD_STEPS,
            Phase.RETREAT: self.RETREAT_HOLD_STEPS,
            Phase.DONE: 0,
        }
        # weights: ee, orientation, posture, control, velocity, final_position
        self.phase_weights_map = {
            Phase.PREGRASP: jnp.array([1.0, 1.0, 35.0, 0.20, 0.05, 1000.0], dtype=jnp.float32),
            Phase.DESCEND: jnp.array([1.2, 1.0, 35.0, 0.20, 0.05, 1000.0], dtype=jnp.float32),
            Phase.CLOSE: jnp.array([1.0, 1.0, 35.0, 0.15, 0.05, 900.0], dtype=jnp.float32),
            Phase.LIFT: jnp.array([1.2, 1.0, 20.0, 0.12, 0.05, 1200.0], dtype=jnp.float32),
            Phase.TRANSPORT: jnp.array([1.1, 1.0, 20.0, 0.12, 0.05, 1200.0], dtype=jnp.float32),
            Phase.PLACE: jnp.array([1.3, 1.0, 20.0, 0.12, 0.05, 1200.0], dtype=jnp.float32),
            Phase.OPEN: jnp.array([0.8, 1.0, 15.0, 0.10, 0.05, 800.0], dtype=jnp.float32),
            Phase.RETREAT: jnp.array([1.0, 1.0, 15.0, 0.10, 0.05, 1000.0], dtype=jnp.float32),
            Phase.DONE: jnp.array([0.0, 0.0, 10.0, 0.05, 0.05, 0.0], dtype=jnp.float32),
        }

    def reset_phase(self) -> None:
        self.phase = Phase.PREGRASP
        self._hold_count = 0
        self.set_phase(self.phase)

    def goal_position_for_phase(
        self,
        phase: Phase | None = None,
        object_pos: jax.Array | None = None,
        target_pos: jax.Array | None = None,
    ) -> jax.Array:
        phase = self.phase if phase is None else Phase(phase)
        if object_pos is None and target_pos is None:
            return self.phase_goal_pos_map[phase]

        object_pos = (
            self.initial_object_pos
            if object_pos is None
            else jnp.asarray(object_pos, dtype=jnp.float32)
        )
        target_pos = (
            self.default_target_pos
            if target_pos is None
            else jnp.asarray(target_pos, dtype=jnp.float32)
        )

        if phase == Phase.PREGRASP:
            return object_pos + self.pregrasp_offset
        if phase in (Phase.DESCEND, Phase.CLOSE):
            return object_pos
        if phase == Phase.LIFT:
            return object_pos + self.carry_offset
        if phase == Phase.TRANSPORT:
            return target_pos + self.carry_offset
        if phase in (Phase.PLACE, Phase.OPEN):
            return target_pos + self.place_offset
        return target_pos + self.retreat_offset

    def set_phase(
        self,
        phase: Phase,
        object_pos: jax.Array | None = None,
        target_pos: jax.Array | None = None,
    ) -> None:
        self.phase = Phase(phase)
        self.reference = self.reference_for_phase(
            self.phase,
            object_pos=object_pos,
            target_pos=target_pos,
        )
        self.goal_pos = self.reference.goal_pos
        self.goal_q = self.reference.goal_q
        self.goal_tau = self.reference.goal_tau
        self.reference_vec = self.reference.as_vector()

    def reference_for_phase(
        self,
        phase: Phase | None = None,
        object_pos: jax.Array | None = None,
        target_pos: jax.Array | None = None,
    ) -> PandaPickAndPlaceReference:
        phase = self.phase if phase is None else Phase(phase)
        if object_pos is None and target_pos is None:
            goal_pos = self.phase_goal_pos_map[phase]
            goal_q = self.phase_goal_q_map[phase]
            goal_tau = self.phase_goal_tau_map[phase]
        else:
            goal_pos = self.goal_position_for_phase(
                phase,
                object_pos=object_pos,
                target_pos=target_pos,
            )
            goal_q = jnp.asarray(
                self.solve_ik(np.asarray(goal_pos), np.asarray(self.goal_rotation)),
                dtype=jnp.float32,
            )
            goal_tau = self.gravity_torques(goal_q)
        return PandaPickAndPlaceReference(
            goal_pos=goal_pos,
            goal_q=goal_q,
            goal_x_axis=jnp.asarray(self.goal_rotation[:, 0], dtype=jnp.float32),
            goal_z_axis=jnp.asarray(self.goal_rotation[:, 2], dtype=jnp.float32),
            goal_tau=goal_tau,
            weights=self.phase_weights_map[phase],
        )

    def reference_vector(
        self,
        phase: Phase | None = None,
        object_pos: jax.Array | None = None,
        target_pos: jax.Array | None = None,
    ) -> jax.Array:
        return self.reference_for_phase(
            phase,
            object_pos=object_pos,
            target_pos=target_pos,
        ).as_vector()

    def gripper_target(self, phase: Phase | None = None) -> float:
        phase = self.phase if phase is None else Phase(phase)
        if phase in (Phase.PREGRASP, Phase.DESCEND, Phase.OPEN, Phase.RETREAT, Phase.DONE):
            return self.GRIPPER_OPEN
        return self.GRIPPER_CLOSE

    def object_goal_position(
        self,
        phase: Phase | None = None,
        object_pos: jax.Array | None = None,
        target_pos: jax.Array | None = None,
    ) -> jax.Array:
        phase = self.phase if phase is None else Phase(phase)
        object_pos = (
            self.initial_object_pos
            if object_pos is None
            else jnp.asarray(object_pos, dtype=jnp.float32)
        )
        target_pos = (
            self.default_target_pos
            if target_pos is None
            else jnp.asarray(target_pos, dtype=jnp.float32)
        )
        if phase in (Phase.PREGRASP, Phase.DESCEND, Phase.CLOSE):
            return object_pos
        if phase == Phase.LIFT:
            return object_pos + self.carry_offset
        if phase == Phase.TRANSPORT:
            return target_pos + self.carry_offset
        if phase == Phase.PLACE:
            return target_pos + self.place_offset
        return target_pos

    def nominal_torque_sequence_from_state(
        self,
        state: jax.Array,
        horizon: int,
        dt: float,
        phase: Phase | None = None,
        object_pos: jax.Array | None = None,
        target_pos: jax.Array | None = None,
    ) -> jax.Array:
        phase = self.phase if phase is None else Phase(phase)
        reference = self.reference_for_phase(
            phase,
            object_pos=object_pos,
            target_pos=target_pos,
        )
        return self.nominal_torque_sequence_to_goal(
            state,
            reference.goal_q,
            horizon,
            dt,
        )

    def pose_error(self, state: jax.Array, phase: Phase | None = None) -> tuple[float, float]:
        phase = self.phase if phase is None else Phase(phase)
        q = jnp.asarray(state[: self.nq], dtype=jnp.float32)
        ee_pos, ee_x, ee_z = self.ee_features(q)
        pos_err = float(jnp.linalg.norm(ee_pos - self.phase_goal_pos_map[phase]))
        z_cost = 1.0 - float(jnp.clip(jnp.dot(ee_z, jnp.asarray(self.goal_rotation[:, 2])), -1.0, 1.0))
        x_cost = 1.0 - float(jnp.clip(jnp.dot(ee_x, jnp.asarray(self.goal_rotation[:, 0])), -1.0, 1.0))
        return pos_err, z_cost + 0.5 * x_cost

    def object_error(self, object_pos: jax.Array, phase: Phase | None = None) -> float:
        phase = self.phase if phase is None else Phase(phase)
        return float(jnp.linalg.norm(jnp.asarray(object_pos) - self.object_goal_position(phase)))

    def phase_complete(self, state: jax.Array, object_pos: jax.Array, finger_width: float) -> bool:
        phase = self.phase
        if phase == Phase.DONE:
            return True

        ee_err, ori_err = self.pose_error(state, phase)
        obj_err = self.object_error(object_pos, phase)
        finger_open = finger_width >= self.OPEN_FINGER_WIDTH

        if phase == Phase.PREGRASP:
            return ee_err <= self.PREGRASP_POS_TOL and ori_err <= self.SUCCESS_ORI_TOL
        if phase == Phase.DESCEND:
            return ee_err <= self.DESCEND_POS_TOL and ori_err <= self.SUCCESS_ORI_TOL
        if phase == Phase.CLOSE:
            return finger_width <= self.CLOSED_FINGER_WIDTH and ee_err <= 2.0 * self.DESCEND_POS_TOL
        if phase in (Phase.LIFT, Phase.TRANSPORT):
            return obj_err <= self.CARRY_OBJ_TOL and ori_err <= self.SUCCESS_ORI_TOL
        if phase == Phase.PLACE:
            return obj_err <= self.PLACE_OBJ_TOL and ee_err <= 2.0 * self.DESCEND_POS_TOL and ori_err <= self.SUCCESS_ORI_TOL
        if phase == Phase.OPEN:
            return finger_open and obj_err <= 1.5 * self.PLACE_OBJ_TOL
        return ee_err <= self.RETREAT_POS_TOL and ori_err <= self.SUCCESS_ORI_TOL and finger_open

    def update_phase(self, state: jax.Array, object_pos: jax.Array, finger_width: float) -> bool:
        old_phase = self.phase
        if self.phase_complete(state, object_pos, finger_width):
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self._hold_count >= self.phase_hold_steps_map[self.phase]:
            self.phase = self.phase_next_map[self.phase]
            if self.phase != old_phase:
                self._hold_count = 0
                self.set_phase(self.phase)

        return self.phase != old_phase

    def sequence_complete(self) -> bool:
        return self.phase == Phase.DONE


class PandaPickAndPlaceObjective(FactoryObjective):
    """Phase-conditioned arm objective assembled from the ``pick_and_place`` OCP.

    Per-phase weights are carried in the reference (6-vector); defaults reproduce
    the original hand-rolled weights (see ``sbmpc/ocp_configs/pick_and_place.yaml``).
    """

    def __init__(self, planner: PandaPickAndPlacePlanner, ocp_config=None):
        self.planner = planner
        self.nq = planner.nq
        self.nv = planner.nv
        if ocp_config is None:
            ocp_config = load_ocp_config("pick_and_place")
        self.ocp_config = ocp_config
        super().__init__(build_cost_model(ocp_config, planner))

    def reference_vector(self, phase: Phase | None = None) -> jax.Array:
        return self.planner.reference_vector(phase)


def make_panda_pick_and_place_config(
    planner: PandaPickAndPlacePlanner,
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
    config.sim_iterations = 700

    config.MPC.dt = 0.02
    config.MPC.lambda_mpc = 0.05
    config.MPC.std_dev_mppi = 0.05 * planner.torque_limits
    config.MPC.smoothing = "Spline"
    config.MPC.gains = gains
    if gains:
        config.MPC.horizon = 8
        config.MPC.num_parallel_computations = 14
        config.MPC.num_control_points = 4
        config.MPC.gain_method = "exact"
        config.MPC.gain_samples_per_cycle = 14
        config.MPC.gain_buffer_size = 14
    else:
        config.MPC.horizon = 16
        config.MPC.num_parallel_computations = 32
        config.MPC.num_control_points = 4
    config.MPC.initial_guess = planner.nominal_torque_sequence_from_state(
        jnp.concatenate([planner.home_q, jnp.zeros(planner.nv, dtype=jnp.float32)]),
        config.MPC.horizon,
        config.MPC.dt,
        Phase.PREGRASP,
    )
    config.solver_dynamics = DynamicsModel.CUSTOM
    config.sim_dynamics = DynamicsModel.CUSTOM
    return config
