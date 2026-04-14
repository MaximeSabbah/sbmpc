from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.panda_pick_and_place import (
    Phase,
    PandaPickAndPlaceObjective,
    PandaPickAndPlacePlanner,
    PandaPickAndPlaceReference,
    make_panda_pick_and_place_config,
)
from sbmpc.settings import Config
from sbmpc.simulation import build_model_and_solver


@dataclass(frozen=True)
class TaskPose:
    """Task-space pose container. Only the position is used by the planner today."""

    position: np.ndarray
    quaternion: np.ndarray | None = None


@dataclass(frozen=True)
class GripperCommand:
    action: str
    width: float


@dataclass(frozen=True)
class PlannerDiagnostics:
    planning_time_ms: float
    running_cost: float
    gain_norm: float
    torque_norm: float
    position_error: float
    orientation_error: float
    object_error: float | None
    goal_position: np.ndarray


@dataclass(frozen=True)
class PlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray
    phase: Phase
    next_phase: Phase
    gripper_command: GripperCommand
    diagnostics: PlannerDiagnostics


class PandaPickAndPlaceController:
    """Stable, non-ROS adapter around the Panda pick-and-place planner."""

    def __init__(
        self,
        planner: PandaPickAndPlacePlanner | None = None,
        config: Config | None = None,
        *,
        gains: bool = True,
        num_steps: int = 1,
        visualize: bool = False,
    ) -> None:
        self.planner = PandaPickAndPlacePlanner() if planner is None else planner
        self.objective = PandaPickAndPlaceObjective(self.planner)
        self.config = (
            make_panda_pick_and_place_config(
                self.planner,
                visualize=visualize,
                gains=gains,
            )
            if config is None
            else config
        )
        self.model, self.controller = build_model_and_solver(
            self.config,
            self.objective,
            custom_dynamics_fn=self.planner.dynamics,
        )
        self._default_num_steps = self._validate_num_steps(num_steps)
        self._solution_initialized = False
        self._last_reference_signature: tuple[object, ...] | None = None

    def warmup(
        self,
        phase: Phase = Phase.PREGRASP,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
        num_steps: int | None = None,
    ) -> PlannerOutput:
        return self.step(
            self.planner.home_q,
            jnp.zeros(self.planner.nv, dtype=jnp.float32),
            phase,
            object_pose=object_pose,
            target_pose=target_pose,
            num_steps=num_steps,
            reset_guess=True,
        )

    def reference_for_phase(
        self,
        phase: Phase,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
    ) -> PandaPickAndPlaceReference:
        object_pos = self._position_from_pose(object_pose, "object_pose")
        target_pos = self._position_from_pose(target_pose, "target_pose")
        return self.planner.reference_for_phase(
            phase,
            object_pos=object_pos,
            target_pos=target_pos,
        )

    def step(
        self,
        q: np.ndarray,
        v: np.ndarray,
        phase: Phase,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
        *,
        num_steps: int | None = None,
        reset_guess: bool = False,
    ) -> PlannerOutput:
        phase = Phase(phase)
        q = self._joint_vector(q, self.planner.nq, "q")
        v = self._joint_vector(v, self.planner.nv, "v")
        object_pos = self._position_from_pose(object_pose, "object_pose")
        target_pos = self._position_from_pose(target_pose, "target_pose")

        self.planner.set_phase(
            phase,
            object_pos=object_pos,
            target_pos=target_pos,
        )
        reference = self.planner.reference
        state = jnp.concatenate([q, v], axis=0)
        reference_signature = self._reference_signature(
            phase,
            object_pos=object_pos,
            target_pos=target_pos,
        )
        effective_num_steps = self._default_num_steps if num_steps is None else num_steps
        effective_num_steps = self._validate_num_steps(effective_num_steps)
        if (
            reset_guess
            or not self._solution_initialized
            or reference_signature != self._last_reference_signature
        ):
            self._seed_nominal_solution(state, reference.goal_q)

        start_time = time.time_ns()
        input_sequence = self.controller.command(
            state,
            self.planner.reference_vec,
            shift_guess=True,
            num_steps=effective_num_steps,
        )
        input_sequence = jax.block_until_ready(input_sequence)
        self._solution_initialized = True
        self._last_reference_signature = reference_signature
        gains = np.asarray(jax.block_until_ready(self.controller.gains), dtype=np.float32)
        tau_ff = np.asarray(input_sequence[0], dtype=np.float32)
        planning_time_ms = 1e-6 * (time.time_ns() - start_time)

        ee_pos, ee_x, ee_z = self.planner.ee_features(q)
        position_error = float(jnp.linalg.norm(ee_pos - reference.goal_pos))
        orientation_error = float(
            (1.0 - jnp.clip(jnp.dot(ee_z, reference.goal_z_axis), -1.0, 1.0))
            + 0.5 * (1.0 - jnp.clip(jnp.dot(ee_x, reference.goal_x_axis), -1.0, 1.0))
        )
        object_error = None
        if object_pos is not None:
            object_goal = self.planner.object_goal_position(
                phase,
                object_pos=object_pos,
                target_pos=target_pos,
            )
            object_error = float(jnp.linalg.norm(object_pos - object_goal))

        running_cost = float(
            jax.block_until_ready(
                self.objective.running_cost(
                    state,
                    jnp.asarray(tau_ff, dtype=jnp.float32),
                    self.planner.reference_vec,
                )
            )
        )
        gripper_width = float(self.planner.gripper_target(phase))
        gripper_command = GripperCommand(
            action="open" if gripper_width >= self.planner.GRIPPER_OPEN else "close",
            width=gripper_width,
        )
        diagnostics = PlannerDiagnostics(
            planning_time_ms=planning_time_ms,
            running_cost=running_cost,
            gain_norm=float(np.linalg.norm(gains)),
            torque_norm=float(np.linalg.norm(tau_ff)),
            position_error=position_error,
            orientation_error=orientation_error,
            object_error=object_error,
            goal_position=np.asarray(reference.goal_pos, dtype=np.float32),
        )
        return PlannerOutput(
            tau_ff=tau_ff,
            K=gains,
            phase=phase,
            next_phase=self.planner.phase_next_map[phase],
            gripper_command=gripper_command,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _joint_vector(values: np.ndarray, size: int, name: str) -> jax.Array:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
        return jnp.asarray(array, dtype=jnp.float32)

    @staticmethod
    def _position_from_pose(
        pose: TaskPose | np.ndarray | None,
        name: str,
    ) -> jax.Array | None:
        if pose is None:
            return None
        position = pose.position if isinstance(pose, TaskPose) else pose
        array = np.asarray(position, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"{name} position must have shape (3,), got {array.shape}.")
        return jnp.asarray(array, dtype=jnp.float32)

    def _seed_nominal_solution(self, state: jax.Array, goal_q: jax.Array) -> None:
        self.controller.sampler.optimal_samples = self.planner.nominal_torque_sequence_to_goal(
            state,
            goal_q,
            self.config.MPC.horizon,
            self.config.MPC.dt,
        )
        self.controller.gains_obj.cur_gains = jnp.zeros(
            (self.planner.nu, self.planner.nx),
            dtype=getattr(self.config.general, "dtype", jnp.float32),
        )

    @staticmethod
    def _validate_num_steps(value: int) -> int:
        num_steps = int(value)
        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")
        return num_steps

    @staticmethod
    def _reference_signature(
        phase: Phase,
        *,
        object_pos: jax.Array | None,
        target_pos: jax.Array | None,
    ) -> tuple[object, ...]:
        return (
            phase.name,
            PandaPickAndPlaceController._vector_signature(object_pos),
            PandaPickAndPlaceController._vector_signature(target_pos),
        )

    @staticmethod
    def _vector_signature(value: jax.Array | None) -> tuple[float, ...] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        return tuple(float(entry) for entry in array)
