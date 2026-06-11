from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.settings import Config
from sbmpc.simulation import build_model_and_solver

from .panda_pick_and_place import (
    Phase,
    PandaPickAndPlaceObjective,
    PandaPickAndPlacePlanner,
    PandaPickAndPlaceReference,
    make_panda_pick_and_place_config,
)
from .panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.ocp import load_ocp_config


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
    """Per-step planner report.

    ``planning_time_ms`` is the foreground planning latency: the blocked
    ``controller.command`` call plus fetching the feedback gains.
    ``planner_prepare_time_ms`` (input conversion/seeding) and
    ``planner_command_time_ms`` (the blocked command call alone) are its
    main components.
    """

    planning_time_ms: float
    running_cost: float | None
    gain_norm: float
    torque_norm: float
    position_error: float | None
    orientation_error: float | None
    object_error: float | None
    goal_position: np.ndarray
    gain_mode: str | None = None
    planner_prepare_time_ms: float | None = None
    planner_command_time_ms: float | None = None


@dataclass(frozen=True)
class PlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray
    phase: object
    next_phase: object
    gripper_command: GripperCommand
    diagnostics: PlannerDiagnostics


GAIN_MODE_FEEDFORWARD = "feedforward"
GAIN_MODE_EXACT_FEEDBACK = "exact_feedback"
SUPPORTED_GAIN_MODES = {GAIN_MODE_FEEDFORWARD, GAIN_MODE_EXACT_FEEDBACK}


def _resolve_gain_mode(config: Config, gain_mode: str | None) -> str:
    if gain_mode is None:
        return GAIN_MODE_EXACT_FEEDBACK if config.MPC.gains else GAIN_MODE_FEEDFORWARD
    mode = gain_mode.strip().lower()
    if mode not in SUPPORTED_GAIN_MODES:
        valid = ", ".join(sorted(SUPPORTED_GAIN_MODES))
        raise ValueError(f"unsupported gain_mode {gain_mode!r}. Choose from: {valid}.")
    return mode


def _apply_gain_mode_to_config(config: Config, gain_mode: str) -> None:
    if gain_mode == GAIN_MODE_FEEDFORWARD:
        config.MPC.gains = False
        return
    if gain_mode == GAIN_MODE_EXACT_FEEDBACK:
        config.MPC.gains = True
        config.MPC.gain_method = "exact"
        gain_samples = config.MPC.num_gain_samples
        if gain_samples is not None and gain_samples > config.MPC.num_parallel_computations:
            raise ValueError(
                "num_gain_samples cannot exceed num_parallel_computations."
            )
        return
    raise ValueError(f"unsupported gain_mode: {gain_mode!r}.")


def _current_gains_numpy(controller) -> np.ndarray:
    return np.asarray(jax.block_until_ready(controller.gains), dtype=np.float32)


class PandaPickAndPlaceController:
    """Synchronous non-ROS adapter around the Panda pick-and-place planner."""

    def __init__(
        self,
        planner: PandaPickAndPlacePlanner | None = None,
        config: Config | None = None,
        *,
        gains: bool = True,
        num_steps: int = 1,
        visualize: bool = False,
        gain_mode: str | None = None,
        compute_running_cost: bool = True,
        compute_task_diagnostics: bool = True,
    ) -> None:
        self.planner = PandaPickAndPlacePlanner() if planner is None else planner
        self.objective = PandaPickAndPlaceObjective(self.planner)
        self.config = (
            make_panda_pick_and_place_config(
                self.planner, visualize=visualize, gains=gains
            )
            if config is None
            else config
        )
        self.gain_mode = _resolve_gain_mode(self.config, gain_mode)
        _apply_gain_mode_to_config(self.config, self.gain_mode)
        self.model, self.controller = build_model_and_solver(
            self.config, self.objective, custom_dynamics_fn=self.planner.dynamics
        )
        self._default_num_steps = self._validate_num_steps(num_steps)
        self._solution_initialized = False
        self._last_reference_signature: tuple[object, ...] | None = None
        self._started = False
        self._compute_running_cost = compute_running_cost
        self._compute_task_diagnostics = compute_task_diagnostics

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self.controller.close()
        self._started = False

    def reset_runtime_state_after_warmup(self) -> None:
        self._started = False
        self._solution_initialized = False
        self._last_reference_signature = None

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
            reset_guess=not self._solution_initialized,
        )

    def reference_for_phase(
        self,
        phase: Phase,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
    ) -> PandaPickAndPlaceReference:
        return self.planner.reference_for_phase(
            phase,
            object_pos=self._position_from_pose(object_pose, "object_pose"),
            target_pos=self._position_from_pose(target_pose, "target_pose"),
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
        prepare_start = time.perf_counter()
        phase = Phase(phase)
        q = self._joint_vector(q, self.planner.nq, "q")
        v = self._joint_vector(v, self.planner.nv, "v")
        object_pos = self._position_from_pose(object_pose, "object_pose")
        target_pos = self._position_from_pose(target_pose, "target_pose")
        self.planner.set_phase(phase, object_pos=object_pos, target_pos=target_pos)
        reference = self.planner.reference
        state = jnp.concatenate([q, v], axis=0)
        reference_signature = self._reference_signature(
            phase, object_pos=object_pos, target_pos=target_pos
        )
        effective_num_steps = self._validate_num_steps(
            self._default_num_steps if num_steps is None else num_steps
        )
        if (
            reset_guess
            or not self._solution_initialized
            or reference_signature != self._last_reference_signature
        ):
            self._seed_nominal_solution(state, reference.goal_q)
        planner_prepare_time_ms = 1000.0 * (time.perf_counter() - prepare_start)

        command_start = time.perf_counter()
        input_sequence = self.controller.command(
            state,
            self.planner.reference_vec,
            shift_guess=True,
            num_steps=effective_num_steps,
        )
        input_sequence = jax.block_until_ready(input_sequence)
        planner_command_time_ms = 1000.0 * (time.perf_counter() - command_start)
        self._solution_initialized = True
        self._last_reference_signature = reference_signature

        tau_ff = np.asarray(input_sequence[0], dtype=np.float32)
        gains_array = _current_gains_numpy(self.controller)
        planning_time_ms = 1000.0 * (time.perf_counter() - command_start)

        position_error = None
        orientation_error = None
        object_error = None
        if self._compute_task_diagnostics:
            ee_pos, ee_x, ee_z = self.planner.ee_features(q)
            position_error = float(jnp.linalg.norm(ee_pos - reference.goal_pos))
            orientation_error = float(
                1.0
                - jnp.clip(jnp.dot(ee_z, reference.goal_z_axis), -1.0, 1.0)
                + 0.5
                * (
                    1.0
                    - jnp.clip(jnp.dot(ee_x, reference.goal_x_axis), -1.0, 1.0)
                )
            )
            if object_pos is not None:
                object_goal = self.planner.object_goal_position(
                    phase, object_pos=object_pos, target_pos=target_pos
                )
                object_error = float(jnp.linalg.norm(object_pos - object_goal))
        running_cost = None
        if self._compute_running_cost:
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
        return PlannerOutput(
            tau_ff=tau_ff,
            K=gains_array,
            phase=phase,
            next_phase=self.planner.phase_next_map[phase],
            gripper_command=GripperCommand(
                action=(
                    "open"
                    if gripper_width >= self.planner.GRIPPER_OPEN
                    else "close"
                ),
                width=gripper_width,
            ),
            diagnostics=PlannerDiagnostics(
                planning_time_ms=planning_time_ms,
                running_cost=running_cost,
                gain_norm=float(np.linalg.norm(gains_array)),
                torque_norm=float(np.linalg.norm(tau_ff)),
                position_error=position_error,
                orientation_error=orientation_error,
                object_error=object_error,
                goal_position=np.asarray(reference.goal_pos, dtype=np.float32),
                gain_mode=self.gain_mode,
                planner_prepare_time_ms=planner_prepare_time_ms,
                planner_command_time_ms=planner_command_time_ms,
            ),
        )

    @staticmethod
    def _joint_vector(values: np.ndarray, size: int, name: str) -> jax.Array:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
        return jnp.asarray(array, dtype=jnp.float32)


    @staticmethod
    def _position_from_pose(
        pose: TaskPose | np.ndarray | None, name: str
    ) -> jax.Array | None:
        if pose is None:
            return None
        position = pose.position if isinstance(pose, TaskPose) else pose
        array = np.asarray(position, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"{name} position must have shape (3,), got {array.shape}.")
        return jnp.asarray(array, dtype=jnp.float32)

    def _seed_nominal_solution(self, state: jax.Array, goal_q: jax.Array) -> None:
        self.controller.sampler.optimal_samples = (
            self.planner.nominal_torque_sequence_to_goal(
                state, goal_q, self.config.MPC.horizon, self.config.MPC.dt
            )
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
        return tuple(float(entry) for entry in np.asarray(value).reshape(-1))


class PandaPregraspController:
    """Synchronous non-ROS adapter for the fixed Panda pregrasp task."""

    PHASE_NAME = "PREGRASP"
    GRIPPER_OPEN = 0.04

    def __init__(
        self,
        planner: PandaPregraspPlanner | None = None,
        config: Config | None = None,
        *,
        gains: bool = True,
        num_steps: int = 1,
        visualize: bool = False,
        gain_mode: str | None = None,
        compute_running_cost: bool = True,
        compute_task_diagnostics: bool = True,
        ocp_config=None,
    ) -> None:
        self.planner = PandaPregraspPlanner() if planner is None else planner
        if ocp_config is None:
            ocp_config = load_ocp_config("pregrasp")
        self.objective = PandaPregraspObjective(self.planner, ocp_config=ocp_config)
        self.config = (
            make_panda_pregrasp_config(
                self.planner, visualize=visualize, gains=gains, ocp=ocp_config
            )
            if config is None
            else config
        )
        self.gain_mode = _resolve_gain_mode(self.config, gain_mode)
        _apply_gain_mode_to_config(self.config, self.gain_mode)
        self.model, self.controller = build_model_and_solver(
            self.config, self.objective, custom_dynamics_fn=self.planner.dynamics
        )
        self._default_num_steps = self._validate_num_steps(num_steps)
        self._solution_initialized = False
        self._started = False
        self._compute_running_cost = compute_running_cost
        self._compute_task_diagnostics = compute_task_diagnostics

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self.controller.close()
        self._started = False

    def reset_runtime_state_after_warmup(self) -> None:
        """Discard warmup state while retaining compiled JAX executables."""
        self._started = False
        self._solution_initialized = False

    def warmup(
        self,
        phase: object | None = None,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
        num_steps: int | None = None,
    ) -> PlannerOutput:
        del phase, object_pose, target_pose
        return self.step(
            self.planner.home_q,
            jnp.zeros(self.planner.nv, dtype=jnp.float32),
            num_steps=num_steps,
            reset_guess=not self._solution_initialized,
        )

    def step(
        self,
        q: np.ndarray,
        v: np.ndarray,
        phase: object | None = None,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
        *,
        num_steps: int | None = None,
        reset_guess: bool = False,
    ) -> PlannerOutput:
        del phase, object_pose, target_pose
        prepare_start = time.perf_counter()
        q = PandaPickAndPlaceController._joint_vector(q, self.planner.nq, "q")
        v = PandaPickAndPlaceController._joint_vector(v, self.planner.nv, "v")
        state = jnp.concatenate([q, v], axis=0)
        effective_num_steps = self._validate_num_steps(
            self._default_num_steps if num_steps is None else num_steps
        )
        if reset_guess or not self._solution_initialized:
            self._seed_gravity_comp_solution(state)
        planner_prepare_time_ms = 1000.0 * (time.perf_counter() - prepare_start)

        command_start = time.perf_counter()
        input_sequence = self.controller.command(
            state,
            self.planner.reference_vec,
            shift_guess=True,
            num_steps=effective_num_steps,
        )
        input_sequence = jax.block_until_ready(input_sequence)
        planner_command_time_ms = 1000.0 * (time.perf_counter() - command_start)
        self._solution_initialized = True

        tau_ff = np.asarray(input_sequence[0], dtype=np.float32)
        gains_array = _current_gains_numpy(self.controller)
        planning_time_ms = 1000.0 * (time.perf_counter() - command_start)

        reference = self.planner.reference
        position_error = None
        orientation_error = None
        if self._compute_task_diagnostics:
            ee_pos, ee_x, ee_z = self.planner.ee_features(q)
            position_error = float(jnp.linalg.norm(ee_pos - reference.goal_pos))
            orientation_error = float(
                1.0
                - jnp.clip(jnp.dot(ee_z, reference.goal_z_axis), -1.0, 1.0)
                + 0.5
                * (
                    1.0
                    - jnp.clip(jnp.dot(ee_x, reference.goal_x_axis), -1.0, 1.0)
                )
            )
        running_cost = None
        if self._compute_running_cost:
            running_cost = float(
                jax.block_until_ready(
                    self.objective.running_cost(
                        state,
                        jnp.asarray(tau_ff, dtype=jnp.float32),
                        self.planner.reference_vec,
                    )
                )
            )

        return PlannerOutput(
            tau_ff=tau_ff,
            K=gains_array,
            phase=self.PHASE_NAME,
            next_phase=self.PHASE_NAME,
            gripper_command=GripperCommand(action="open", width=self.GRIPPER_OPEN),
            diagnostics=PlannerDiagnostics(
                planning_time_ms=planning_time_ms,
                running_cost=running_cost,
                gain_norm=float(np.linalg.norm(gains_array)),
                torque_norm=float(np.linalg.norm(tau_ff)),
                position_error=position_error,
                orientation_error=orientation_error,
                object_error=None,
                goal_position=np.asarray(reference.goal_pos, dtype=np.float32),
                gain_mode=self.gain_mode,
                planner_prepare_time_ms=planner_prepare_time_ms,
                planner_command_time_ms=planner_command_time_ms,
            ),
        )

    def predict_state(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = PandaPickAndPlaceController._joint_vector(q, self.planner.nq, "q")
        v = PandaPickAndPlaceController._joint_vector(v, self.planner.nv, "v")
        tau = PandaPickAndPlaceController._joint_vector(
            tau, self.planner.nu, "tau"
        )
        predicted = self.model.integrate_sim(
            jnp.concatenate([q, v], axis=0), tau, float(dt)
        )
        predicted = np.asarray(jax.block_until_ready(predicted), dtype=np.float32)
        return predicted[: self.planner.nq], predicted[self.planner.nq :]

    def _seed_gravity_comp_solution(self, state: jax.Array) -> None:
        q = jnp.asarray(state[: self.planner.nq], dtype=jnp.float32)
        tau = jnp.clip(
            self.planner.gravity_torques(q),
            -self.planner.torque_limits,
            self.planner.torque_limits,
        )
        self.controller.sampler.optimal_samples = jnp.tile(
            tau.astype(jnp.float32), (self.config.MPC.horizon, 1)
        )

    def _validate_num_steps(self, value: int) -> int:
        num_steps = int(value)
        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")
        return num_steps
