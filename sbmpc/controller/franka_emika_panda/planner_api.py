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
        reference_vec = self.planner.reference_vector_for_state(q, phase=phase)
        planner_prepare_time_ms = 1000.0 * (time.perf_counter() - prepare_start)

        command_start = time.perf_counter()
        input_sequence = self.controller.command(
            state,
            reference_vec,
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
                        reference_vec,
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
        self.ocp_config = ocp_config
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
        self._last_tau_ff: np.ndarray | None = None
        self._trajectory_start_q: jax.Array | None = None
        self._trajectory_duration_sec = 0.0
        self._trajectory_step_index = 0
        self._trajectory_q_refs: jax.Array | None = None
        self._trajectory_v_refs: jax.Array | None = None
        self._trajectory_u_refs: jax.Array | None = None
        self._trajectory_reference_vecs: jax.Array | None = None
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
        self._last_tau_ff = None
        self._reset_trajectory()

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
        previous_u = (
            None
            if reset_guess or not self._solution_initialized
            else self._last_tau_ff
        )
        reference_vec = self._reference_for_current_horizon(
            q,
            v,
            previous_u=previous_u,
            reset_trajectory=reset_guess or not self._solution_initialized,
        )
        if reset_guess or not self._solution_initialized:
            self._seed_nominal_solution_from_reference(reference_vec)
        planner_prepare_time_ms = 1000.0 * (time.perf_counter() - prepare_start)

        command_start = time.perf_counter()
        input_sequence = self.controller.command(
            state,
            reference_vec,
            shift_guess=True,
            num_steps=effective_num_steps,
        )
        input_sequence = jax.block_until_ready(input_sequence)
        planner_command_time_ms = 1000.0 * (time.perf_counter() - command_start)
        self._solution_initialized = True

        tau_ff = np.asarray(input_sequence[0], dtype=np.float32)
        self._last_tau_ff = tau_ff.copy()
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
                        self._running_cost_reference(reference_vec),
                    )
                )
            )

        if self._trajectory_enabled():
            self._trajectory_step_index += 1

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

    def _trajectory_enabled(self) -> bool:
        return bool(self.ocp_config.trajectory.enabled)

    def _reset_trajectory(self) -> None:
        self._trajectory_start_q = None
        self._trajectory_duration_sec = 0.0
        self._trajectory_step_index = 0
        self._trajectory_q_refs = None
        self._trajectory_v_refs = None
        self._trajectory_u_refs = None
        self._trajectory_reference_vecs = None

    def _initialize_trajectory(self, q: jax.Array) -> None:
        spec = self.ocp_config.trajectory
        dt = float(self.config.MPC.dt)
        horizon = int(self.config.MPC.horizon)
        start = jnp.asarray(q, dtype=jnp.float32)
        goal = jnp.asarray(self.planner.goal_q, dtype=jnp.float32)
        start_np = np.asarray(start, dtype=np.float64)
        goal_np = np.asarray(goal, dtype=np.float64)
        dq_abs = np.abs(goal_np - start_np)
        velocity_limit = (
            np.asarray(self.planner.velocity_limits, dtype=np.float64)
            * float(spec.max_velocity_fraction)
        )
        # Minimum-jerk peak speed is 1.875 * |dq| / duration.
        duration_from_limits = float(
            np.max(1.875 * dq_abs / np.maximum(velocity_limit, 1e-6), initial=0.0)
        )
        duration = max(float(spec.duration_sec), duration_from_limits, dt)
        trajectory_steps = int(np.ceil(duration / dt))
        sample_count = max(horizon + 1, trajectory_steps + horizon + 1)
        steps = np.arange(sample_count, dtype=np.float64)
        s = np.clip((steps * dt) / duration, 0.0, 1.0)
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        blend_dot = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        delta = goal_np - start_np
        q_ref = start_np[None, :] + blend[:, None] * delta[None, :]
        v_ref = blend_dot[:, None] * delta[None, :]

        self._trajectory_start_q = start
        self._trajectory_duration_sec = duration
        self._trajectory_step_index = 0
        self._trajectory_q_refs = jnp.asarray(q_ref, dtype=jnp.float32)
        self._trajectory_v_refs = jnp.asarray(v_ref, dtype=jnp.float32)
        self._trajectory_u_refs = self._u_references_for_q_refs(q_ref)
        self._trajectory_reference_vecs = self.planner.reference_vectors_for_states(
            self._trajectory_q_refs,
            self._trajectory_v_refs,
            self._trajectory_u_refs,
            self._trajectory_u_refs[0],
        )
        jax.block_until_ready(self._trajectory_q_refs)
        jax.block_until_ready(self._trajectory_v_refs)
        jax.block_until_ready(self._trajectory_u_refs)
        jax.block_until_ready(self._trajectory_reference_vecs)

    def _u_references_for_q_refs(self, q_refs: np.ndarray) -> jax.Array:
        if self.ocp_config.references.u_ref == "zero":
            return jnp.zeros((q_refs.shape[0], self.planner.nu), dtype=jnp.float32)
        if self.ocp_config.references.u_ref == "gravity_q_ref":
            return self.planner.gravity_torques_batch(q_refs)
        raise ValueError(f"unsupported u_ref policy: {self.ocp_config.references.u_ref}")

    def _previous_u_reference(
        self,
        previous_u: np.ndarray | None,
        first_u_ref: jax.Array,
    ) -> jax.Array:
        policy = self.ocp_config.references.u_prev_ref
        if policy == "previous_control":
            if previous_u is None:
                return first_u_ref
            return jnp.asarray(previous_u, dtype=jnp.float32)
        if policy == "u_ref":
            return first_u_ref
        if policy == "zero":
            return jnp.zeros(self.planner.nu, dtype=jnp.float32)
        raise ValueError(f"unsupported u_prev_ref policy: {policy}")

    def _reference_for_current_horizon(
        self,
        q: jax.Array,
        v: jax.Array,
        *,
        previous_u: np.ndarray | None,
        reset_trajectory: bool,
    ) -> jax.Array:
        if not self._trajectory_enabled():
            return self.planner.reference_vector_for_policy(
                q,
                v,
                self.ocp_config.references,
                previous_u,
            )

        if reset_trajectory or self._trajectory_start_q is None:
            self._initialize_trajectory(q)

        window = self._trajectory_reference_window(previous_u)
        if self.ocp_config.trajectory.horizon_reference == "constant":
            # Hold the current plan point across the whole horizon as a pure
            # position setpoint (zero velocity reference). The solver broadcasts
            # this 1-D reference, intentionally bypassing the horizon lookahead
            # for a calmer, non-anticipatory regulator.
            v_start = 9 + self.planner.nq + 2 * self.planner.nv
            return window[0, :].at[v_start : v_start + self.planner.nv].set(0.0)
        return window

    def _trajectory_reference_window(self, previous_u: np.ndarray | None) -> jax.Array:
        if self._trajectory_reference_vecs is None or self._trajectory_u_refs is None:
            raise RuntimeError("trajectory has not been initialized")
        horizon = int(self.config.MPC.horizon)
        window = horizon + 1
        sample_count = int(self._trajectory_reference_vecs.shape[0])
        start = min(self._trajectory_step_index, max(0, sample_count - window))
        stop = start + window
        refs = self._trajectory_reference_vecs[start:stop, :]
        first_u_prev = self._previous_u_reference(
            previous_u,
            self._trajectory_u_refs[start],
        )
        u_prev_start = 9 + self.planner.nq + self.planner.nv
        u_prev_stop = u_prev_start + self.planner.nv
        return refs.at[0, u_prev_start:u_prev_stop].set(first_u_prev)

    def _seed_nominal_solution_from_reference(self, reference_vec: jax.Array) -> None:
        u_start = 9 + self.planner.nq
        u_stop = u_start + self.planner.nu
        if reference_vec.ndim == 2:
            tau = reference_vec[: self.config.MPC.horizon, u_start:u_stop]
        else:
            tau = jnp.tile(reference_vec[u_start:u_stop], (self.config.MPC.horizon, 1))
        self.controller.sampler.optimal_samples = jnp.clip(
            tau.astype(jnp.float32),
            -self.planner.torque_limits,
            self.planner.torque_limits,
        )

    @staticmethod
    def _running_cost_reference(reference_vec: jax.Array) -> jax.Array:
        return reference_vec[0, :] if reference_vec.ndim == 2 else reference_vec

    def _validate_num_steps(self, value: int) -> int:
        num_steps = int(value)
        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")
        return num_steps
