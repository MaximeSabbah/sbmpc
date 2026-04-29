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
    running_cost: float | None
    gain_norm: float
    torque_norm: float
    position_error: float
    orientation_error: float
    object_error: float | None
    goal_position: np.ndarray
    gain_mode: str | None = None
    foreground_planning_time_ms: float | None = None
    background_gain_time_ms: float | None = None
    async_gain_worker_running: bool = False
    async_gain_worker_error: str | None = None
    gain_age_cycles: float | None = None
    gain_window_fill: int = 0
    gain_completed_batch_count: int = 0
    gain_dropped_snapshot_count: int = 0


@dataclass(frozen=True)
class PlannerOutput:
    tau_ff: np.ndarray
    K: np.ndarray
    phase: Phase
    next_phase: Phase
    gripper_command: GripperCommand
    diagnostics: PlannerDiagnostics


GAIN_MODE_FEEDFORWARD = "feedforward"
GAIN_MODE_EXACT_ASYNC_FEEDBACK = "exact_async_feedback"
SUPPORTED_GAIN_MODES = {GAIN_MODE_FEEDFORWARD, GAIN_MODE_EXACT_ASYNC_FEEDBACK}
ASYNC_GAIN_WARMUP_TIMEOUT_SEC = 120.0


def _resolve_gain_mode(config: Config, gain_mode: str | None) -> str:
    if gain_mode is not None:
        mode = gain_mode.strip().lower()
        if mode not in SUPPORTED_GAIN_MODES:
            valid = ", ".join(sorted(SUPPORTED_GAIN_MODES))
            raise ValueError(f"unsupported gain_mode '{gain_mode}'. Choose from: {valid}.")
        return mode

    if not config.MPC.gains:
        return GAIN_MODE_FEEDFORWARD
    if config.MPC.gain_method == "exact":
        if (
            config.MPC.gain_samples_per_cycle is not None
            and config.MPC.gain_buffer_size is not None
        ):
            return GAIN_MODE_EXACT_ASYNC_FEEDBACK
        raise ValueError(
            "exact planner gains require gain_samples_per_cycle and "
            "gain_buffer_size so exact gradients can run in the background."
        )
    raise ValueError(f"unsupported gain method: {config.MPC.gain_method!r}.")


def _apply_gain_mode_to_config(config: Config, gain_mode: str) -> None:
    if gain_mode == GAIN_MODE_FEEDFORWARD:
        config.MPC.gains = False
        config.MPC.gain_samples_per_cycle = None
        config.MPC.gain_buffer_size = None
        return
    if gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
        config.MPC.gains = True
        config.MPC.gain_method = "exact"
        if (
            config.MPC.gain_samples_per_cycle is None
            or config.MPC.gain_buffer_size is None
        ):
            raise ValueError(
                "exact_async_feedback requires gain_samples_per_cycle and "
                "gain_buffer_size."
            )
        return
    raise ValueError(f"unsupported gain_mode: {gain_mode!r}.")


def _finite_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _async_gain_diagnostics(controller, gain_mode: str) -> dict[str, object]:
    if gain_mode != GAIN_MODE_EXACT_ASYNC_FEEDBACK:
        return {
            "background_gain_time_ms": None,
            "async_gain_worker_running": False,
            "async_gain_worker_error": None,
            "gain_age_cycles": None,
            "gain_window_fill": 0,
            "gain_completed_batch_count": 0,
            "gain_dropped_snapshot_count": 0,
        }

    status = controller.background_gain_status()
    return {
        "background_gain_time_ms": _finite_or_none(status.get("gain_refresh_ms")),
        "async_gain_worker_running": bool(status.get("worker_running", False)),
        "async_gain_worker_error": status.get("worker_error"),
        "gain_age_cycles": _finite_or_none(status.get("published_gain_age_cycles")),
        "gain_window_fill": int(status.get("rolling_window_fill", 0)),
        "gain_completed_batch_count": int(status.get("completed_batch_count", 0)),
        "gain_dropped_snapshot_count": int(status.get("dropped_snapshot_count", 0)),
    }


def _async_gain_batches_to_first_publish(config: Config) -> int:
    samples_per_cycle = int(config.MPC.gain_samples_per_cycle or 0)
    buffer_size = int(config.MPC.gain_buffer_size or 0)
    if samples_per_cycle <= 0 or buffer_size <= 0:
        return 1
    return max(1, (buffer_size + samples_per_cycle - 1) // samples_per_cycle)


def _wait_for_async_gain_batches(controller, min_completed_batches: int) -> None:
    ok = controller.wait_for_async_exact_gain_batches(
        min_completed_batches,
        timeout_sec=ASYNC_GAIN_WARMUP_TIMEOUT_SEC,
    )
    if ok:
        return
    status = controller.background_gain_status()
    raise TimeoutError(
        "Timed out waiting for async exact-gain warmup batches "
        f"({status.get('completed_batch_count', 0)}/{min_completed_batches} complete, "
        f"window_fill={status.get('rolling_window_fill', 0)}, "
        f"worker_error={status.get('worker_error')})."
    )


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
        gain_mode: str | None = None,
        compute_running_cost: bool = True,
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
        self.gain_mode = _resolve_gain_mode(self.config, gain_mode)
        _apply_gain_mode_to_config(self.config, self.gain_mode)
        self.model, self.controller = build_model_and_solver(
            self.config,
            self.objective,
            custom_dynamics_fn=self.planner.dynamics,
        )
        self._default_num_steps = self._validate_num_steps(num_steps)
        self._solution_initialized = False
        self._last_reference_signature: tuple[object, ...] | None = None
        self._started = False
        self._async_warmup_complete = False
        self._compute_running_cost = compute_running_cost

    def start(self) -> None:
        if self._started:
            return
        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            self.controller.start_background_gains(reset_published_gain=True)
        self._started = True

    def close(self) -> None:
        self.controller.close()
        self._started = False

    def warmup(
        self,
        phase: Phase = Phase.PREGRASP,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
        num_steps: int | None = None,
    ) -> PlannerOutput:
        async_completed_before = 0
        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            async_completed_before = int(
                self.controller.background_gain_status().get("completed_batch_count", 0)
            )
        output = self.step(
            self.planner.home_q,
            jnp.zeros(self.planner.nv, dtype=jnp.float32),
            phase,
            object_pose=object_pose,
            target_pose=target_pose,
            num_steps=num_steps,
            reset_guess=not self._solution_initialized,
        )
        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            if self._async_warmup_complete:
                _wait_for_async_gain_batches(
                    self.controller,
                    async_completed_before + 1,
                )
            else:
                output = self._warmup_exact_async_until_ready(
                    output,
                    phase=phase,
                    object_pose=object_pose,
                    target_pose=target_pose,
                    num_steps=num_steps,
                )
        return output

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
        reset_gain_state = (
            reset_guess
            or not self._solution_initialized
            or reference_signature != self._last_reference_signature
        )
        if (
            reset_guess
            or not self._solution_initialized
            or reference_signature != self._last_reference_signature
        ):
            self._seed_nominal_solution(state, reference.goal_q)
        if reset_gain_state:
            if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK and self._started:
                self.controller.stop_background_gains()
                self._started = False
            self.controller.reset_published_gains()
            self._async_warmup_complete = False

        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            self.start()

        start_time = time.time_ns()
        input_sequence = self.controller.command(
            state,
            self.planner.reference_vec,
            shift_guess=True,
            num_steps=effective_num_steps,
            update_gains=self.gain_mode != GAIN_MODE_EXACT_ASYNC_FEEDBACK,
            capture_gain_context=self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK,
        )
        input_sequence = jax.block_until_ready(input_sequence)
        self._solution_initialized = True
        self._last_reference_signature = reference_signature
        gains = np.asarray(jax.block_until_ready(self.controller.gains), dtype=np.float32)
        tau_ff = np.asarray(input_sequence[0], dtype=np.float32)
        planning_time_ms = 1e-6 * (time.time_ns() - start_time)
        gain_diag = _async_gain_diagnostics(self.controller, self.gain_mode)

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
            gain_mode=self.gain_mode,
            foreground_planning_time_ms=planning_time_ms,
            **gain_diag,
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

    def _warmup_exact_async_until_ready(
        self,
        output: PlannerOutput,
        *,
        phase: Phase,
        object_pose: TaskPose | np.ndarray | None,
        target_pose: TaskPose | np.ndarray | None,
        num_steps: int | None,
    ) -> PlannerOutput:
        if self._async_warmup_complete:
            return output

        q = self.planner.home_q
        v = jnp.zeros(self.planner.nv, dtype=jnp.float32)
        target_batches = _async_gain_batches_to_first_publish(self.config)
        target_fill = int(self.config.MPC.gain_buffer_size or 0)

        _wait_for_async_gain_batches(self.controller, 1)
        while True:
            status = self.controller.background_gain_status()
            completed = int(status.get("completed_batch_count", 0))
            fill = int(status.get("rolling_window_fill", 0))
            if completed >= target_batches and fill >= target_fill:
                break
            output = self.step(
                q,
                v,
                phase,
                object_pose=object_pose,
                target_pose=target_pose,
                num_steps=num_steps,
                reset_guess=False,
            )
            _wait_for_async_gain_batches(self.controller, completed + 1)

        completed = int(
            self.controller.background_gain_status().get("completed_batch_count", 0)
        )
        output = self.step(
            q,
            v,
            phase,
            object_pose=object_pose,
            target_pose=target_pose,
            num_steps=num_steps,
            reset_guess=False,
        )
        _wait_for_async_gain_batches(self.controller, completed + 1)
        self._async_warmup_complete = True
        return output

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


class PandaPregraspController:
    """Stable, non-ROS adapter around the Panda pregrasp planner."""

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
        reseed_every_step: bool = False,
        gain_mode: str | None = None,
        compute_running_cost: bool = True,
    ) -> None:
        self.planner = PandaPregraspPlanner() if planner is None else planner
        self.objective = PandaPregraspObjective(self.planner)
        self.config = (
            make_panda_pregrasp_config(
                self.planner,
                visualize=visualize,
                gains=gains,
            )
            if config is None
            else config
        )
        self.gain_mode = _resolve_gain_mode(self.config, gain_mode)
        _apply_gain_mode_to_config(self.config, self.gain_mode)
        self.model, self.controller = build_model_and_solver(
            self.config,
            self.objective,
            custom_dynamics_fn=self.planner.dynamics,
        )
        self._default_num_steps = self._validate_num_steps(num_steps)
        self._solution_initialized = False
        self._reseed_every_step = reseed_every_step
        self._started = False
        self._async_warmup_complete = False
        self._compute_running_cost = compute_running_cost

    def start(self) -> None:
        if self._started:
            return
        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            self.controller.start_background_gains(reset_published_gain=True)
        self._started = True

    def close(self) -> None:
        self.controller.close()
        self._started = False

    def warmup(
        self,
        phase: object | None = None,
        object_pose: TaskPose | np.ndarray | None = None,
        target_pose: TaskPose | np.ndarray | None = None,
        num_steps: int | None = None,
    ) -> PlannerOutput:
        del phase, object_pose, target_pose
        async_completed_before = 0
        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            async_completed_before = int(
                self.controller.background_gain_status().get("completed_batch_count", 0)
            )
        output = self.step(
            self.planner.home_q,
            jnp.zeros(self.planner.nv, dtype=jnp.float32),
            num_steps=num_steps,
            reset_guess=not self._solution_initialized,
        )
        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            if self._async_warmup_complete:
                _wait_for_async_gain_batches(
                    self.controller,
                    async_completed_before + 1,
                )
            else:
                output = self._warmup_exact_async_until_ready(
                    output,
                    num_steps=num_steps,
                )
        return output

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
        q = PandaPickAndPlaceController._joint_vector(q, self.planner.nq, "q")
        v = PandaPickAndPlaceController._joint_vector(v, self.planner.nv, "v")
        state = jnp.concatenate([q, v], axis=0)
        effective_num_steps = self._default_num_steps if num_steps is None else num_steps
        effective_num_steps = self._validate_num_steps(effective_num_steps)
        reset_gain_state = reset_guess or not self._solution_initialized
        if reset_guess or self._reseed_every_step or not self._solution_initialized:
            self._seed_nominal_solution(state)
        if reset_gain_state:
            if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK and self._started:
                self.controller.stop_background_gains()
                self._started = False
            self.controller.reset_published_gains()
            self._async_warmup_complete = False

        if self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK:
            self.start()

        reference = self.planner.reference
        start_time = time.time_ns()
        input_sequence = self.controller.command(
            state,
            self.planner.reference_vec,
            shift_guess=True,
            num_steps=effective_num_steps,
            update_gains=self.gain_mode != GAIN_MODE_EXACT_ASYNC_FEEDBACK,
            capture_gain_context=self.gain_mode == GAIN_MODE_EXACT_ASYNC_FEEDBACK,
        )
        input_sequence = jax.block_until_ready(input_sequence)
        self._solution_initialized = True
        gains = np.asarray(jax.block_until_ready(self.controller.gains), dtype=np.float32)
        tau_ff = np.asarray(input_sequence[0], dtype=np.float32)
        planning_time_ms = 1e-6 * (time.time_ns() - start_time)
        gain_diag = _async_gain_diagnostics(self.controller, self.gain_mode)

        ee_pos, ee_x, ee_z = self.planner.ee_features(q)
        position_error = float(jnp.linalg.norm(ee_pos - reference.goal_pos))
        orientation_error = float(
            (1.0 - jnp.clip(jnp.dot(ee_z, reference.goal_z_axis), -1.0, 1.0))
            + 0.5 * (1.0 - jnp.clip(jnp.dot(ee_x, reference.goal_x_axis), -1.0, 1.0))
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
        diagnostics = PlannerDiagnostics(
            planning_time_ms=planning_time_ms,
            running_cost=running_cost,
            gain_norm=float(np.linalg.norm(gains)),
            torque_norm=float(np.linalg.norm(tau_ff)),
            position_error=position_error,
            orientation_error=orientation_error,
            object_error=None,
            goal_position=np.asarray(reference.goal_pos, dtype=np.float32),
            gain_mode=self.gain_mode,
            foreground_planning_time_ms=planning_time_ms,
            **gain_diag,
        )
        return PlannerOutput(
            tau_ff=tau_ff,
            K=gains,
            phase=self.PHASE_NAME,
            next_phase=self.PHASE_NAME,
            gripper_command=GripperCommand(action="open", width=self.GRIPPER_OPEN),
            diagnostics=diagnostics,
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
        tau = PandaPickAndPlaceController._joint_vector(tau, self.planner.nu, "tau")
        state = jnp.concatenate([q, v], axis=0)
        predicted = self.model.integrate_sim(state, tau, float(dt))
        predicted = np.asarray(jax.block_until_ready(predicted), dtype=np.float32)
        return predicted[: self.planner.nq], predicted[self.planner.nq :]

    def _seed_nominal_solution(self, state: jax.Array) -> None:
        self.controller.sampler.optimal_samples = self.planner.nominal_torque_sequence_from_state(
            state,
            self.config.MPC.horizon,
            self.config.MPC.dt,
        )

    def _warmup_exact_async_until_ready(
        self,
        output: PlannerOutput,
        *,
        num_steps: int | None,
    ) -> PlannerOutput:
        if self._async_warmup_complete:
            return output

        q = self.planner.home_q
        v = jnp.zeros(self.planner.nv, dtype=jnp.float32)
        target_batches = _async_gain_batches_to_first_publish(self.config)
        target_fill = int(self.config.MPC.gain_buffer_size or 0)

        _wait_for_async_gain_batches(self.controller, 1)
        while True:
            status = self.controller.background_gain_status()
            completed = int(status.get("completed_batch_count", 0))
            fill = int(status.get("rolling_window_fill", 0))
            if completed >= target_batches and fill >= target_fill:
                break
            output = self.step(
                q,
                v,
                num_steps=num_steps,
                reset_guess=False,
            )
            _wait_for_async_gain_batches(self.controller, completed + 1)

        completed = int(
            self.controller.background_gain_status().get("completed_batch_count", 0)
        )
        output = self.step(
            q,
            v,
            num_steps=num_steps,
            reset_guess=False,
        )
        _wait_for_async_gain_batches(self.controller, completed + 1)
        self._async_warmup_complete = True
        return output

    @staticmethod
    def _validate_num_steps(value: int) -> int:
        num_steps = int(value)
        if num_steps <= 0:
            raise ValueError("num_steps must be strictly positive.")
        return num_steps
