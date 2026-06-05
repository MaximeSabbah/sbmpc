import jax.numpy as jnp
import numpy as np
import pytest

from sbmpc.controller.franka_emika_panda.planner_api import PandaPickAndPlaceController, PandaPregraspController, TaskPose
from sbmpc.controller.franka_emika_panda.panda_pregrasp import PandaPregraspPlanner, make_panda_pregrasp_config
from sbmpc.controller.franka_emika_panda.panda_pick_and_place import (
    Phase,
    PandaPickAndPlacePlanner,
    make_panda_pick_and_place_config,
)


_CONTROLLERS: list[PandaPickAndPlaceController | PandaPregraspController] = []


def track_controller(controller: PandaPickAndPlaceController | PandaPregraspController):
    _CONTROLLERS.append(controller)
    return controller


def teardown_module() -> None:
    for controller in reversed(_CONTROLLERS):
        controller.close()
    _CONTROLLERS.clear()


def build_controller(gains: bool) -> PandaPickAndPlaceController:
    planner = PandaPickAndPlacePlanner()
    config = make_panda_pick_and_place_config(planner, visualize=False, gains=gains)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    if gains:
        config.MPC.gain_samples_per_cycle = 4
        config.MPC.gain_buffer_size = 8
    return track_controller(PandaPickAndPlaceController(planner=planner, config=config))


def build_pregrasp_controller(gains: bool) -> PandaPregraspController:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=gains)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    if gains:
        config.MPC.gain_samples_per_cycle = 4
        config.MPC.gain_buffer_size = 8
    return track_controller(PandaPregraspController(planner=planner, config=config))


def test_panda_pick_and_place_controller_step_returns_ros_ready_shapes() -> None:
    controller = build_controller(gains=True)

    output = controller.step(
        controller.planner.home_q,
        jnp.zeros(controller.planner.nv, dtype=jnp.float32),
        Phase.PREGRASP,
    )

    assert output.tau_ff.shape == (controller.planner.nu,)
    assert output.K.shape == (controller.planner.nu, controller.planner.nx)
    assert output.phase == Phase.PREGRASP
    assert output.next_phase == Phase.DESCEND
    assert output.gripper_command.action == "open"
    assert np.isclose(output.gripper_command.width, controller.planner.GRIPPER_OPEN)
    assert output.diagnostics.goal_position.shape == (3,)
    assert np.isfinite(output.diagnostics.planning_time_ms)
    assert np.isfinite(output.diagnostics.running_cost)
    assert np.isfinite(output.diagnostics.gain_norm)
    assert np.isfinite(output.diagnostics.torque_norm)
    assert np.all(np.isfinite(output.tau_ff))
    assert np.all(np.isfinite(output.K))


def test_panda_pick_and_place_controller_step_uses_task_context() -> None:
    controller = build_controller(gains=False)
    target_pose = np.asarray(
        controller.planner.default_target_pos + jnp.array([0.01, -0.02, 0.03], dtype=jnp.float32),
        dtype=np.float32,
    )

    output = controller.step(
        controller.planner.home_q,
        jnp.zeros(controller.planner.nv, dtype=jnp.float32),
        Phase.TRANSPORT,
        object_pose=TaskPose(
            position=np.asarray(controller.planner.initial_object_pos, dtype=np.float32)
        ),
        target_pose=TaskPose(position=target_pose),
    )

    np.testing.assert_allclose(
        output.diagnostics.goal_position,
        target_pose + np.asarray(controller.planner.carry_offset, dtype=np.float32),
        atol=1e-6,
    )
    assert output.phase == Phase.TRANSPORT
    assert output.next_phase == Phase.PLACE
    assert output.gripper_command.action == "close"
    assert output.diagnostics.object_error is not None
    assert np.isfinite(output.diagnostics.object_error)


def test_panda_pick_and_place_controller_reuses_solution_guess_for_same_context() -> None:
    controller = build_controller(gains=True)
    call_count = 0
    original = controller.planner.nominal_torque_sequence_to_goal

    def wrapped(state, goal_q, horizon, dt):
        nonlocal call_count
        call_count += 1
        return original(state, goal_q, horizon, dt)

    controller.planner.nominal_torque_sequence_to_goal = wrapped

    q = controller.planner.home_q
    v = jnp.zeros(controller.planner.nv, dtype=jnp.float32)
    controller.step(q, v, Phase.PREGRASP)
    controller.step(q, v, Phase.PREGRASP)

    assert call_count == 1


def test_panda_pick_and_place_controller_resets_solution_guess_on_phase_change() -> None:
    controller = build_controller(gains=True)
    call_count = 0
    original = controller.planner.nominal_torque_sequence_to_goal

    def wrapped(state, goal_q, horizon, dt):
        nonlocal call_count
        call_count += 1
        return original(state, goal_q, horizon, dt)

    controller.planner.nominal_torque_sequence_to_goal = wrapped

    q = controller.planner.home_q
    v = jnp.zeros(controller.planner.nv, dtype=jnp.float32)
    controller.step(q, v, Phase.PREGRASP)
    controller.step(q, v, Phase.TRANSPORT)

    assert call_count == 2


def test_panda_pregrasp_controller_step_returns_ros_ready_shapes() -> None:
    controller = build_pregrasp_controller(gains=True)

    output = controller.step(
        controller.planner.home_q,
        jnp.zeros(controller.planner.nv, dtype=jnp.float32),
    )

    assert output.tau_ff.shape == (controller.planner.nu,)
    assert output.K.shape == (controller.planner.nu, controller.planner.nx)
    assert output.phase == "PREGRASP"
    assert output.next_phase == "PREGRASP"
    assert output.gripper_command.action == "open"
    assert np.isclose(output.gripper_command.width, controller.GRIPPER_OPEN)
    assert output.diagnostics.goal_position.shape == (3,)
    assert output.diagnostics.object_error is None
    assert np.isfinite(output.diagnostics.planning_time_ms)
    assert np.isfinite(output.diagnostics.running_cost)
    assert np.isfinite(output.diagnostics.gain_norm)
    assert np.isfinite(output.diagnostics.torque_norm)
    assert np.all(np.isfinite(output.tau_ff))
    assert np.all(np.isfinite(output.K))
    assert output.diagnostics.gain_mode == "exact_async_feedback"
    assert output.diagnostics.foreground_planning_time_ms is not None


def test_panda_pregrasp_controller_split_feedforward_and_gain_outputs() -> None:
    controller = build_pregrasp_controller(gains=True)

    feedforward = controller.step_feedforward(
        controller.planner.home_q,
        jnp.zeros(controller.planner.nv, dtype=jnp.float32),
    )
    gain = controller.latest_gain(feedforward.diagnostics)

    assert feedforward.tau_ff.shape == (controller.planner.nu,)
    assert not hasattr(feedforward, "K")
    assert gain.K.shape == (controller.planner.nu, controller.planner.nx)
    assert gain.diagnostics.foreground_planning_time_ms is not None
    assert np.isfinite(gain.diagnostics.torque_norm)
    assert np.isfinite(gain.diagnostics.gain_norm)
    assert np.all(np.isfinite(feedforward.tau_ff))
    assert np.all(np.isfinite(gain.K))


def test_panda_pregrasp_controller_exact_async_refreshes_without_worker() -> None:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    config.MPC.gain_samples_per_cycle = 2
    config.MPC.gain_buffer_size = 4
    controller = track_controller(
        PandaPregraspController(
            planner=planner,
            config=config,
            gain_mode="exact_async_feedback",
            compute_running_cost=False,
            compute_task_diagnostics=False,
        )
    )

    q = controller.planner.home_q
    v = jnp.zeros(controller.planner.nv, dtype=jnp.float32)
    feedforward = controller.step_feedforward(q, v)
    status = controller.controller.background_gain_status()
    assert status["worker_running"] is False

    gain = controller.refresh_gain_if_budget(feedforward.diagnostics, budget_sec=None)
    assert gain is not None
    assert gain.diagnostics.gain_mode == "exact_async_feedback"
    assert gain.diagnostics.async_gain_worker_running is False
    assert gain.diagnostics.gain_completed_batch_count == 1
    assert gain.diagnostics.gain_window_fill == 2


def test_panda_pregrasp_controller_can_skip_task_diagnostics() -> None:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=False)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    controller = track_controller(
        PandaPregraspController(
            planner=planner,
            config=config,
            compute_running_cost=False,
            compute_task_diagnostics=False,
        )
    )

    output = controller.step(
        controller.planner.home_q,
        jnp.zeros(controller.planner.nv, dtype=jnp.float32),
    )

    assert output.diagnostics.running_cost is None
    assert output.diagnostics.position_error is None
    assert output.diagnostics.orientation_error is None
    assert output.diagnostics.goal_position.shape == (3,)


def test_panda_pregrasp_controller_does_not_use_cubic_trajectory_seed() -> None:
    controller = build_pregrasp_controller(gains=True)
    call_count = 0
    original = controller.planner.nominal_torque_sequence_from_state

    def wrapped(state, horizon, dt):
        nonlocal call_count
        call_count += 1
        return original(state, horizon, dt)

    controller.planner.nominal_torque_sequence_from_state = wrapped

    q = controller.planner.home_q
    v = jnp.zeros(controller.planner.nv, dtype=jnp.float32)
    controller.step(q, v)
    controller.step(q, v)

    assert call_count == 0


def test_panda_pregrasp_controller_rejects_per_step_trajectory_reseed() -> None:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    config.MPC.gain_samples_per_cycle = 4
    config.MPC.gain_buffer_size = 8

    with pytest.raises(ValueError, match="reseed_every_step"):
        PandaPregraspController(
            planner=planner,
            config=config,
            reseed_every_step=True,
        )


def test_panda_pregrasp_controller_feedforward_mode_returns_zero_gain() -> None:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    controller = track_controller(
        PandaPregraspController(
            planner=planner,
            config=config,
            gain_mode="feedforward",
        )
    )

    output = controller.step(
        controller.planner.home_q,
        jnp.zeros(controller.planner.nv, dtype=jnp.float32),
    )

    assert output.diagnostics.gain_mode == "feedforward"
    assert np.allclose(output.K, np.zeros_like(output.K))


def test_panda_pregrasp_controller_exact_async_mode_does_not_start_worker() -> None:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    config.MPC.gain_samples_per_cycle = 2
    config.MPC.gain_buffer_size = 4
    controller = track_controller(
        PandaPregraspController(
            planner=planner,
            config=config,
            gain_mode="exact_async_feedback",
        )
    )

    try:
        controller.start()
        status = controller.controller.background_gain_status()
        assert status["worker_running"] is False
        assert controller.controller.phase0_exact_gain_status()["worker_running"] is False
        assert controller.gain_mode == "exact_async_feedback"
    finally:
        controller.close()

    assert not controller.controller.background_gain_status()["worker_running"]


def test_panda_pregrasp_controller_reset_runtime_state_after_warmup_keeps_optimizer_first_seed() -> None:
    planner = PandaPregraspPlanner()
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    config.MPC.gain_samples_per_cycle = 2
    config.MPC.gain_buffer_size = 4
    controller = track_controller(
        PandaPregraspController(
            planner=planner,
            config=config,
            gain_mode="exact_async_feedback",
        )
    )

    q = controller.planner.home_q
    v = jnp.zeros(controller.planner.nv, dtype=jnp.float32)
    controller.step(q, v)
    assert controller.controller.background_gain_status()["worker_running"] is False
    assert (
        controller.controller.phase0_exact_gain_status()["completed_batch_count"]
        >= 1
    )

    controller.reset_runtime_state_after_warmup()

    assert not controller.controller.background_gain_status()["worker_running"]
    assert (
        controller.controller.phase0_exact_gain_status()["completed_batch_count"]
        == 0
    )
    assert controller._started is False
    assert controller._solution_initialized is False

    call_count = 0
    original = controller.planner.nominal_torque_sequence_from_state

    def wrapped(state, horizon, dt):
        nonlocal call_count
        call_count += 1
        return original(state, horizon, dt)

    controller.planner.nominal_torque_sequence_from_state = wrapped
    controller.step(q, v)

    assert call_count == 0
