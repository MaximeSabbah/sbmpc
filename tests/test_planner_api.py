import numpy as np
import jax.numpy as jnp

from sbmpc import PandaPickAndPlaceController, TaskPose
from sbmpc.panda_pick_and_place import (
    Phase,
    PandaPickAndPlacePlanner,
    make_panda_pick_and_place_config,
)


def build_controller(gains: bool) -> PandaPickAndPlaceController:
    planner = PandaPickAndPlacePlanner()
    config = make_panda_pick_and_place_config(planner, visualize=False, gains=gains)
    config.MPC.horizon = 4
    config.MPC.num_parallel_computations = 8
    config.MPC.num_control_points = 2
    return PandaPickAndPlaceController(planner=planner, config=config)


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
