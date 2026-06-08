import argparse
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from sbmpc.controller.franka_emika_panda.panda_pick_and_place import (
    Phase,
    PandaPickAndPlaceObjective,
    PandaPickAndPlacePlanner,
    make_panda_pick_and_place_config,
)
from sbmpc.simulation import build_model_and_solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scripted Panda pick-and-place phases with torque MPPI."
    )
    parser.add_argument("--gains", action="store_true", help="Compute finite-difference MPPI gains.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--iterations", type=int, default=None, help="Maximum control iterations.")
    parser.add_argument("--horizon", type=int, default=None, help="Override MPC horizon.")
    parser.add_argument("--samples", type=int, default=None, help="Override MPPI samples.")
    parser.add_argument("--control-points", type=int, default=None, help="Override spline control points.")
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Keep looping after DONE instead of stopping automatically.",
    )
    return parser.parse_args()


def move_towards(value: float, target: float, max_step: float) -> float:
    return float(value + np.clip(target - value, -max_step, max_step))


def sync_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    planner: PandaPickAndPlacePlanner,
    state: jax.Array,
    object_pos: jax.Array,
    finger_width: float,
) -> None:
    q = np.asarray(state[: planner.nq])
    data.qpos[: planner.nq] = q
    data.qvel[: planner.nv] = np.asarray(state[planner.nq : planner.nq + planner.nv])
    data.qpos[7:9] = finger_width
    data.qvel[7:9] = 0.0
    data.qpos[9:12] = np.asarray(object_pos)
    data.qpos[12:16] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[9:15] = 0.0
    data.mocap_pos[0] = np.asarray(planner.default_target_pos)
    mujoco.mj_forward(model, data)


def update_scripted_object(
    planner: PandaPickAndPlacePlanner,
    state: jax.Array,
    object_pos: jax.Array,
    finger_width: float,
    attached: bool,
) -> tuple[jax.Array, bool]:
    ee_pos = planner.ee_position(state[: planner.nq])
    ee_err, _ = planner.pose_error(state, Phase.CLOSE)

    if planner.phase == Phase.CLOSE and finger_width <= planner.CLOSED_FINGER_WIDTH and ee_err <= 0.025:
        attached = True

    if attached and planner.phase in (Phase.CLOSE, Phase.LIFT, Phase.TRANSPORT, Phase.PLACE, Phase.OPEN):
        object_pos = ee_pos

    if planner.phase in (Phase.OPEN, Phase.RETREAT, Phase.DONE) and finger_width >= planner.OPEN_FINGER_WIDTH:
        attached = False
        object_pos = planner.default_target_pos

    if not attached and planner.phase in (Phase.PREGRASP, Phase.DESCEND, Phase.CLOSE):
        object_pos = planner.initial_object_pos

    return jnp.asarray(object_pos, dtype=jnp.float32), attached


def main() -> None:
    args = parse_args()

    planner = PandaPickAndPlacePlanner()
    objective = PandaPickAndPlaceObjective(planner)
    config = make_panda_pick_and_place_config(
        planner,
        visualize=not args.headless,
        gains=args.gains,
    )
    if args.iterations is not None:
        config.sim_iterations = args.iterations
    if args.horizon is not None:
        config.MPC.horizon = args.horizon
    if args.samples is not None:
        config.MPC.num_parallel_computations = args.samples
    if args.control_points is not None:
        config.MPC.num_control_points = args.control_points
    if any(value is not None for value in (args.horizon, args.samples, args.control_points)):
        config.MPC.initial_guess = planner.nominal_torque_sequence_from_state(
            jnp.concatenate([planner.home_q, jnp.zeros(planner.nv, dtype=jnp.float32)]),
            config.MPC.horizon,
            config.MPC.dt,
            Phase.PREGRASP,
        )

    model, controller = build_model_and_solver(
        config,
        objective,
        custom_dynamics_fn=planner.dynamics,
    )

    state = jnp.concatenate([planner.home_q, jnp.zeros(planner.nv, dtype=jnp.float32)])
    object_pos = jnp.asarray(planner.initial_object_pos, dtype=jnp.float32)
    finger_width = planner.GRIPPER_OPEN
    attached = False
    last_phase = planner.phase

    mj_model = mujoco.MjModel.from_xml_path(planner.scene_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    sync_scene(mj_model, mj_data, planner, state, object_pos, finger_width)

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"gains: {config.MPC.gains}")
    print(
        f"mpc: dt={config.MPC.dt:.3f}s horizon={config.MPC.horizon} "
        f"samples={config.MPC.num_parallel_computations} "
        f"control_points={config.MPC.num_control_points} "
        f"gain_method={config.MPC.gain_method} "
        f"gain_samples={config.MPC.num_gain_samples}"
    )
    print(f"visualize: {config.general.visualize}")
    print(f"object: {planner.initial_object_pos} -> target: {planner.default_target_pos}")

    def step_once(iteration: int) -> bool:
        nonlocal state, object_pos, finger_width, attached, last_phase

        t0 = time.time_ns()
        phase = planner.phase
        reference = objective.reference_vector(phase)
        controller.sampler.optimal_samples = planner.nominal_torque_sequence_from_state(
            state,
            config.MPC.horizon,
            config.MPC.dt,
            phase,
        )
        input_sequence = controller.command(state, reference, num_steps=1).block_until_ready()
        tau = input_sequence[0].block_until_ready()
        state = model.integrate(state, tau, config.sim.dt).block_until_ready()

        finger_width = move_towards(
            finger_width,
            planner.gripper_target(phase),
            max_step=0.010,
        )
        object_pos, attached = update_scripted_object(
            planner,
            state,
            object_pos,
            finger_width,
            attached,
        )
        changed = planner.update_phase(state, object_pos, finger_width)
        if changed:
            print(f"\n>>> Phase: {last_phase.name} -> {planner.phase.name}")
            last_phase = planner.phase
            controller.sampler.optimal_samples = planner.nominal_torque_sequence_from_state(
                state,
                config.MPC.horizon,
                config.MPC.dt,
                planner.phase,
            )

        plan_ms = 1e-6 * (time.time_ns() - t0)
        ee_err, ori_err = planner.pose_error(state)
        obj_err = planner.object_error(object_pos)
        gain_norm = float(jnp.linalg.norm(controller.gains))
        print(
            f"[{planner.phase.name:>9s}] "
            f"iter={iteration:04d} "
            f"plan={plan_ms:5.1f}ms "
            f"|K|={gain_norm:6.3f} "
            f"ee={100 * ee_err:4.1f}cm "
            f"obj={100 * obj_err:4.1f}cm "
            f"ori={ori_err:5.3f} "
            f"finger={100 * finger_width:4.1f}cm "
            f"attached={int(attached)}",
            end="\n" if args.headless else "\r",
        )
        return changed

    if args.headless:
        for iteration in range(config.sim_iterations):
            step_once(iteration)
            if planner.sequence_complete() and not args.keep_running:
                break
        print("\nDone.")
        return

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            loop_start = time.time()
            step_once(planner._hold_count + int(mj_data.time / config.sim.dt))
            sync_scene(mj_model, mj_data, planner, state, object_pos, finger_width)
            viewer.sync()
            mj_data.time += config.sim.dt
            if planner.sequence_complete() and not args.keep_running:
                break
            sleep_time = config.sim.dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nDone.")


if __name__ == "__main__":
    main()
