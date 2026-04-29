import argparse

import jax
import jax.numpy as jnp

from sbmpc.examples.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all


def post_update(sim) -> None:
    planner = sim.planner
    state = sim.current_state_vec()
    q = state[: planner.nq]
    ee_pos = planner.ee_position(q)
    err = float(jnp.linalg.norm(ee_pos - planner.goal_pos))
    gain_norm = float(jnp.linalg.norm(sim.controller.gains[0]))
    print(
        f"iter={sim.iter:04d} "
        f"ee={jnp.asarray(ee_pos)} "
        f"goal={jnp.asarray(planner.goal_pos)} "
        f"err={err:.3f} "
        f"plan={sim.last_command_time_ms:.1f}ms "
        f"|K|={gain_norm:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Panda PREGRASP MPPI on the Hydrax pick-and-place scene."
    )
    parser.add_argument(
        "--gains",
        action="store_true",
        help="Compute F-MPPI feedback gains. Disabled by default for fast behavior checks.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the number of simulation iterations.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override the MPC rollout horizon.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override the number of MPPI samples.",
    )
    parser.add_argument(
        "--control-points",
        type=int,
        default=None,
        help="Override the spline control point count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(
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
    if any(
        value is not None
        for value in (args.horizon, args.samples, args.control_points)
    ):
        config.MPC.initial_guess = planner.nominal_torque_sequence(
            config.MPC.horizon,
            config.MPC.dt,
        )

    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    sim.planner = planner
    sim.post_update = post_update

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"gains: {config.MPC.gains}")
    print(
        f"mpc: dt={config.MPC.dt:.3f}s horizon={config.MPC.horizon} "
        f"samples={config.MPC.num_parallel_computations} "
        f"control_points={config.MPC.num_control_points} "
        f"gain_method={config.MPC.gain_method} "
        f"gK={config.MPC.gain_samples_per_cycle} "
        f"gM={config.MPC.gain_buffer_size}"
    )
    print(f"visualize: {config.general.visualize}")
    print(f"home_q: {planner.home_q}")
    print(f"goal_pos: {planner.goal_pos}")
    print(f"goal_q: {planner.goal_q}")

    sim.simulate()
