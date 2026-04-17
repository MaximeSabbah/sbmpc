import jax
import jax.numpy as jnp

from sbmpc.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_model_and_solver


def test_panda_pregrasp_solver_step() -> None:
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)

    config.MPC.horizon = 8
    config.MPC.num_parallel_computations = 16
    config.MPC.num_control_points = 3

    _, solver = build_model_and_solver(
        config,
        objective,
        custom_dynamics_fn=planner.dynamics,
    )

    x0 = jnp.concatenate(
        [planner.home_q, jnp.zeros(planner.nv, dtype=jnp.float32)]
    )
    reference = objective.reference_vector()
    control_sequence = solver.command(
        x0, reference, shift_guess=False, num_steps=1
    )
    control_sequence = jax.block_until_ready(control_sequence)
    gains = jax.block_until_ready(solver.gains)

    assert control_sequence.shape == (config.MPC.horizon, planner.nu)
    assert gains.shape == (planner.nu, planner.nx)
    assert jnp.all(jnp.isfinite(control_sequence))
    assert jnp.all(jnp.isfinite(gains))


def test_panda_pregrasp_nominal_seed_supports_dt_schedule() -> None:
    planner = PandaPregraspPlanner()
    dt_array = planner.step_durations(8, 0.02, [(4, 1), (4, 4)])
    control_sequence = planner.nominal_torque_sequence(8, dt_array)

    assert dt_array.shape == (8,)
    assert control_sequence.shape == (8, planner.nu)
    assert jnp.all(jnp.isfinite(control_sequence))
