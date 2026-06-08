import jax
import jax.numpy as jnp

from sbmpc.controller.franka_emika_panda.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_model_and_solver


def test_panda_pregrasp_solver_step() -> None:
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=False, gains=True)
    assert config.general.integrator_type == "custom_discrete"


    config.MPC.horizon = 8
    config.MPC.num_parallel_computations = 16
    config.MPC.num_control_points = 3
    config.MPC.gain_samples_per_cycle = 8
    config.MPC.gain_buffer_size = 8

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
