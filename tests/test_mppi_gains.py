import jax
import jax.numpy as jnp

from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_model_and_solver


A = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)
B = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
B_MPPI = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
Q = jnp.eye(2, dtype=jnp.float32)
DT = 0.05
HORIZON = 25


def dynamics(x, u, p):
    del p
    return (A @ x + B_MPPI @ u).astype(jnp.float32)


class Objective(BaseObjective):
    def __init__(self, terminal_cost):
        super().__init__()
        self.terminal_cost = terminal_cost

    def running_cost(self, state, inputs, reference):
        del inputs
        err = state - reference
        return jnp.asarray(20.0 * (err.T @ Q @ err), dtype=jnp.float32)

    def final_cost(self, state, reference):
        err = state - reference
        return jnp.asarray(20.0 * (err.T @ self.terminal_cost @ err), dtype=jnp.float32)


def linear_seed():
    controls = jnp.zeros((HORIZON, 2), dtype=jnp.float32)
    controls = controls.at[:, 0].set(jnp.linspace(0.25, 0.05, HORIZON))
    terminal_cost = jnp.array([[1.0, 0.0], [0.0, 0.5]], dtype=jnp.float32)
    return controls, terminal_cost


def build_solver(
    gain_method,
    terminal_cost,
    *,
    num_parallel_computations=3000,
    num_gain_samples=None,
):
    robot_config = settings.RobotConfig()
    robot_config.nq = 1
    robot_config.nv = 1
    robot_config.nu = 2
    robot_config.q_init = jnp.array([0.0], dtype=jnp.float32)

    config = settings.Config(robot_config)
    config.MPC.dt = DT
    config.MPC.horizon = HORIZON
    config.MPC.std_dev_mppi = jnp.array([0.5, 0.0], dtype=jnp.float32)
    config.MPC.num_parallel_computations = num_parallel_computations
    config.MPC.lambda_mpc = 2.0
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = True
    config.MPC.gain_method = gain_method
    config.MPC.num_gain_samples = num_gain_samples
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM

    _, solver = build_model_and_solver(
        config, Objective(terminal_cost), custom_dynamics_fn=dynamics
    )
    return solver


def run_gain():
    seed, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=32,
        num_gain_samples=8,
    )
    solver.sampler.optimal_samples = seed
    solver.command(
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.0], dtype=jnp.float32),
        shift_guess=False,
        num_steps=1,
    ).block_until_ready()
    return jax.block_until_ready(solver.gains)


def test_exact_mppi_gains_are_finite_and_nonzero():
    exact = run_gain()

    assert exact.shape == (2, 2)
    assert jnp.all(jnp.isfinite(exact))
    assert not jnp.allclose(exact, jnp.zeros_like(exact))
