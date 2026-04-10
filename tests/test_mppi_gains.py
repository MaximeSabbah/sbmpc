import control
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


def lqr_seed():
    # Match the semi-implicit Euler integrator used by the default model.
    ad = jnp.array([[1.0, DT], [0.0, 1.0]], dtype=jnp.float32)
    bd = jnp.array([[DT * DT], [DT]], dtype=jnp.float32)
    k, s, _ = control.dlqr(ad, bd, Q, Q[0, 0])
    k = jnp.asarray(k, dtype=jnp.float32)
    s = jnp.asarray(s, dtype=jnp.float32)

    x = jnp.array([0.0, 0.0], dtype=jnp.float32)
    x_des = jnp.array([0.5, 0.0], dtype=jnp.float32)
    controls = jnp.zeros((HORIZON, 2), dtype=jnp.float32)
    for idx in range(HORIZON):
        u = jnp.asarray(-k @ (x - x_des), dtype=jnp.float32)
        controls = controls.at[idx, 0].set(u[0])
        x = ad @ x + bd @ u
    return controls, s


def build_solver(gain_method, terminal_cost):
    robot_config = settings.RobotConfig()
    robot_config.nq = 1
    robot_config.nv = 1
    robot_config.nu = 2
    robot_config.q_init = jnp.array([0.0], dtype=jnp.float32)

    config = settings.Config(robot_config)
    config.MPC.dt = DT
    config.MPC.horizon = HORIZON
    config.MPC.std_dev_mppi = jnp.array([0.5, 0.0], dtype=jnp.float32)
    config.MPC.num_parallel_computations = 3000
    config.MPC.lambda_mpc = 2.0
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = True
    config.MPC.gain_method = gain_method
    config.MPC.gain_fd_scheme = "central"
    config.MPC.gain_fd_epsilon = 1e-3
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM

    _, solver = build_model_and_solver(
        config, Objective(terminal_cost), custom_dynamics_fn=dynamics
    )
    return solver


def run_gain(gain_method):
    seed, terminal_cost = lqr_seed()
    solver = build_solver(gain_method, terminal_cost)
    solver.sampler.optimal_samples = seed
    solver.command(
        jnp.array([0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.0], dtype=jnp.float32),
        shift_guess=False,
        num_steps=1,
    ).block_until_ready()
    return jax.block_until_ready(solver.gains)


def test_finite_difference_mppi_gains_match_exact_ad_gains():
    exact = run_gain("exact")
    finite_difference = run_gain("finite_difference")

    assert exact.shape == (2, 2)
    assert finite_difference.shape == exact.shape
    assert jnp.all(jnp.isfinite(exact))
    assert jnp.all(jnp.isfinite(finite_difference))
    assert jnp.linalg.norm(finite_difference - exact, ord=jnp.inf) < 2e-2
