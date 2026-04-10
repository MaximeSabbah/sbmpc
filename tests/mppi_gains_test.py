"""
In this script we compare the mppi gains obtained from out differentiation procedure with the ones of an optimal LQR
controller.
We use the same Linear dynamics and cost of the LQR.
"""

import control

import jax.numpy as jnp

import matplotlib.pyplot as plt

from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.simulation import build_model_and_solver


# Simple double integrator model
A = jnp.array([[0, 1], [0, 0]], dtype=jnp.float32)
B = jnp.array([[0], [1]], dtype=jnp.float32)

Q = jnp.array([[1, 0], [0, 1]], dtype=jnp.float32)
R = Q[0, 0]

Ad = jnp.eye(2, 2) + 0.05 * A
Bd = 0.05*B


K, S, E = control.dlqr(Ad, Bd, Q, R)
K = jnp.asarray(K, dtype=jnp.float32)
S = jnp.asarray(S, dtype=jnp.float32)
# Note that the feedback fains from the LQR are supposed to be applied like u = -K x
print("LQR gains: ", -K)

x = jnp.array([0.0, 0.0], dtype=jnp.float32)
x_des = jnp.array([0.5, 0.0], dtype=jnp.float32)
optimal_inputs = jnp.zeros((25, 2), dtype=jnp.float32)
for i in range(25):
    u = jnp.asarray(-K @ (x - x_des), dtype=jnp.float32)
    optimal_inputs = optimal_inputs.at[i, 0].set(u[0])
    x = Ad @ x + Bd @ u


# Redefine B matrix since mppi does not support single input systems (to be fixed)
B_mppi = jnp.array([[0, 0], [1, 0]], dtype=jnp.float32)

def dynamics(x, u, p):
    return (A @ x + B_mppi @ u).astype(jnp.float32)


class Objective(BaseObjective):
    def running_cost(self, state, inputs, reference):
        return 20*((state - reference).T @ Q @ (state - reference))

    def final_cost(self, state, reference):
        return 20*(state - reference).T @ S @ (state - reference)


if __name__ == "__main__":

    robot_config = settings.RobotConfig()

    robot_config.nq = 1
    robot_config.nv = 1
    robot_config.nu = 2

    robot_config.q_init = jnp.array([0.], dtype=jnp.float32)  # hovering position

    config = settings.Config(robot_config)

    config.general.integrator_type = "euler"

    config.MPC.dt = 0.05
    config.MPC.horizon = 25
    config.MPC.std_dev_mppi = jnp.array([0.5, 0.0])
    config.MPC.num_parallel_computations = 10000
    config.MPC.lambda_mpc = 2.0
    config.MPC.num_control_points = config.MPC.horizon
    config.MPC.gains = True

    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM

    objective = Objective()

    model, solver = build_model_and_solver(config, objective, custom_dynamics_fn=dynamics)

    solver.sampler.optimal_samples = optimal_inputs

    input = solver.command(jnp.array([0.0, 0.0], dtype=jnp.float32), jnp.array([0.5, 0.0], dtype=jnp.float32), False, num_steps=1).block_until_ready()

    mppi_gains = solver.gains[0]

    print("MPPI gains: ", mppi_gains)


    print("error norm: ", jnp.linalg.norm(mppi_gains + K, jnp.inf))

    config_fd = settings.Config(robot_config)
    config_fd.general.integrator_type = "euler"
    config_fd.MPC.dt = config.MPC.dt
    config_fd.MPC.horizon = config.MPC.horizon
    config_fd.MPC.std_dev_mppi = config.MPC.std_dev_mppi
    config_fd.MPC.num_parallel_computations = config.MPC.num_parallel_computations
    config_fd.MPC.lambda_mpc = config.MPC.lambda_mpc
    config_fd.MPC.num_control_points = config.MPC.num_control_points
    config_fd.MPC.gains = True
    config_fd.MPC.gain_method = "finite_difference"
    config_fd.MPC.gain_fd_scheme = "central"
    config_fd.MPC.gain_fd_epsilon = 1e-3
    config_fd.solver_dynamics = settings.DynamicsModel.CUSTOM
    config_fd.sim_dynamics = settings.DynamicsModel.CUSTOM

    _, solver_fd = build_model_and_solver(config_fd, objective, custom_dynamics_fn=dynamics)
    solver_fd.sampler.optimal_samples = optimal_inputs
    solver_fd.command(jnp.array([0.0, 0.0], dtype=jnp.float32), jnp.array([0.5, 0.0], dtype=jnp.float32), False, num_steps=1).block_until_ready()
    mppi_gains_fd = solver_fd.gains[0]
    print("MPPI finite-difference gains: ", mppi_gains_fd)
    print("finite-difference error norm: ", jnp.linalg.norm(mppi_gains_fd + K, jnp.inf))


