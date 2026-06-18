from sbmpc.model import BaseModel
from sbmpc.settings import Config
from sbmpc.sampler import Sampler
from sbmpc.gains import Gains

import jax.numpy as jnp
import jax

from functools import partial

from abc import ABC, abstractmethod

from sbmpc.filter import cubic_spline_matrix


class BaseObjective(ABC):
    def __init__(self, robot_model=None):
        self.robot_model = robot_model

    @abstractmethod
    def running_cost(self, state, inputs, reference):
        pass

    def final_cost(self, state, reference):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def cost_and_constraints(self, state, inputs, reference, previous_inputs=None):
        del previous_inputs
        return self.running_cost(state, inputs, reference) + jnp.sum(
            self.make_barrier(self.constraints(state, inputs, reference))
        )

    def initial_previous_input_reference(self, reference, fallback):
        del reference
        return fallback

    def final_cost_and_constraints(self, state, reference):
        return self.final_cost(state, reference) + jnp.sum(
            self.make_barrier(self.terminal_constraints(state, reference))
        )

    def make_barrier(self, constraint_array):
        constraint_array = jnp.asarray(constraint_array, dtype=jnp.float32)
        constraint_array = jnp.where(
            constraint_array > 0,
            jnp.asarray(1e3, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return constraint_array

    def constraints(self, state, inputs, reference):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def terminal_constraints(self, state, reference):
        return jnp.asarray(0.0, dtype=jnp.float32)



@partial(jax.jit, static_argnums=(1,))
def select_nominal_and_lowest_cost_indices(costs, sample_count):
    """Keep sample 0 and fill the gain batch with the lowest-cost samples."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count > costs.shape[0]:
        raise ValueError("sample_count cannot exceed the number of costs")
    if sample_count == 1:
        return jnp.zeros((1,), dtype=jnp.int32)

    finite_costs = jnp.where(jnp.isfinite(costs[1:]), costs[1:], jnp.inf)
    _, lowest_indices = jax.lax.top_k(-finite_costs, sample_count - 1)
    return jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.int32),
            lowest_indices.astype(jnp.int32) + 1,
        ]
    )


class RolloutGenerator:
    def __init__(self, model: BaseModel, objective: BaseObjective, config: Config):
        """
        Initializes the rollout generator with the model, the objective, configurations and initial guess.
        Parameters
        ----------
        model: BaseModel
            The model propagated during rollouts.
        objective: BaseObjective
            Required to compute the cost function in the rollout.
        config_mpc: ConfigMPC
            Contains the MPC related parameters such as the time horizon, number of samples, etc.
        """

        self.dtype_general = config.general.dtype
        self.device = config.general.device

        self.model = model
        self.objective = objective

        self.config = config

        # Sampling time for discrete time model
        self.dt = jnp.asarray(config.MPC.dt, dtype=self.dtype_general)
        self.horizon = config.MPC.horizon
        self.rollout_time_grid = (
            jnp.arange(self.horizon, dtype=self.dtype_general) * self.dt
        )
        # Control horizon of the MPC (steps)
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel
        # check if we need to move it
        self.num_parallel_computations = config.MPC.num_parallel_computations

        self.compute_gains = config.MPC.gains
        self.gain_method = config.MPC.gain_method
        self.compute_exact_gains = self.compute_gains and self.gain_method == "exact"
        configured_gain_samples = config.MPC.num_gain_samples
        self.num_gain_samples = (
            self.num_parallel_computations
            if configured_gain_samples is None
            else int(configured_gain_samples)
        )
        if (
            self.compute_exact_gains
            and self.num_gain_samples > self.num_parallel_computations
        ):
            raise ValueError(
                f"num_gain_samples ({self.num_gain_samples}) cannot exceed "
                f"num_parallel_computations ({self.num_parallel_computations})."
            )

        self.num_control_points = config.MPC.num_control_points
        self.control_points_sparsity = self.horizon // self.num_control_points

        self.input_max_full_horizon = jnp.tile(model.input_max, (self.horizon, 1))
        self.input_min_full_horizon = jnp.tile(model.input_min, (self.horizon, 1))

        self.control_spline_indices = jnp.round(
            jnp.linspace(0, self.horizon - 1, self.num_control_points)
        ).astype(jnp.int32)
        self.control_spline_grid = self.rollout_time_grid[self.control_spline_indices]
        if self.config.MPC.smoothing == "Spline":
            self.control_interp_matrix = cubic_spline_matrix(
                self.control_spline_grid,
                self.rollout_time_grid,
            )
        else:
            self.control_interp_matrix = None

        if self.compute_exact_gains:
            self.rollout_gradients_to_state = jax.jit(
                jax.vmap(
                    self.rollout_single_state_gradient,
                    in_axes=(None, None, 0),
                    out_axes=0,
                ),
                device=self.device,
            )
        else:
            self.rollout_gradients_to_state = None

        # Rename functions for cost during rollout
        self.cost_and_constraints = self.objective.cost_and_constraints
        self.final_cost_and_constraints = self.objective.final_cost_and_constraints

    def clip_input_single(self, control_variables):
        return jnp.clip(
            control_variables, self.input_min_full_horizon, self.input_max_full_horizon
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0))
    def rollout_all(self, initial_state, reference, control_variables):
        return self.rollout_single(initial_state, reference, control_variables)

    def interpolate_control(self, control_variables):
        """ "
        Interpolates the control variables over the full horizon or passes the control variables directly
        """
        if self.config.MPC.smoothing == "Spline":
            control_interp = self.control_interp_matrix @ control_variables
            return self.clip_input_single(control_interp)
        else:
            return self.clip_input_single(control_variables)

    def rollout_single(self, initial_state, reference, control_variables):
        cost = jnp.asarray(0.0, dtype=self.dtype_general)
        curr_state = initial_state

        control_variables = self.interpolate_control(control_variables)

        previous_input = self.objective.initial_previous_input_reference(
            reference[0, :], control_variables[0, :]
        )

        def cost_and_state_rollout(idx, cost_and_state):
            cost, curr_state, previous_input = cost_and_state
            inputs = control_variables[idx, :]
            cost += self.dt * self.cost_and_constraints(
                curr_state,
                inputs,
                reference[idx, :],
                previous_input,
            )
            next_state = self.model.integrate_rollout_single(
                curr_state, inputs, self.dt
            )

            return cost, next_state, inputs

        cost, final_state, _ = jax.lax.fori_loop(
            0,
            self.horizon,
            cost_and_state_rollout,
            (cost, curr_state, previous_input),
        )

        cost += self.final_cost_and_constraints(final_state, reference[self.horizon, :])

        return cost, control_variables

    def rollout_single_state_gradient(
        self, initial_state, reference, control_variables
    ):
        """Compute dJ/dx for one already-scored control sample.

        This is the Feedback-MPPI gain path, computed with forward-mode AD
        (jvp) so MJX dynamics remain differentiable. Reverse-mode AD through
        MJX's internal solver while loops is not supported by JAX.
        """

        def cost_from_state(state):
            cost, _ = self.rollout_single(state, reference, control_variables)
            return cost

        basis = jnp.eye(self.model.nx, dtype=self.dtype_general)
        return jax.vmap(
            lambda tangent: jax.jvp(cost_from_state, (initial_state,), (tangent,))[1]
        )(basis)

    @partial(jax.jit, static_argnums=(0,))
    def do_rollout(self, state, reference, optimal_samples, samples_delta):
        if self.config.MPC.smoothing == "Spline":
            control_vars_all = (
                optimal_samples[self.control_spline_indices, :] + samples_delta
            )
        else:
            control_vars_all = optimal_samples + samples_delta

        if reference.ndim == 1:
            reference = jnp.tile(reference, (self.horizon + 1, 1))

        sampled_control_vars = control_vars_all
        costs, control_actions_all = self.rollout_all(
            state, reference, sampled_control_vars
        )
        samples_delta_clipped = self.compute_samples_delta(
            control_actions_all, optimal_samples
        )

        if self.compute_exact_gains:
            gain_indices = select_nominal_and_lowest_cost_indices(
                costs,
                self.num_gain_samples,
            )
            gain_control_vars = sampled_control_vars[gain_indices]
            gradients = self.rollout_gradients_to_state(
                state, reference, gain_control_vars
            )
            gain_costs = costs[gain_indices]
            gain_samples = samples_delta_clipped[gain_indices]
        else:
            gradients = None
            gain_costs = None
            gain_samples = None

        return (
            samples_delta_clipped,
            costs,
            gain_samples,
            gain_costs,
            gradients,
        )

    def compute_samples_delta(self, control_action, optimal_samples):
        samples_delta_clipped = control_action - optimal_samples
        return samples_delta_clipped


class Controller:
    def __init__(
        self, rollout_gen: RolloutGenerator, sampler: Sampler, gains_obj: Gains
    ):
        self.rollout_gen = rollout_gen
        self.objective = rollout_gen.objective
        self.sampler = sampler
        self.gains_obj = gains_obj

    def command(
        self,
        state,
        reference,
        shift_guess=True,
        num_steps=1,
    ):
        optimal_samples = self.sampler.optimal_samples

        for _ in range(num_steps):
            previous_optimal_samples = optimal_samples
            raw_samples_delta = self.sampler.sample_input_sequence(
                self.sampler.master_key
            )
            (
                samples,
                costs,
                gain_samples,
                gain_costs,
                gradients,
            ) = self.rollout_gen.do_rollout(
                state,
                reference,
                previous_optimal_samples,
                raw_samples_delta,
            )
            optimal_samples = self.sampler.update(
                previous_optimal_samples, samples, costs
            )
            if self.gains_obj.compute_gains:
                self.gains_obj.cur_gains = self.gains_obj.gains_computation(
                    gain_costs, gain_samples, gradients
                )

        if shift_guess:
            self.sampler.optimal_samples = self._shift_guess(optimal_samples)
        else:
            self.sampler.optimal_samples = optimal_samples

        return optimal_samples

    def close(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _shift_guess(self, optimal_samples):
        optimal_samples_shifted = jnp.roll(optimal_samples, shift=-1, axis=0)
        optimal_samples_shifted = optimal_samples_shifted.at[-1, :].set(
            optimal_samples_shifted[-2:-1, :].reshape(-1)
        )
        return optimal_samples_shifted

    @property
    def gains(self):
        return self.gains_obj.cur_gains
