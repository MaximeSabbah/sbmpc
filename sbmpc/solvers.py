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

    def cost_and_constraints(self, state, inputs, reference):
        return self.running_cost(state, inputs, reference) + jnp.sum(self.make_barrier(self.constraints(state, inputs, reference)))

    def final_cost_and_constraints(self, state, reference):
        return self.final_cost(state, reference) + jnp.sum(self.make_barrier(self.terminal_constraints(state, reference)))

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



class RolloutGenerator():

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
        self.rollout_time_grid = jnp.arange(self.horizon, dtype=self.dtype_general) * self.dt
        # Control horizon of the MPC (steps)
        # Monte-carlo samples, that is the number of trajectories that are evaluated in parallel 
        # check if we need to move it
        self.num_parallel_computations = config.MPC.num_parallel_computations

        self.compute_gains = config.MPC.gains
        self.gain_method = config.MPC.gain_method
        gain_samples_per_cycle = config.MPC.gain_samples_per_cycle
        gain_buffer_size = config.MPC.gain_buffer_size
        self.buffered_exact_gains = (
            self.compute_gains
            and self.gain_method == "exact"
            and gain_samples_per_cycle is not None
            and gain_buffer_size is not None
            and gain_samples_per_cycle < self.num_parallel_computations
        )
        self.compute_exact_gains = (
            self.compute_gains
            and self.gain_method == "exact"
            and not self.buffered_exact_gains
        )
        self.gain_fd_epsilon = jnp.asarray(config.MPC.gain_fd_epsilon, dtype=self.dtype_general)
        self.gain_fd_scheme = config.MPC.gain_fd_scheme
        self.gain_fd_num_samples = config.MPC.gain_fd_num_samples  # None = use all samples
        
        # Covariance of the input action
        # self.sigma_mppi = jnp.diag(config.MPC.std_dev_mppi**2)
  
        self.num_control_points = config.MPC.num_control_points
        self.control_points_sparsity = self.horizon // self.num_control_points

        self.input_max_full_horizon = jnp.tile(model.input_max, (self.horizon, 1))
        self.input_min_full_horizon = jnp.tile(model.input_min, (self.horizon, 1))
        self.clip_input = jax.jit(self.clip_input, device=self.device)

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

        #self.gains = jnp.zeros((model.nu, model.nx))
        # self.ctrl_sens_to_state = jax.jit(jax.jacfwd(self.compute_control_mppi, argnums=0, has_aux=True), device=self.device)
        if self.compute_gains and self.gain_method == "exact":
            self.rollout_sens_to_state = jax.jit(
                jax.vmap(
                    self.rollout_single_with_state_gradient,
                    in_axes=(None, None, 0),
                    out_axes=(0, 0),
                ),
                device=self.device,
            )
        else:
            self.rollout_sens_to_state = None

        # Rename functions for cost during rollout
        self.cost_and_constraints = self.objective.cost_and_constraints
        self.final_cost_and_constraints = self.objective.final_cost_and_constraints



    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def clip_input(self, control_variables):
        return jnp.clip(control_variables, self.input_min_full_horizon, self.input_max_full_horizon)

    def clip_input_single(self, control_variables):
        return jnp.clip(control_variables, self.input_min_full_horizon, self.input_max_full_horizon)
    

    @partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0))
    def rollout_all(self, initial_state, reference, control_variables):
        if self.config.MPC.sensitivity:
            return self.rollout_single_with_sensitivity(initial_state, reference, control_variables)
        else:
            return self.rollout_single(initial_state, reference, control_variables)
    
    def interpolate_control(self, control_variables):
        """"
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
        
        def cost_and_state_rollout(idx, cost_and_state):
            cost, curr_state = cost_and_state
            cost += self.dt*self.cost_and_constraints(curr_state, control_variables[idx, :], reference[idx, :])
            next_state = self.model.integrate_rollout_single(curr_state, control_variables[idx, :], self.dt)

            return cost, next_state

        cost, final_state = jax.lax.fori_loop(0, self.horizon, cost_and_state_rollout, (cost, curr_state))

        cost += self.dt*self.final_cost_and_constraints(final_state, reference[self.horizon, :])

        return cost, control_variables

    def rollout_single_with_state_gradient(self, initial_state, reference, control_variables):
        """Rollout cost plus dJ/dx using forward-mode AD.

        This is the Feedback-MPPI gain path from the original implementation,
        but computed with forward sensitivities so MJX dynamics remain
        differentiable. Reverse-mode AD through MJX's internal solver while
        loops is not supported by JAX.
        """
        cost_and_control = self.rollout_single(initial_state, reference, control_variables)

        def cost_from_state(state):
            cost, _ = self.rollout_single(state, reference, control_variables)
            return cost

        basis = jnp.eye(self.model.nx, dtype=self.dtype_general)
        gradient = jax.vmap(
            lambda tangent: jax.jvp(cost_from_state, (initial_state,), (tangent,))[1]
        )(basis)
        return cost_and_control, gradient


    def rollout_single_with_sensitivity(self, initial_state, reference, control_variables):
        cost = jnp.asarray(0.0, dtype=self.dtype_general)
        curr_state = initial_state
        curr_state_sens = jnp.zeros((self.model.nx, self.model.np))

        control_variables = self.interpolate_control(control_variables)

        def cost_and_state_rollout(idx, cost_and_state):
            cost, curr_state = cost_and_state
            cost += self.dt*self.cost_and_constraints(curr_state, control_variables[idx, :], reference[idx, :])
            next_state = self.model.integrate_rollout_single(curr_state, control_variables[idx, :], self.dt)

            return cost, next_state

        cost, final_state = jax.lax.fori_loop(0, self.horizon, cost_and_state_rollout, (cost, curr_state))

        cost += self.dt*self.final_cost_and_constraints(final_state, reference[self.horizon, :])

        return cost, control_variables

    # TODO UPDATE
    # @partial(jax.vmap, in_axes=(None, None, None, 0, None), out_axes=(0, 0))
    # def rollout_with_sensitivity(self, initial_state, reference, control_variables, mppi_gains):
    #     """
    #     Rollout of the system and associated parametric sensitivity dynamics
    #     :param initial_state:
    #     :param reference:
    #     :param control_variables:
    #     :param mppi_gains:
    #     :return:
    #     """
    #     cost = 0
    #     curr_state_sens = jnp.zeros((self.model.nx, self.model.np))
    #     curr_state = initial_state
    #     input_sequence = jnp.zeros((self.horizon, self.model.nu), dtype=self.dtype_general)
    #     if self.config.MPC["smoothing"] == "Spline":
    #         control_interp = cubic_spline(jnp.arange(0, self.horizon, self.control_points_sparsity),
    #                                     control_variables,
    #                                     jnp.arange(0, self.horizon))
    #         control_variables = self.clip_input_single(control_interp)

    #     for idx in range(self.horizon):
    #         curr_input = jax.lax.dynamic_slice(control_variables, (idx, 0), (1, self.model.nu)).reshape(-1)
    #         curr_input_sens = mppi_gains @ curr_state_sens
    #         cost_and_constraints = self.cost_and_constraints((curr_state, curr_state_sens), (curr_input, curr_input_sens), reference[idx, :])
    #         # Integrate the dynamics
    #         curr_state = self.model.integrate_rollout_single(curr_state[:self.model.nx], curr_input, self.dt)
    #         curr_state_sens = self.model.sensitivity_step(curr_state, curr_input, self.model.nominal_parameters, curr_state_sens, curr_input_sens, self.dt)
    #         cost += cost_and_constraints
    #         input_sequence = input_sequence.at[idx, :].set(curr_input)

    #     cost += self.final_cost_and_constraints((curr_state, curr_state_sens), reference[self.horizon, :])

    #     return cost, input_sequence

    @partial(jax.jit, static_argnums=(0,))  
    def do_rollout(self, state, reference, optimal_samples, samples_delta, gains):
        gradients = None

        if self.config.MPC.smoothing == "Spline":
            control_vars_all = optimal_samples[self.control_spline_indices, :] + samples_delta
        else:
            control_vars_all = optimal_samples + samples_delta

        # If the reference is just a state, repeat it along the horizon
        if reference.ndim == 1:
            reference = jnp.tile(reference, (self.horizon+1, 1))

        if self.compute_exact_gains:
            (costs, control_vars_all), gradients = self.rollout_sens_to_state(state, reference, control_vars_all)
        else:
            costs, control_vars_all = self.rollout_all(state, reference, control_vars_all)

        samples_delta_clipped = self.compute_samples_delta(control_vars_all, optimal_samples)
        

        return samples_delta_clipped, costs, gradients
    

    def compute_samples_delta(self, control_action, optimal_samples):
        samples_delta_clipped = (control_action - optimal_samples)
        return samples_delta_clipped


class Controller:
    def __init__(self, rollout_gen : RolloutGenerator, sampler : Sampler, gains_obj: Gains):

        self.rollout_gen = rollout_gen
        self.objective = rollout_gen.objective
        self.sampler = sampler
        self.gains_obj = gains_obj

        # Buffered-gain state (exact path only). Activates when BOTH knobs are set and
        # samples_per_cycle < total MPPI samples. The full MPPI batch still runs rollout-only;
        # the exact sensitivity pass is restricted to the buffered subset below.
        mpc = rollout_gen.config.MPC
        K = mpc.gain_samples_per_cycle
        M = mpc.gain_buffer_size
        N = rollout_gen.num_parallel_computations
        if (K is None) != (M is None):
            raise ValueError(
                "gain_samples_per_cycle and gain_buffer_size must be set together."
            )
        if K is not None and K > N:
            raise ValueError(
                f"gain_samples_per_cycle ({K}) cannot exceed the number of samples ({N})."
            )
        self._gain_buffered = rollout_gen.buffered_exact_gains
        if self._gain_buffered:
            if M % K != 0:
                raise ValueError(
                    f"gain_buffer_size ({M}) must be a positive multiple of "
                    f"gain_samples_per_cycle ({K})."
                )
            nu = rollout_gen.model.nu
            nx = rollout_gen.model.nx
            dtype = rollout_gen.dtype_general
            self._gain_K = int(K)
            self._gain_M = int(M)
            self._gain_stride = self._gain_M // self._gain_K
            self._gain_buf_costs = jnp.zeros((self._gain_M,), dtype=dtype)
            self._gain_buf_deltas = jnp.zeros((self._gain_M, nu), dtype=dtype)
            self._gain_buf_grads = jnp.zeros((self._gain_M, nx), dtype=dtype)
            self._gain_buf_fill = 0
            self._gain_buf_idx = 0
            self._gain_cycle = 0

    def command(self, state, reference, shift_guess=True, num_steps=1):

        optimal_samples = self.sampler.optimal_samples
        gains = self.gains_obj.cur_gains

        for i in range(num_steps):
            previous_optimal_samples = optimal_samples
            raw_samples_delta = self.sampler.sample_input_sequence(self.sampler.master_key)
            samples, costs, gradients = self.rollout_gen.do_rollout(
                state, reference, previous_optimal_samples, raw_samples_delta, gains
            )
            optimal_samples = self.sampler.update(previous_optimal_samples, samples, costs)
            # update gains
            if self.gains_obj.compute_gains and self.rollout_gen.gain_method == "finite_difference":
                self.gains_obj.cur_gains = self._finite_difference_gains(
                    state,
                    reference,
                    previous_optimal_samples,
                    raw_samples_delta,
                    samples,
                    costs,
                )
            elif self._gain_buffered and self.gains_obj.compute_gains and self.rollout_gen.gain_method == "exact":
                self.gains_obj.cur_gains = self._buffered_exact_gains(
                    state,
                    reference,
                    previous_optimal_samples,
                    raw_samples_delta,
                )
            else:
                self.gains_obj.cur_gains = self.gains_obj.gains_computation(costs, samples, gradients)
       
        # update sampler best control vars
        if shift_guess:
            self.sampler.optimal_samples = self._shift_guess(optimal_samples)
        else:
            self.sampler.optimal_samples = optimal_samples
        
        return optimal_samples
    

    def _buffered_exact_gains(
        self,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
    ):
        """Exact-path gain with sub-sampled backprop accumulated into a FIFO buffer.

        Each cycle backprops a rotating `gain_samples_per_cycle` subset (K) and appends
        (cost, Δu[0], ∇J) to a ring buffer of capacity `gain_buffer_size` (M).
        Once the buffer has filled and on every (M/K)-th cycle thereafter,
        runs the paper gain formula over the full M-sized buffer and caches K.
        Between ticks reuses the cached K.
        """
        rg = self.rollout_gen
        K = self._gain_K
        M = self._gain_M
        stride = self._gain_stride

        sample_indices = (
            jnp.arange(K, dtype=jnp.int32) + self._gain_cycle * K
        ) % rg.num_parallel_computations
        raw_sub = raw_samples_delta[sample_indices]
        if rg.config.MPC.smoothing == "Spline":
            control_vars_sub = optimal_samples[rg.control_spline_indices, :] + raw_sub
        else:
            control_vars_sub = optimal_samples + raw_sub

        if reference.ndim == 1:
            reference_tiled = jnp.tile(reference, (rg.horizon + 1, 1))
        else:
            reference_tiled = reference

        (costs_sub, control_vars_returned), grads_sub = rg.rollout_sens_to_state(
            state, reference_tiled, control_vars_sub
        )
        deltas_sub = rg.compute_samples_delta(control_vars_returned, optimal_samples)
        deltas_first = deltas_sub[:, 0, :]

        idx = self._gain_buf_idx
        self._gain_buf_costs = self._gain_buf_costs.at[idx:idx + K].set(costs_sub)
        self._gain_buf_deltas = self._gain_buf_deltas.at[idx:idx + K].set(deltas_first)
        self._gain_buf_grads = self._gain_buf_grads.at[idx:idx + K].set(grads_sub)

        self._gain_buf_idx = (idx + K) % M
        self._gain_buf_fill = min(M, self._gain_buf_fill + K)
        self._gain_cycle += 1

        # Update K on the stride tick once the buffer has filled at least once.
        if self._gain_buf_fill >= M and (self._gain_cycle % stride == 0):
            return self.gains_obj.gains_computation(
                self._gain_buf_costs,
                self._gain_buf_deltas[:, jnp.newaxis, :],
                self._gain_buf_grads,
            )
        return self.gains_obj.cur_gains

    @partial(jax.jit, static_argnums=(0,))
    def _finite_difference_gains(
        self,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    ):
        rollout_gen = self.rollout_gen
        eps = rollout_gen.gain_fd_epsilon
        nx = rollout_gen.model.nx
        eye = jnp.eye(nx, dtype=rollout_gen.dtype_general)

        if rollout_gen.config.MPC.smoothing == "Spline":
            control_vars_all = (
                optimal_samples[rollout_gen.control_spline_indices, :] + raw_samples_delta
            )
        else:
            control_vars_all = optimal_samples + raw_samples_delta

        if reference.ndim == 1:
            reference = jnp.tile(reference, (rollout_gen.horizon + 1, 1))

        # Subsample for FD if gain_fd_num_samples is set — reduces nx×N_fd rollouts
        n_fd = rollout_gen.gain_fd_num_samples
        if n_fd is not None and n_fd < control_vars_all.shape[0]:
            control_vars_fd = control_vars_all[:n_fd]
            optimal_fd = optimal_samples[:n_fd]
            delta_fd = samples_delta_clipped[:n_fd]
            costs_fd = nominal_costs[:n_fd]
        else:
            control_vars_fd = control_vars_all
            optimal_fd = optimal_samples
            delta_fd = samples_delta_clipped
            costs_fd = nominal_costs

        nominal_action = self.sampler.compute_action(
            optimal_fd, delta_fd, costs_fd
        )[0]

        def first_action_for_state(perturbed_state):
            costs, _ = rollout_gen.rollout_all(perturbed_state, reference, control_vars_fd)
            return self.sampler.compute_action(
                optimal_fd, delta_fd, costs
            )[0]

        plus_actions = jax.vmap(first_action_for_state)(state + eps * eye)
        if rollout_gen.gain_fd_scheme == "central":
            minus_actions = jax.vmap(first_action_for_state)(state - eps * eye)
            gains = ((plus_actions - minus_actions) / (2.0 * eps)).T
        else:
            gains = ((plus_actions - nominal_action[jnp.newaxis, :]) / eps).T

        return jnp.nan_to_num(gains).astype(rollout_gen.dtype_general)

    @partial(jax.jit, static_argnums=(0,))
    def _shift_guess(self, optimal_samples):
        optimal_samples_shifted = jnp.roll(optimal_samples, shift=-1, axis=0)
        optimal_samples_shifted = optimal_samples_shifted.at[-1, :].set(
            optimal_samples_shifted[-2:-1, :].reshape(-1))
        return optimal_samples_shifted
    
    @property
    def gains(self):
        return self.gains_obj.cur_gains
