from sbmpc.model import BaseModel
from sbmpc.settings import Config
from sbmpc.sampler import Sampler
from sbmpc.gains import Gains

import jax.numpy as jnp
import jax

from functools import partial
from dataclasses import dataclass
import time
import threading

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



@dataclass(frozen=True)
class ExactGainPlanContext:
    cycle_id: int
    state: jax.Array
    reference: jax.Array
    optimal_samples: jax.Array
    raw_samples_delta: jax.Array
    samples_delta_clipped: jax.Array
    nominal_costs: jax.Array


@dataclass(frozen=True)
class ExactGainSnapshot:
    cycle_id: int
    sample_indices: jax.Array
    state: jax.Array
    reference: jax.Array
    optimal_samples: jax.Array
    control_variables: jax.Array
    costs: jax.Array
    delta_u0: jax.Array


@dataclass(frozen=True)
class ProcessedGainBatch:
    cycle_id: int
    sample_indices: jax.Array
    costs: jax.Array
    delta_u0: jax.Array
    gradients: jax.Array


class RollingGainWindow:
    def __init__(self, capacity, batch_size, nu, nx, dtype, publish_stride=1):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.publish_stride = int(publish_stride)
        self.costs = jnp.zeros((self.capacity,), dtype=dtype)
        self.delta_u0 = jnp.zeros((self.capacity, nu), dtype=dtype)
        self.gradients = jnp.zeros((self.capacity, nx), dtype=dtype)
        self.fill = 0
        self.cursor = 0
        self.append_count = 0

    def reset(self):
        self.costs = jnp.zeros_like(self.costs)
        self.delta_u0 = jnp.zeros_like(self.delta_u0)
        self.gradients = jnp.zeros_like(self.gradients)
        self.fill = 0
        self.cursor = 0
        self.append_count = 0

    def _write_ring(self, target, values):
        batch_size = values.shape[0]
        start = self.cursor
        end = start + batch_size
        if end <= self.capacity:
            return target.at[start:end].set(values)
        split = self.capacity - start
        target = target.at[start:].set(values[:split])
        return target.at[: batch_size - split].set(values[split:])

    def append(self, batch: ProcessedGainBatch):
        if batch.costs.shape[0] != self.batch_size:
            raise ValueError(
                f"expected batch size {self.batch_size}, got {batch.costs.shape[0]}"
            )
        self.costs = self._write_ring(self.costs, batch.costs)
        self.delta_u0 = self._write_ring(self.delta_u0, batch.delta_u0)
        self.gradients = self._write_ring(self.gradients, batch.gradients)
        self.cursor = (self.cursor + self.batch_size) % self.capacity
        self.fill = min(self.capacity, self.fill + self.batch_size)
        self.append_count += 1

    def ready_to_publish(self):
        return (
            self.fill >= self.capacity
            and self.append_count % self.publish_stride == 0
        )

    def compute_gain(self, gains_obj: Gains):
        return gains_obj.gains_computation(
            self.costs,
            self.delta_u0[:, jnp.newaxis, :],
            self.gradients,
        )

    def ordered_costs(self):
        if self.fill < self.capacity:
            return self.costs[: self.fill]
        return jnp.concatenate(
            (self.costs[self.cursor :], self.costs[: self.cursor]),
            axis=0,
        )


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
            and gain_samples_per_cycle <= self.num_parallel_computations
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
        self._zero_gains = jnp.zeros_like(self.gains_obj.cur_gains)

        # Buffered-gain state (exact path only). Activates when BOTH knobs are set.
        # The full MPPI batch still runs rollout-only; the exact sensitivity pass is
        # restricted to the promoted subset below.
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
            self._sync_exact_window = RollingGainWindow(
                self._gain_M,
                self._gain_K,
                nu,
                nx,
                dtype,
                publish_stride=1,
            )
            self._phase0_exact_window = RollingGainWindow(
                self._gain_M,
                self._gain_K,
                nu,
                nx,
                dtype,
                publish_stride=1,
            )
            self._gain_cycle = 0
            self._phase0_capture_cycle = 0
            self._phase0_pending_context = None
            self._phase0_dropped_snapshot_count = 0
            self._phase0_queue_depth_max = 0
            self._phase0_first_gain_ready_cycle = None
            self._phase0_last_published_cycle = None
            self._phase0_last_refresh = {}
            self._async_exact_window = RollingGainWindow(
                self._gain_M,
                self._gain_K,
                nu,
                nx,
                dtype,
                publish_stride=1,
            )
            self._async_capture_cycle = 0
            self._async_pending_context = None
            self._async_dropped_snapshot_count = 0
            self._async_queue_depth_max = 0
            self._async_first_gain_ready_cycle = None
            self._async_last_published_cycle = None
            self._async_last_refresh = {}
            self._async_completed_batch_count = 0
            self._async_worker_error = None
        else:
            self._sync_exact_window = None
            self._phase0_exact_window = None
            self._async_exact_window = None
            self._phase0_capture_cycle = 0
            self._phase0_pending_context = None
            self._phase0_dropped_snapshot_count = 0
            self._phase0_queue_depth_max = 0
            self._phase0_first_gain_ready_cycle = None
            self._phase0_last_published_cycle = None
            self._phase0_last_refresh = {}
            self._async_capture_cycle = 0
            self._async_pending_context = None
            self._async_dropped_snapshot_count = 0
            self._async_queue_depth_max = 0
            self._async_first_gain_ready_cycle = None
            self._async_last_published_cycle = None
            self._async_last_refresh = {}
            self._async_completed_batch_count = 0
            self._async_worker_error = None

        self._gain_lock = threading.Lock()
        self._async_condition = threading.Condition()
        self._async_thread = None
        self._async_running = False

    def _get_current_gains(self):
        with self._gain_lock:
            return self.gains_obj.cur_gains

    def _set_current_gains(self, gains):
        with self._gain_lock:
            self.gains_obj.cur_gains = gains

    def command(
        self,
        state,
        reference,
        shift_guess=True,
        num_steps=1,
        update_gains=True,
        capture_gain_context=False,
    ):

        optimal_samples = self.sampler.optimal_samples
        gains = self._get_current_gains()

        for i in range(num_steps):
            previous_optimal_samples = optimal_samples
            raw_samples_delta = self.sampler.sample_input_sequence(self.sampler.master_key)
            samples, costs, gradients = self.rollout_gen.do_rollout(
                state, reference, previous_optimal_samples, raw_samples_delta, gains
            )
            optimal_samples = self.sampler.update(previous_optimal_samples, samples, costs)
            if capture_gain_context and self._gain_buffered and self.rollout_gen.gain_method == "exact":
                self._capture_exact_gain_context(
                    state,
                    reference,
                    previous_optimal_samples,
                    raw_samples_delta,
                    samples,
                    costs,
            )
            # update gains
            if not update_gains:
                new_gains = None
            elif self.gains_obj.compute_gains and self.rollout_gen.gain_method == "finite_difference":
                new_gains = self._finite_difference_gains(
                    state,
                    reference,
                    previous_optimal_samples,
                    raw_samples_delta,
                    samples,
                    costs,
                )
            elif self._gain_buffered and self.gains_obj.compute_gains and self.rollout_gen.gain_method == "exact":
                new_gains = self._buffered_exact_gains(
                    state,
                    reference,
                    previous_optimal_samples,
                    raw_samples_delta,
                    samples,
                    costs,
                )
            else:
                new_gains = self.gains_obj.gains_computation(costs, samples, gradients)
            if new_gains is not None:
                self._set_current_gains(new_gains)
       
        # update sampler best control vars
        if shift_guess:
            self.sampler.optimal_samples = self._shift_guess(optimal_samples)
        else:
            self.sampler.optimal_samples = optimal_samples
        
        return optimal_samples

    def _make_exact_gain_context(
        self,
        cycle_id,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    ):
        return ExactGainPlanContext(
            cycle_id=cycle_id,
            state=state,
            reference=reference,
            optimal_samples=optimal_samples,
            raw_samples_delta=raw_samples_delta,
            samples_delta_clipped=samples_delta_clipped,
            nominal_costs=nominal_costs,
        )

    def _select_exact_gain_sample_indices(self, cycle_id, costs=None):
        del cycle_id
        total = self.rollout_gen.num_parallel_computations
        if self._gain_K >= total:
            return jnp.arange(total, dtype=jnp.int32)
        if costs is None:
            return jnp.arange(self._gain_K, dtype=jnp.int32)
        if self._gain_K == 1:
            return jnp.zeros((1,), dtype=jnp.int32)

        _, top_non_nominal = jax.lax.top_k(-costs[1:], self._gain_K - 1)
        return jnp.concatenate(
            (
                jnp.zeros((1,), dtype=jnp.int32),
                top_non_nominal.astype(jnp.int32) + 1,
            ),
            axis=0,
        )

    def _pack_exact_gain_snapshot(self, context: ExactGainPlanContext, sample_indices):
        rg = self.rollout_gen
        raw_sub = context.raw_samples_delta[sample_indices]
        if rg.config.MPC.smoothing == "Spline":
            control_variables = (
                context.optimal_samples[rg.control_spline_indices, :] + raw_sub
            )
        else:
            control_variables = context.optimal_samples + raw_sub

        if context.reference.ndim == 1:
            reference = jnp.tile(context.reference, (rg.horizon + 1, 1))
        else:
            reference = context.reference

        samples_sub = context.samples_delta_clipped[sample_indices]
        return ExactGainSnapshot(
            cycle_id=context.cycle_id,
            sample_indices=sample_indices,
            state=context.state,
            reference=reference,
            optimal_samples=context.optimal_samples,
            control_variables=control_variables,
            costs=context.nominal_costs[sample_indices],
            delta_u0=samples_sub[:, 0, :],
        )

    def _process_exact_gain_snapshot(self, snapshot: ExactGainSnapshot):
        _, gradients = self.rollout_gen.rollout_sens_to_state(
            snapshot.state,
            snapshot.reference,
            snapshot.control_variables,
        )
        return ProcessedGainBatch(
            cycle_id=snapshot.cycle_id,
            sample_indices=snapshot.sample_indices,
            costs=snapshot.costs,
            delta_u0=snapshot.delta_u0,
            gradients=gradients,
        )

    def _refresh_exact_gain_context(self, context: ExactGainPlanContext, window: RollingGainWindow):
        t_select = time.perf_counter()
        sample_indices = self._select_exact_gain_sample_indices(
            context.cycle_id,
            context.nominal_costs,
        )
        jax.block_until_ready(sample_indices)
        subset_select_ms = (time.perf_counter() - t_select) * 1000.0

        t_pack = time.perf_counter()
        snapshot = self._pack_exact_gain_snapshot(context, sample_indices)
        jax.block_until_ready(snapshot.costs)
        jax.block_until_ready(snapshot.delta_u0)
        jax.block_until_ready(snapshot.control_variables)
        snapshot_pack_ms = (time.perf_counter() - t_pack) * 1000.0

        t_grad = time.perf_counter()
        batch = self._process_exact_gain_snapshot(snapshot)
        jax.block_until_ready(batch.gradients)
        gain_grad_ms = (time.perf_counter() - t_grad) * 1000.0

        window.append(batch)
        gain_published = False
        gain_synth_ms = 0.0
        new_gains = None
        if window.ready_to_publish():
            t_synth = time.perf_counter()
            new_gains = window.compute_gain(self.gains_obj)
            jax.block_until_ready(new_gains)
            gain_synth_ms = (time.perf_counter() - t_synth) * 1000.0
            gain_published = True

        return {
            "subset_select_ms": subset_select_ms,
            "snapshot_pack_ms": snapshot_pack_ms,
            "gain_grad_ms": gain_grad_ms,
            "gain_synth_ms": gain_synth_ms,
            "gain_refresh_ms": subset_select_ms
            + snapshot_pack_ms
            + gain_grad_ms
            + gain_synth_ms,
            "gain_published": gain_published,
            "source_cycle_id": context.cycle_id,
        }, new_gains

    def _empty_exact_gain_refresh_status(self):
        return {
            "subset_select_ms": 0.0,
            "snapshot_pack_ms": 0.0,
            "gain_grad_ms": 0.0,
            "gain_synth_ms": 0.0,
            "gain_refresh_ms": 0.0,
            "gain_published": False,
            "source_cycle_id": None,
        }

    def _capture_exact_gain_context(
        self,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    ):
        with self._async_condition:
            async_running = self._async_running
        if async_running:
            self._enqueue_async_gain_context(
                state,
                reference,
                optimal_samples,
                raw_samples_delta,
                samples_delta_clipped,
                nominal_costs,
            )
        else:
            self._capture_phase0_gain_context(
                state,
                reference,
                optimal_samples,
                raw_samples_delta,
                samples_delta_clipped,
                nominal_costs,
            )

    def _capture_phase0_gain_context(
        self,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    ):
        if self._phase0_pending_context is not None:
            self._phase0_dropped_snapshot_count += 1
        cycle_id = self._phase0_capture_cycle
        self._phase0_capture_cycle += 1
        self._phase0_pending_context = self._make_exact_gain_context(
            cycle_id,
            state,
            reference,
            optimal_samples,
            raw_samples_delta,
            samples_delta_clipped,
            nominal_costs,
        )
        self._phase0_queue_depth_max = max(self._phase0_queue_depth_max, 1)

    def reset_phase0_exact_gain_probe(self, reset_published_gain=True):
        if not self._gain_buffered or self.rollout_gen.gain_method != "exact":
            return
        self._phase0_exact_window.reset()
        self._phase0_pending_context = None
        self._phase0_dropped_snapshot_count = 0
        self._phase0_queue_depth_max = 0
        self._phase0_first_gain_ready_cycle = None
        self._phase0_last_published_cycle = None
        self._phase0_last_refresh = {}
        self._phase0_capture_cycle = 0
        if reset_published_gain:
            self._set_current_gains(self._zero_gains)

    def phase0_probe_status(self):
        age_cycles = float("nan")
        current_cycle = max(0, self._phase0_capture_cycle - 1)
        if self._phase0_last_published_cycle is not None:
            age_cycles = float(current_cycle - self._phase0_last_published_cycle)
        return {
            "first_gain_ready_cycle": self._phase0_first_gain_ready_cycle,
            "published_gain_age_cycles": age_cycles,
            "queue_depth_max": int(self._phase0_queue_depth_max),
            "dropped_snapshot_count": int(self._phase0_dropped_snapshot_count),
            "rolling_window_fill": int(
                self._phase0_exact_window.fill
                if self._phase0_exact_window is not None
                else 0
            ),
        }

    def phase0_refresh_exact_gains(self):
        if not self._gain_buffered or self.rollout_gen.gain_method != "exact":
            raise ValueError("Phase 0 exact-gain probe requires buffered exact gains.")
        if self._phase0_pending_context is None:
            result = self.phase0_probe_status()
            result.update(self._empty_exact_gain_refresh_status())
            self._phase0_last_refresh = result
            return result

        context = self._phase0_pending_context
        self._phase0_pending_context = None
        refresh, new_gains = self._refresh_exact_gain_context(
            context,
            self._phase0_exact_window,
        )
        if refresh["gain_published"]:
            self._set_current_gains(new_gains)
            self._phase0_last_published_cycle = context.cycle_id
            if self._phase0_first_gain_ready_cycle is None:
                self._phase0_first_gain_ready_cycle = context.cycle_id

        result = self.phase0_probe_status()
        result.update(refresh)
        self._phase0_last_refresh = result
        return result
    
    def start_async_exact_gain_worker(self, reset_published_gain=True):
        if not self._gain_buffered or self.rollout_gen.gain_method != "exact":
            raise ValueError("Async exact-gain worker requires buffered exact gains.")
        with self._async_condition:
            if self._async_running:
                return
            self._async_exact_window.reset()
            self._async_capture_cycle = 0
            self._async_pending_context = None
            self._async_dropped_snapshot_count = 0
            self._async_queue_depth_max = 0
            self._async_first_gain_ready_cycle = None
            self._async_last_published_cycle = None
            self._async_last_refresh = {}
            self._async_completed_batch_count = 0
            self._async_worker_error = None
            self._async_running = True
            if reset_published_gain:
                self._set_current_gains(self._zero_gains)
            self._async_thread = threading.Thread(
                target=self._async_exact_gain_worker_loop,
                name="sbmpc-exact-gain-worker",
                daemon=True,
            )
            self._async_thread.start()

    def stop_async_exact_gain_worker(self, wait=True):
        with self._async_condition:
            thread = self._async_thread
            self._async_running = False
            self._async_pending_context = None
            self._async_condition.notify_all()
        if wait and thread is not None:
            thread.join()
        with self._async_condition:
            if self._async_thread is thread:
                self._async_thread = None

    def _enqueue_async_gain_context(
        self,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    ):
        with self._async_condition:
            cycle_id = self._async_capture_cycle
            self._async_capture_cycle += 1
            context = self._make_exact_gain_context(
                cycle_id,
                state,
                reference,
                optimal_samples,
                raw_samples_delta,
                samples_delta_clipped,
                nominal_costs,
            )
            if self._async_pending_context is not None:
                self._async_dropped_snapshot_count += 1
            self._async_pending_context = context
            self._async_queue_depth_max = max(self._async_queue_depth_max, 1)
            self._async_condition.notify()

    def _async_exact_gain_worker_loop(self):
        while True:
            with self._async_condition:
                while self._async_running and self._async_pending_context is None:
                    self._async_condition.wait()
                if not self._async_running:
                    return
                context = self._async_pending_context
                self._async_pending_context = None

            try:
                refresh, new_gains = self._refresh_exact_gain_context(
                    context,
                    self._async_exact_window,
                )
                if refresh["gain_published"]:
                    self._set_current_gains(new_gains)

                with self._async_condition:
                    if refresh["gain_published"]:
                        self._async_last_published_cycle = context.cycle_id
                        if self._async_first_gain_ready_cycle is None:
                            self._async_first_gain_ready_cycle = context.cycle_id
                    status = self._async_probe_status_locked()
                    status.update(refresh)
                    self._async_completed_batch_count += 1
                    status["completed_batch_count"] = self._async_completed_batch_count
                    self._async_last_refresh = status
                    self._async_condition.notify_all()
            except BaseException as exc:
                with self._async_condition:
                    self._async_worker_error = repr(exc)
                    self._async_running = False
                    self._async_condition.notify_all()
                return

    def _async_probe_status_locked(self):
        age_cycles = float("nan")
        current_cycle = max(0, self._async_capture_cycle - 1)
        if self._async_last_published_cycle is not None:
            age_cycles = float(current_cycle - self._async_last_published_cycle)
        return {
            "first_gain_ready_cycle": self._async_first_gain_ready_cycle,
            "published_gain_age_cycles": age_cycles,
            "queue_depth_max": int(self._async_queue_depth_max),
            "dropped_snapshot_count": int(self._async_dropped_snapshot_count),
            "rolling_window_fill": int(
                self._async_exact_window.fill
                if self._async_exact_window is not None
                else 0
            ),
            "completed_batch_count": int(self._async_completed_batch_count),
            "worker_error": self._async_worker_error,
        }

    def async_exact_gain_status(self):
        if not self._gain_buffered or self.rollout_gen.gain_method != "exact":
            result = self._empty_exact_gain_refresh_status()
            result.update(
                {
                    "first_gain_ready_cycle": None,
                    "published_gain_age_cycles": float("nan"),
                    "queue_depth_max": 0,
                    "dropped_snapshot_count": 0,
                    "rolling_window_fill": 0,
                    "completed_batch_count": 0,
                    "worker_error": None,
                }
            )
            return result
        with self._async_condition:
            result = self._async_probe_status_locked()
            if self._async_last_refresh:
                for key in (
                    "subset_select_ms",
                    "snapshot_pack_ms",
                    "gain_grad_ms",
                    "gain_synth_ms",
                    "gain_refresh_ms",
                    "gain_published",
                    "source_cycle_id",
                ):
                    result[key] = self._async_last_refresh.get(key, 0.0)
            else:
                result.update(self._empty_exact_gain_refresh_status())
            return result

    def wait_for_async_exact_gain_batches(self, min_completed_batches, timeout_sec=10.0):
        deadline = time.perf_counter() + timeout_sec
        with self._async_condition:
            while self._async_completed_batch_count < min_completed_batches:
                if self._async_worker_error is not None:
                    raise RuntimeError(self._async_worker_error)
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return False
                self._async_condition.wait(timeout=remaining)
            return True


    def _buffered_exact_gains(
        self,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    ):
        context = self._make_exact_gain_context(
            self._gain_cycle,
            state,
            reference,
            optimal_samples,
            raw_samples_delta,
            samples_delta_clipped,
            nominal_costs,
        )
        sample_indices = self._select_exact_gain_sample_indices(
            self._gain_cycle,
            context.nominal_costs,
        )
        snapshot = self._pack_exact_gain_snapshot(context, sample_indices)
        batch = self._process_exact_gain_snapshot(snapshot)
        self._sync_exact_window.append(batch)
        self._gain_cycle += 1

        if self._sync_exact_window.ready_to_publish():
            return self._sync_exact_window.compute_gain(self.gains_obj)
        return self._get_current_gains()

    @partial(jax.jit, static_argnums=(0,))
    def _fd_sample_indices(self, nominal_costs):
        del nominal_costs
        total = self.rollout_gen.num_parallel_computations
        subset_size = self.rollout_gen.gain_fd_num_samples
        if subset_size is None or subset_size >= total:
            return jnp.arange(total, dtype=jnp.int32)
        return jnp.arange(subset_size, dtype=jnp.int32)

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

        sample_indices = self._fd_sample_indices(nominal_costs)
        control_vars_fd = control_vars_all[sample_indices]
        delta_fd = samples_delta_clipped[sample_indices]
        costs_fd = nominal_costs[sample_indices]

        nominal_action = self.sampler.compute_action(
            optimal_samples, delta_fd, costs_fd
        )[0]

        def first_action_for_state(perturbed_state):
            costs, _ = rollout_gen.rollout_all(perturbed_state, reference, control_vars_fd)
            return self.sampler.compute_action(
                optimal_samples, delta_fd, costs
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
        return self._get_current_gains()
