from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from sbmpc.settings import Config

from functools import partial


class Gains(ABC):
    def __init__(self, config: Config) -> None:
        self.compute_gains = config.MPC.gains
        self.lam = jnp.asarray(config.MPC.lambda_mpc, dtype=config.general.dtype)
        self.cur_gains = jnp.zeros(
            (config.robot.nu, config.robot.nx), dtype=config.general.dtype
        )

    @abstractmethod
    def gains_computation(self, key) -> jnp.ndarray:
        pass


class MPPIGain(Gains):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    @partial(jax.jit, static_argnums=(0,))
    def gains_computation(self, costs, samples_delta, gradients) -> jnp.ndarray:
        if self.compute_gains:
            costs, best_cost = self._saturate_costs(costs)
            exp_costs = self._exp_costs_shifted(costs, best_cost)
            denom = jnp.maximum(
                jnp.sum(exp_costs),
                jnp.asarray(1e-8, dtype=exp_costs.dtype),
            )
            weights = exp_costs / denom
            gradients = jnp.nan_to_num(gradients)
            samples_delta = jnp.nan_to_num(samples_delta)
            weights_grad_shift = jnp.sum(weights[:, jnp.newaxis] * gradients, axis=0)
            weights_grad = (
                -self.lam * weights[:, jnp.newaxis] * (gradients - weights_grad_shift)
            )
            gains = jnp.sum(
                jnp.einsum("bi,bo->bio", weights_grad, samples_delta[:, 0, :]), axis=0
            ).T
        else:
            gains = self.cur_gains
        return jnp.nan_to_num(gains)

    def _saturate_costs(self, costs):
        # Saturate the cost in case of NaN or inf
        costs = jnp.where(jnp.isnan(costs), 1e6, costs)
        costs = jnp.where(jnp.isinf(costs), 1e6, costs)
        best_cost = costs.take(jnp.nanargmin(costs))
        return costs, best_cost

    def _exp_costs_shifted(self, costs, best_cost):
        return jnp.exp(-self.lam * (costs - best_cost))
