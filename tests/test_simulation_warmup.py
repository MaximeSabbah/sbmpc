from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sbmpc.simulation import warmup_controller


class FakeController:
    def __init__(self):
        self.sampler = SimpleNamespace(
            optimal_samples=jnp.ones((2, 1), dtype=jnp.float32),
            master_key=jnp.array([1, 2], dtype=jnp.uint32),
        )
        self.gains_obj = SimpleNamespace(
            compute_gains=True,
            cur_gains=jnp.ones((1, 2), dtype=jnp.float32),
        )
        self.calls = []

    @property
    def gains(self):
        return self.gains_obj.cur_gains

    def command(self, state, reference, *, shift_guess, num_steps):
        self.calls.append((shift_guess, num_steps))
        value = jnp.asarray(len(self.calls), dtype=jnp.float32)
        self.sampler.optimal_samples = jnp.full((2, 1), value)
        self.sampler.master_key = self.sampler.master_key + 1
        self.gains_obj.cur_gains = jnp.full((1, 2), value)
        return self.sampler.optimal_samples


def test_warmup_controller_uses_runtime_path_and_restores_state():
    controller = FakeController()
    initial_optimal = np.asarray(controller.sampler.optimal_samples)
    initial_key = np.asarray(controller.sampler.master_key)
    initial_gains = np.asarray(controller.gains)

    output = warmup_controller(
        controller,
        jnp.zeros(2, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        iterations=3,
    )

    assert controller.calls == [(True, 1)] * 3
    np.testing.assert_allclose(np.asarray(output), 3.0)
    np.testing.assert_array_equal(np.asarray(controller.sampler.optimal_samples), initial_optimal)
    np.testing.assert_array_equal(np.asarray(controller.sampler.master_key), initial_key)
    np.testing.assert_array_equal(np.asarray(controller.gains), initial_gains)


def test_warmup_controller_rejects_nonpositive_iterations():
    controller = FakeController()
    with pytest.raises(ValueError, match="warmup iterations"):
        warmup_controller(
            controller,
            jnp.zeros(2),
            jnp.zeros(1),
            iterations=0,
        )
