import jax
import jax.numpy as jnp
import numpy as np

from sbmpc.solvers import select_nominal_and_lowest_cost_indices


def test_selection_keeps_nominal_then_lowest_cost_samples():
    costs = jnp.array([8.0, 5.0, 2.0, 1.0, 6.0, 7.0, 4.0, 3.0])

    indices = jax.block_until_ready(
        select_nominal_and_lowest_cost_indices(costs, 4)
    )

    np.testing.assert_array_equal(np.asarray(indices), [0, 3, 2, 7])


def test_selection_does_not_duplicate_nominal_when_it_is_best():
    costs = jnp.array([0.1, 5.0, 2.0, 1.0])

    indices = jax.block_until_ready(
        select_nominal_and_lowest_cost_indices(costs, 3)
    )

    np.testing.assert_array_equal(np.asarray(indices), [0, 3, 2])


def test_selection_places_nonfinite_samples_last():
    costs = jnp.array([4.0, jnp.nan, 2.0, jnp.inf, 1.0])

    indices = jax.block_until_ready(
        select_nominal_and_lowest_cost_indices(costs, 3)
    )

    np.testing.assert_array_equal(np.asarray(indices), [0, 4, 2])


def test_selection_supports_nominal_only():
    costs = jnp.array([3.0, 1.0, 2.0])

    indices = jax.block_until_ready(
        select_nominal_and_lowest_cost_indices(costs, 1)
    )

    np.testing.assert_array_equal(np.asarray(indices), [0])
