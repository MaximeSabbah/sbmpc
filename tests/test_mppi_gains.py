import jax
import jax.numpy as jnp

from sbmpc import BaseObjective
import sbmpc.settings as settings
from sbmpc.solvers import ProcessedGainBatch, RollingGainWindow
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
    gain_samples_per_cycle=None,
    gain_buffer_size=None,
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
    config.MPC.gain_samples_per_cycle = gain_samples_per_cycle
    config.MPC.gain_buffer_size = gain_buffer_size
    config.solver_dynamics = settings.DynamicsModel.CUSTOM
    config.sim_dynamics = settings.DynamicsModel.CUSTOM

    _, solver = build_model_and_solver(
        config, Objective(terminal_cost), custom_dynamics_fn=dynamics
    )
    return solver


def run_gain():
    seed, terminal_cost = linear_seed()
    solver = build_solver("exact", terminal_cost)
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


def test_buffered_exact_gain_path_skips_full_batch_exact_rollout():
    seed, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=64,
        gain_samples_per_cycle=8,
        gain_buffer_size=16,
    )
    solver.sampler.optimal_samples = seed

    assert solver.rollout_gen.buffered_exact_gains
    assert not solver.rollout_gen.compute_exact_gains
    assert solver.rollout_gen.rollout_sens_to_state is not None

    state = jnp.array([0.0, 0.0], dtype=jnp.float32)
    reference = jnp.array([0.5, 0.0], dtype=jnp.float32)
    for _ in range(2):
        solver.command(
            state,
            reference,
            shift_guess=False,
            num_steps=1,
        ).block_until_ready()

    gains = jax.block_until_ready(solver.gains)
    assert gains.shape == (2, 2)
    assert jnp.all(jnp.isfinite(gains))


def test_buffered_exact_gain_path_allows_full_batch_promotion():
    _, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=8,
        gain_buffer_size=8,
    )

    assert solver.rollout_gen.buffered_exact_gains
    assert not solver.rollout_gen.compute_exact_gains


def test_exact_gain_snapshot_keeps_nominal_and_lowest_cost_samples():
    _, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=3,
        gain_buffer_size=6,
    )

    optimal_samples = jnp.arange(HORIZON * 2, dtype=jnp.float32).reshape(HORIZON, 2)
    raw_samples_delta = jnp.arange(8 * HORIZON * 2, dtype=jnp.float32).reshape(8, HORIZON, 2)
    samples_delta_clipped = raw_samples_delta + 1000.0
    nominal_costs = jnp.array([8.0, 5.0, 2.0, 1.0, 6.0, 7.0, 4.0, 3.0], dtype=jnp.float32)
    state = jnp.array([0.0, 0.0], dtype=jnp.float32)
    reference = jnp.array([0.5, 0.0], dtype=jnp.float32)

    context = solver._make_exact_gain_context(
        1,
        state,
        reference,
        optimal_samples,
        raw_samples_delta,
        samples_delta_clipped,
        nominal_costs,
    )
    indices = solver._select_exact_gain_sample_indices(
        context.cycle_id,
        context.nominal_costs,
    )
    snapshot = solver._pack_exact_gain_snapshot(context, indices)

    assert tuple(map(int, indices.tolist())) == (0, 3, 2)
    assert snapshot.reference.shape == (HORIZON + 1, 2)
    assert jnp.allclose(snapshot.costs, jnp.array([8.0, 1.0, 2.0], dtype=jnp.float32))
    assert jnp.allclose(snapshot.delta_u0, samples_delta_clipped[jnp.array([0, 3, 2]), 0, :])
    assert jnp.allclose(
        snapshot.control_variables,
        optimal_samples + raw_samples_delta[jnp.array([0, 3, 2])],
    )


def test_rolling_gain_window_replaces_oldest_batch_when_full():
    window = RollingGainWindow(
        capacity=4,
        batch_size=2,
        nu=1,
        nx=1,
        dtype=jnp.float32,
        publish_stride=1,
    )
    batch1 = ProcessedGainBatch(
        cycle_id=0,
        sample_indices=jnp.array([0, 1], dtype=jnp.int32),
        costs=jnp.array([1.0, 2.0], dtype=jnp.float32),
        delta_u0=jnp.array([[10.0], [20.0]], dtype=jnp.float32),
        gradients=jnp.array([[100.0], [200.0]], dtype=jnp.float32),
    )
    batch2 = ProcessedGainBatch(
        cycle_id=1,
        sample_indices=jnp.array([2, 3], dtype=jnp.int32),
        costs=jnp.array([3.0, 4.0], dtype=jnp.float32),
        delta_u0=jnp.array([[30.0], [40.0]], dtype=jnp.float32),
        gradients=jnp.array([[300.0], [400.0]], dtype=jnp.float32),
    )
    batch3 = ProcessedGainBatch(
        cycle_id=2,
        sample_indices=jnp.array([4, 5], dtype=jnp.int32),
        costs=jnp.array([5.0, 6.0], dtype=jnp.float32),
        delta_u0=jnp.array([[50.0], [60.0]], dtype=jnp.float32),
        gradients=jnp.array([[500.0], [600.0]], dtype=jnp.float32),
    )

    window.append(batch1)
    assert window.fill == 2
    assert not window.ready_to_publish()

    window.append(batch2)
    assert window.fill == 4
    assert window.ready_to_publish()
    assert tuple(map(float, window.ordered_costs().tolist())) == (1.0, 2.0, 3.0, 4.0)

    window.append(batch3)
    assert window.fill == 4
    assert tuple(map(float, window.ordered_costs().tolist())) == (3.0, 4.0, 5.0, 6.0)


def test_phase0_exact_refresh_matches_sync_exact_when_k_equals_m():
    seed, terminal_cost = linear_seed()
    state = jnp.array([0.0, 0.0], dtype=jnp.float32)
    reference = jnp.array([0.5, 0.0], dtype=jnp.float32)

    sync_solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=4,
        gain_buffer_size=4,
    )
    sync_solver.sampler.optimal_samples = seed
    sync_solver.command(state, reference, shift_guess=False, num_steps=1).block_until_ready()
    sync_gains = jax.block_until_ready(sync_solver.gains)

    probe_solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=4,
        gain_buffer_size=4,
    )
    probe_solver.sampler.optimal_samples = seed
    probe_solver.reset_phase0_exact_gain_probe(reset_published_gain=True)
    probe_solver.command(
        state,
        reference,
        shift_guess=False,
        num_steps=1,
        update_gains=False,
        capture_gain_context=True,
    ).block_until_ready()
    refresh = probe_solver.phase0_refresh_exact_gains()
    probe_gains = jax.block_until_ready(probe_solver.gains)

    assert refresh["gain_published"]
    assert jnp.allclose(probe_gains, sync_gains, atol=1e-6)


def test_phase0_exact_probe_stays_open_loop_until_window_is_full():
    seed, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=2,
        gain_buffer_size=4,
    )
    solver.sampler.optimal_samples = seed
    solver.reset_phase0_exact_gain_probe(reset_published_gain=True)
    state = jnp.array([0.0, 0.0], dtype=jnp.float32)
    reference = jnp.array([0.5, 0.0], dtype=jnp.float32)

    solver.command(
        state,
        reference,
        shift_guess=False,
        num_steps=1,
        update_gains=False,
        capture_gain_context=True,
    ).block_until_ready()
    refresh_0 = solver.phase0_refresh_exact_gains()
    gains_0 = jax.block_until_ready(solver.gains)

    solver.command(
        state,
        reference,
        shift_guess=False,
        num_steps=1,
        update_gains=False,
        capture_gain_context=True,
    ).block_until_ready()
    refresh_1 = solver.phase0_refresh_exact_gains()
    gains_1 = jax.block_until_ready(solver.gains)

    assert not refresh_0["gain_published"]
    assert jnp.allclose(gains_0, jnp.zeros_like(gains_0))
    assert refresh_1["gain_published"]
    assert refresh_1["first_gain_ready_cycle"] == 1
    assert jnp.all(jnp.isfinite(gains_1))


def test_background_exact_gain_worker_publishes_after_window_is_full():
    seed, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=2,
        gain_buffer_size=4,
    )
    solver.sampler.optimal_samples = seed
    solver.start_async_exact_gain_worker(reset_published_gain=True)
    state = jnp.array([0.0, 0.0], dtype=jnp.float32)
    reference = jnp.array([0.5, 0.0], dtype=jnp.float32)

    try:
        for _ in range(2):
            solver.command(
                state,
                reference,
                shift_guess=False,
                num_steps=1,
                update_gains=False,
                capture_gain_context=True,
            ).block_until_ready()
            assert solver.wait_for_async_exact_gain_batches(
                int(solver.async_exact_gain_status()["completed_batch_count"]) + 1,
                timeout_sec=10.0,
            )

        status = solver.async_exact_gain_status()
        gains = jax.block_until_ready(solver.gains)
        assert status["first_gain_ready_cycle"] == 1
        assert status["rolling_window_fill"] == 4
        assert status["worker_error"] is None
        assert jnp.all(jnp.isfinite(gains))
        assert not jnp.allclose(gains, jnp.zeros_like(gains))
    finally:
        solver.stop_async_exact_gain_worker()


def test_public_background_gain_lifecycle_is_idempotent():
    seed, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
        gain_samples_per_cycle=2,
        gain_buffer_size=4,
    )
    solver.sampler.optimal_samples = seed

    assert solver.start_background_gains(reset_published_gain=True)
    assert solver.background_gain_status()["worker_running"]

    solver.close()
    assert not solver.background_gain_status()["worker_running"]

    solver.stop_background_gains()
    solver.close()
    assert not solver.background_gain_status()["worker_running"]


def test_background_gain_lifecycle_noops_when_not_configured():
    _, terminal_cost = linear_seed()
    solver = build_solver(
        "exact",
        terminal_cost,
        num_parallel_computations=8,
    )

    assert not solver.start_background_gains(reset_published_gain=True)
    solver.stop_background_gains()
    solver.close()
    assert not solver.background_gain_status()["worker_running"]
