"""Unit tests for the cost-term / OCP factory (no heavy MJX planner)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sbmpc.costs import CostModel, FactoryObjective
from sbmpc.ocp import (
    OCPConfig,
    TermSpec,
    build_cost_model,
    load_ocp_config,
    ocp_config_from_dict,
    with_weight_overrides,
)


class FakePlanner:
    """Minimal kinematics provider: ee position == first two joint coords."""

    nq = 2
    nv = 2
    torque_limits = jnp.array([10.0, 10.0], dtype=jnp.float32)

    def ee_features(self, q):
        ee_pos = jnp.array([q[0], q[1], 0.0], dtype=jnp.float32)
        ee_x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
        ee_z = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
        return ee_pos, ee_x, ee_z


def _reference(planner, goal_pos, goal_q, weights=None):
    parts = [
        jnp.asarray(goal_pos, jnp.float32),
        jnp.asarray(goal_q, jnp.float32),
        jnp.array([1.0, 0.0, 0.0], jnp.float32),   # goal_x
        jnp.array([0.0, 0.0, -1.0], jnp.float32),  # goal_z
        jnp.zeros(planner.nv, jnp.float32),        # goal_tau
    ]
    if weights is not None:
        parts.append(jnp.asarray(weights, jnp.float32))
    return jnp.concatenate(parts)


def test_cost_model_weighted_sum_matches_hand_value() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(
            TermSpec("ee_translation_xy", 2.0),
            TermSpec("posture", 3.0),
        ),
        terminal_terms=(TermSpec("ee_position_sq", 5.0),),
    )
    model = build_cost_model(ocp, planner)
    ref = _reference(planner, [1.0, 1.0, 0.0], [0.5, 0.5])
    state = jnp.zeros(4, jnp.float32)  # q=[0,0], v=[0,0]
    inputs = jnp.zeros(2, jnp.float32)

    # 2*smooth_norm([1,1]) + 3*(0.25+0.25)
    expected_running = 2.0 * float(np.sqrt(2.0 + 1e-8)) + 3.0 * 0.5
    np.testing.assert_allclose(
        float(model.running(state, inputs, ref)), expected_running, rtol=1e-5
    )
    # 5 * ||[1,1,0]||^2 = 5*2
    np.testing.assert_allclose(
        float(model.terminal(state, ref)), 10.0, rtol=1e-5
    )


def test_factory_objective_delegates() -> None:
    planner = FakePlanner()
    ocp = OCPConfig("t", (TermSpec("posture", 1.0),), ())
    obj = FactoryObjective(build_cost_model(ocp, planner))
    ref = _reference(planner, [0.0, 0.0, 0.0], [1.0, 0.0])
    state = jnp.zeros(4, jnp.float32)
    np.testing.assert_allclose(
        float(obj.running_cost(state, jnp.zeros(2), ref)), 1.0, rtol=1e-5
    )


def test_ref_weight_index_scales_term() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(TermSpec("posture", 2.0, ref_weight_index=0),),
        terminal_terms=(),
        n_weights=1,
    )
    model = build_cost_model(ocp, planner)
    state = jnp.zeros(4, jnp.float32)
    # posture = (1-0)^2 + 0 = 1; weight 2 * ref_weight 4 = 8
    ref = _reference(planner, [0.0, 0.0, 0.0], [1.0, 0.0], weights=[4.0])
    np.testing.assert_allclose(
        float(model.running(state, jnp.zeros(2), ref)), 8.0, rtol=1e-5
    )


def test_with_weight_overrides_replaces_by_name() -> None:
    ocp = ocp_config_from_dict(
        {
            "running_terms": [
                {"name": "posture", "weight": 1.0},
                {"name": "joint_velocity", "weight": 0.1},
            ]
        }
    )
    patched = with_weight_overrides(ocp, {"joint_velocity": 5.0, "unknown": 9.0})
    weights = {t.name: t.weight for t in patched.running_terms}
    assert weights == {"posture": 1.0, "joint_velocity": 5.0}


def test_pregrasp_ocp_is_tuned_for_real_hardware_handoff() -> None:
    ocp = load_ocp_config("pregrasp")
    running = {term.name: term.weight for term in ocp.running_terms}
    terminal = {term.name: term.weight for term in ocp.terminal_terms}

    assert ocp.mpc.dt == 0.04
    assert ocp.mpc.horizon == 10
    assert ocp.mpc.std_dev_scale == 0.08
    assert running["ee_translation_xy"] == 100.0
    assert running["orientation"] == 60.0
    assert running["posture"] == 10.0
    assert running["control_regularization"] == 1.0
    assert running["joint_velocity"] == 5.0
    assert running["mechanical_power"] == 0.005
    assert terminal["ee_position_sq"] == 1200.0
    assert terminal["orientation"] == 60.0
    assert terminal["posture"] == 10.0
    assert terminal["joint_velocity"] == 5.0
