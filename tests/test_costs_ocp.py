"""Unit tests for the cost-term / OCP factory (no heavy MJX planner)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

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

    def dynamics(self, state, inputs, params):
        del params
        v = state[self.nq :]
        qdd = 2.0 * inputs
        return jnp.concatenate([v, qdd])


def _reference(
    planner,
    ee_pos_ref,
    q_ref,
    weights=None,
    u_ref=None,
    v_ref=None,
):
    if u_ref is None:
        u_ref = jnp.zeros(planner.nv, jnp.float32)
    if v_ref is None:
        v_ref = jnp.zeros(planner.nv, jnp.float32)
    parts = [
        jnp.asarray(ee_pos_ref, jnp.float32),
        jnp.asarray(q_ref, jnp.float32),
        jnp.array([1.0, 0.0, 0.0], jnp.float32),   # ee_x_axis_ref
        jnp.array([0.0, 0.0, -1.0], jnp.float32),  # ee_z_axis_ref
        jnp.asarray(u_ref, jnp.float32),
        jnp.asarray(v_ref, jnp.float32),
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
            TermSpec("position_regularization", 3.0),
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
    ocp = OCPConfig("t", (TermSpec("position_regularization", 1.0),), ())
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
        running_terms=(TermSpec("position_regularization", 2.0, ref_weight_index=0),),
        terminal_terms=(),
        n_weights=1,
    )
    model = build_cost_model(ocp, planner)
    state = jnp.zeros(4, jnp.float32)
    # position_regularization = (1-0)^2 + 0 = 1; weight 2 * ref_weight 4 = 8
    ref = _reference(planner, [0.0, 0.0, 0.0], [1.0, 0.0], weights=[4.0])
    np.testing.assert_allclose(
        float(model.running(state, jnp.zeros(2), ref)), 8.0, rtol=1e-5
    )


def test_control_regularization_uses_local_torque_reference() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(TermSpec("control_regularization", 2.0),),
        terminal_terms=(),
    )
    model = build_cost_model(ocp, planner)
    state = jnp.zeros(4, jnp.float32)
    inputs = jnp.array([3.0, 4.0], jnp.float32)
    ref = _reference(
        planner,
        [0.0, 0.0, 0.0],
        [0.0, 0.0],
        u_ref=[1.0, 2.0],
    )

    # sum((inputs - u_ref)^2) = 4 + 4 = 8, weight = 2.
    np.testing.assert_allclose(
        float(model.running(state, inputs, ref)), 16.0, rtol=1e-5
    )


def test_position_regularization_uses_q_reference() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(TermSpec("position_regularization", 3.0),),
        terminal_terms=(),
    )
    model = build_cost_model(ocp, planner)
    state = jnp.array([2.0, 4.0, 0.0, 0.0], jnp.float32)
    ref = _reference(
        planner,
        [0.0, 0.0, 0.0],
        [1.0, 2.0],
    )

    # position_regularization = (2-1)^2 + (4-2)^2 = 5, weight = 3.
    np.testing.assert_allclose(
        float(model.running(state, jnp.zeros(2), ref)), 15.0, rtol=1e-5
    )


def test_joint_acceleration_uses_planner_dynamics() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(
            TermSpec("joint_acceleration", 3.0),
        ),
        terminal_terms=(),
    )
    model = build_cost_model(ocp, planner)
    ref = _reference(planner, [0.0, 0.0, 0.0], [0.0, 0.0])
    state = jnp.array([0.0, 0.0, 0.5, -0.5], jnp.float32)
    inputs = jnp.array([2.0, 4.0], jnp.float32)

    # Fake dynamics has qdd = 2 * inputs = [4, 8].
    # Squared sum = 80, weight = 3.
    np.testing.assert_allclose(
        float(model.running(state, inputs, ref)), 240.0, rtol=1e-5
    )


def test_velocity_regularization_accepts_per_joint_weights() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(
            TermSpec("velocity_regularization", 2.0, params={"weights": [1.0, 3.0]}),
        ),
        terminal_terms=(),
    )
    model = build_cost_model(ocp, planner)
    state = jnp.array([0.0, 0.0, 2.0, 4.0], jnp.float32)
    ref = _reference(
        planner,
        [0.0, 0.0, 0.0],
        [0.0, 0.0],
        v_ref=[0.0, 0.0],
    )

    # weighted velocity = 1*2^2 + 3*4^2 = 52, term weight = 2.
    np.testing.assert_allclose(
        float(model.running(state, jnp.zeros(2), ref)), 104.0, rtol=1e-5
    )


def test_joint_acceleration_accepts_per_joint_weights() -> None:
    planner = FakePlanner()
    ocp = OCPConfig(
        name="t",
        running_terms=(
            TermSpec("joint_acceleration", 0.5, params={"weights": [1.0, 4.0]}),
        ),
        terminal_terms=(),
    )
    model = build_cost_model(ocp, planner)
    ref = _reference(planner, [0.0, 0.0, 0.0], [0.0, 0.0])
    state = jnp.array([0.0, 0.0, 0.5, -0.5], jnp.float32)
    inputs = jnp.array([2.0, 4.0], jnp.float32)

    # Fake dynamics has qdd = 2 * inputs = [4, 8].
    # weighted acceleration = 1*4^2 + 4*8^2 = 272, term weight = 0.5.
    np.testing.assert_allclose(
        float(model.running(state, inputs, ref)), 136.0, rtol=1e-5
    )


def test_with_weight_overrides_replaces_by_name() -> None:
    ocp = ocp_config_from_dict(
        {
            "running_terms": [
                {"name": "position_regularization", "weight": 1.0},
                {"name": "velocity_regularization", "weight": 0.1},
            ]
        }
    )
    patched = with_weight_overrides(ocp, {"velocity_regularization": 5.0, "unknown": 9.0})
    weights = {t.name: t.weight for t in patched.running_terms}
    assert weights == {"position_regularization": 1.0, "velocity_regularization": 5.0}


def test_reference_policy_parses_valid_choices() -> None:
    ocp = ocp_config_from_dict(
        {
            "references": {
                "q_ref": "measured",
                "v_ref": "zero",
                "u_ref": "gravity_q_ref",
            }
        }
    )

    assert ocp.references.q_ref == "measured"
    assert ocp.references.v_ref == "zero"
    assert ocp.references.u_ref == "gravity_q_ref"


def test_reference_policy_rejects_ambiguous_control_reference() -> None:
    with pytest.raises(ValueError, match=r"references\.u_ref"):
        ocp_config_from_dict({"references": {"u_ref": "gravity_measured"}})


def test_reference_policy_rejects_home_position_reference() -> None:
    with pytest.raises(ValueError, match=r"references\.q_ref"):
        ocp_config_from_dict({"references": {"q_ref": "home"}})


def test_pregrasp_ocp_is_tuned_for_real_hardware_handoff() -> None:
    ocp = load_ocp_config("pregrasp")
    running_terms = {term.name: term for term in ocp.running_terms}
    terminal_terms = {term.name: term for term in ocp.terminal_terms}
    running = {name: term.weight for name, term in running_terms.items()}
    terminal = {name: term.weight for name, term in terminal_terms.items()}

    assert ocp.mpc.dt == 0.04
    assert ocp.mpc.horizon == 12
    assert ocp.mpc.num_control_points == 8
    assert ocp.mpc.std_dev_scale == 0.06
    assert ocp.references.q_ref == "measured"
    assert ocp.references.v_ref == "zero"
    assert ocp.references.u_ref == "gravity_q_ref"
    assert running["ee_translation_xy"] == 500.0
    assert running["ee_translation_z"] == 15.0
    assert running["orientation"] == 300.0
    assert running_terms["orientation"].params["x_axis_weight"] == 1.0
    assert running["position_regularization"] == 5.0
    assert running["control_regularization"] == 0.00005
    assert running_terms["control_regularization"].params["weights"] == [
        1.0,
        3.0,
        1.0,
        2.5,
        0.8,
        1.5,
        0.8,
    ]
    assert running["velocity_regularization"] == 80.0
    assert running_terms["velocity_regularization"].params["weights"] == [
        1.0,
        5.0,
        1.4,
        4.5,
        0.8,
        5.0,
        0.8,
    ]
    assert "joint_acceleration" not in running
    assert running["mechanical_power"] == 0.01
    assert running_terms["mechanical_power"].params["weights"] == [
        1.0,
        2.5,
        1.0,
        2.0,
        0.7,
        1.5,
        0.7,
    ]
    assert "ee_position_sq" not in terminal
    assert "ee_translation_xy" not in terminal
    assert "ee_translation_z" not in terminal
    assert "orientation" not in terminal
    assert "position_regularization" not in terminal
    assert terminal["velocity_regularization"] == 150.0
    assert terminal_terms["velocity_regularization"].params["weights"] == [
        1.0,
        5.0,
        1.4,
        4.5,
        0.8,
        5.0,
        0.8,
    ]
