"""Composable cost-term library and weighted-sum cost model for SB-MPC.

Agimus/Crocoddyl-inspired: an OCP's running and terminal costs are assembled as a
weighted sum of small, reusable, JAX-differentiable terms instead of a hand-rolled
``running_cost``. Terms must stay differentiable because the exact-gain path takes a
jvp of the rollout cost w.r.t. the initial state.

Terms are built by name from a registry (:data:`TERM_BUILDERS`), each bound to a
``planner`` that provides the kinematics (``ee_features``), dof counts (``nq``/``nv``)
and ``torque_limits``. See :mod:`sbmpc.ocp` for the declarative (YAML/dataclass)
assembly layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from sbmpc.solvers import BaseObjective


# A term contribution: (state, inputs, ctx) -> scalar. ``inputs`` is None for
# terminal terms; terms that need the control (e.g. control_regularization) are
# running-only.
TermFn = Callable[[jax.Array, jax.Array | None, "ReferenceContext"], jax.Array]


# --- small numerics shared with the original hand-rolled objectives ---
def smooth_norm(vec: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(jnp.square(vec)) + 1e-8)


def axis_alignment_cost(axis: jax.Array, target_axis: jax.Array) -> jax.Array:
    return 1.0 - jnp.clip(jnp.dot(axis, target_axis), -1.0, 1.0)


@dataclass(frozen=True)
class ReferenceContext:
    """Decoded cost reference for one rollout step.

    Cost terms do not decide where references come from. The planner builds this
    vector from task targets, measured state, nominal trajectories, or any blend
    that is appropriate for the controller.
    """

    ee_pos_ref: jax.Array
    q_ref: jax.Array
    ee_x_axis_ref: jax.Array
    ee_z_axis_ref: jax.Array
    u_ref: jax.Array
    v_ref: jax.Array
    weights: jax.Array | None  # per-phase weight vector, or None when not carried


@dataclass(frozen=True)
class ReferenceLayout:
    """Slices the flat reference vector into a :class:`ReferenceContext`.

    Packing:
    ``[ee_pos_ref(3), q_ref(nq), ee_x_ref(3), ee_z_ref(3), u_ref(nv),
    v_ref(nv), optional weights(n_weights)]``.
    """

    nq: int
    nv: int
    n_weights: int = 0

    def context(self, reference: jax.Array) -> ReferenceContext:
        nq, nv = self.nq, self.nv
        ee_pos_ref = reference[:3]
        q_ref = reference[3 : 3 + nq]
        ee_x_axis_ref = reference[3 + nq : 6 + nq]
        ee_z_axis_ref = reference[6 + nq : 9 + nq]
        u_ref = reference[9 + nq : 9 + nq + nv]
        v_ref = reference[9 + nq + nv : 9 + nq + 2 * nv]
        weights = None
        if self.n_weights > 0:
            start = 9 + nq + 2 * nv
            weights = reference[start : start + self.n_weights]
        return ReferenceContext(
            ee_pos_ref=ee_pos_ref,
            q_ref=q_ref,
            ee_x_axis_ref=ee_x_axis_ref,
            ee_z_axis_ref=ee_z_axis_ref,
            u_ref=u_ref,
            v_ref=v_ref,
            weights=weights,
        )


@dataclass(frozen=True)
class CostTerm:
    """A named, weighted term. Effective weight = ``weight`` * (optional reference
    weight at ``ref_weight_index``)."""

    name: str
    weight: float
    fn: TermFn
    ref_weight_index: int | None = None

    def contribution(
        self,
        state: jax.Array,
        inputs: jax.Array | None,
        ctx: ReferenceContext,
    ) -> jax.Array:
        weight = jnp.asarray(self.weight, dtype=jnp.float32)
        if self.ref_weight_index is not None and ctx.weights is not None:
            weight = weight * ctx.weights[self.ref_weight_index]
        return weight * self.fn(state, inputs, ctx)


class CostModel:
    """Weighted sum of cost terms for running and terminal stages."""

    def __init__(
        self,
        running_terms: list[CostTerm],
        terminal_terms: list[CostTerm],
        layout: ReferenceLayout,
    ) -> None:
        self.running_terms = list(running_terms)
        self.terminal_terms = list(terminal_terms)
        self.layout = layout

    def running(
        self, state: jax.Array, inputs: jax.Array, reference: jax.Array
    ) -> jax.Array:
        ctx = self.layout.context(reference)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for term in self.running_terms:
            total = total + term.contribution(state, inputs, ctx)
        return total.astype(jnp.float32)

    def terminal(self, state: jax.Array, reference: jax.Array) -> jax.Array:
        ctx = self.layout.context(reference)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for term in self.terminal_terms:
            total = total + term.contribution(state, None, ctx)
        return total.astype(jnp.float32)


class FactoryObjective(BaseObjective):
    """``BaseObjective`` backed by a :class:`CostModel`."""

    def __init__(self, cost_model: CostModel):
        super().__init__()
        self.cost_model = cost_model

    def running_cost(self, state, inputs, reference):
        return self.cost_model.running(state, inputs, reference)

    def final_cost(self, state, reference):
        return self.cost_model.terminal(state, reference)


# --- term builders (bound to a planner) -------------------------------------
def _ee_translation_xy(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        ee_pos, _, _ = planner.ee_features(q)
        return smooth_norm((ctx.ee_pos_ref - ee_pos)[:2])

    return fn


def _ee_translation_z(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        ee_pos, _, _ = planner.ee_features(q)
        return jnp.abs((ctx.ee_pos_ref - ee_pos)[2])

    return fn


def _ee_position_sq(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        ee_pos, _, _ = planner.ee_features(q)
        return jnp.sum(jnp.square(ctx.ee_pos_ref - ee_pos))

    return fn


def _orientation(planner, *, x_axis_weight: float = 0.5, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        _, ee_x, ee_z = planner.ee_features(q)
        return axis_alignment_cost(
            ee_z, ctx.ee_z_axis_ref
        ) + x_axis_weight * axis_alignment_cost(
            ee_x,
            ctx.ee_x_axis_ref,
        )

    return fn


def _position_regularization(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        return jnp.sum(jnp.square(q - ctx.q_ref))

    return fn


def _joint_weights(planner, weights, *, name: str):
    if weights is None:
        return jnp.ones(planner.nv, dtype=jnp.float32)
    weights_arr = jnp.asarray(weights, dtype=jnp.float32)
    if weights_arr.shape != (planner.nv,):
        raise ValueError(
            f"{name} weights must contain one value per joint "
            f"({planner.nv}), got shape {weights_arr.shape}."
        )
    return weights_arr


def _control_regularization(planner, *, weights=None, **_):
    joint_weights = _joint_weights(planner, weights, name="control_regularization")

    def fn(state, inputs, ctx):
        return jnp.sum(joint_weights * jnp.square(inputs - ctx.u_ref))

    return fn


def _velocity_regularization(planner, *, weights=None, **_):
    nq = planner.nq
    joint_weights = _joint_weights(planner, weights, name="velocity_regularization")

    def fn(state, inputs, ctx):
        v = state[nq:]
        return jnp.sum(joint_weights * jnp.square(v - ctx.v_ref))

    return fn


def _joint_acceleration(planner, *, weights=None, **_):
    """Penalize predicted joint acceleration ``qdd`` from the rollout dynamics."""
    dynamics = getattr(planner, "dynamics", None)
    if not callable(dynamics):
        raise ValueError("joint_acceleration cost requires planner.dynamics")

    nq = planner.nq
    nv = planner.nv
    joint_weights = _joint_weights(planner, weights, name="joint_acceleration")

    def fn(state, inputs, ctx):
        del ctx
        if inputs is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        xdot = dynamics(
            state,
            inputs,
            jnp.zeros(0, dtype=jnp.asarray(state).dtype),
        )
        qdd = xdot[nq : nq + nv]
        return jnp.sum(joint_weights * jnp.square(qdd))

    return fn


def _mechanical_power(planner, *, weights=None, **_):
    """Penalize joint mechanical power tau*omega (opt-in; default weight 0)."""
    nq = planner.nq
    joint_weights = _joint_weights(planner, weights, name="mechanical_power")

    def fn(state, inputs, ctx):
        v = state[nq:]
        return jnp.sum(joint_weights * jnp.square(inputs * v))

    return fn


TERM_BUILDERS: dict[str, Callable[..., TermFn]] = {
    "ee_translation_xy": _ee_translation_xy,
    "ee_translation_z": _ee_translation_z,
    "ee_position_sq": _ee_position_sq,
    "orientation": _orientation,
    "position_regularization": _position_regularization,
    "control_regularization": _control_regularization,
    "velocity_regularization": _velocity_regularization,
    "joint_acceleration": _joint_acceleration,
    "mechanical_power": _mechanical_power,
}
