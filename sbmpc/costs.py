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
    """Decoded reference for one rollout step (frame goals + optional weights)."""

    goal_pos: jax.Array
    goal_q: jax.Array
    goal_x_axis: jax.Array
    goal_z_axis: jax.Array
    goal_tau: jax.Array
    weights: jax.Array | None  # per-phase weight vector, or None when not carried


@dataclass(frozen=True)
class ReferenceLayout:
    """Slices the flat reference vector into a :class:`ReferenceContext`.

    Matches the existing Franka packing: ``[goal_pos(3), goal_q(nq), goal_x(3),
    goal_z(3), goal_tau(nv), (optional) weights(n_weights)]``.
    """

    nq: int
    nv: int
    n_weights: int = 0

    def context(self, reference: jax.Array) -> ReferenceContext:
        nq, nv = self.nq, self.nv
        goal_pos = reference[:3]
        goal_q = reference[3 : 3 + nq]
        goal_x = reference[3 + nq : 6 + nq]
        goal_z = reference[6 + nq : 9 + nq]
        goal_tau = reference[9 + nq : 9 + nq + nv]
        weights = None
        if self.n_weights > 0:
            start = 9 + nq + nv
            weights = reference[start : start + self.n_weights]
        return ReferenceContext(
            goal_pos=goal_pos,
            goal_q=goal_q,
            goal_x_axis=goal_x,
            goal_z_axis=goal_z,
            goal_tau=goal_tau,
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
        return smooth_norm((ctx.goal_pos - ee_pos)[:2])

    return fn


def _ee_translation_z(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        ee_pos, _, _ = planner.ee_features(q)
        return jnp.abs((ctx.goal_pos - ee_pos)[2])

    return fn


def _ee_position_sq(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        ee_pos, _, _ = planner.ee_features(q)
        return jnp.sum(jnp.square(ctx.goal_pos - ee_pos))

    return fn


def _orientation(planner, *, x_axis_weight: float = 0.5, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        _, ee_x, ee_z = planner.ee_features(q)
        return axis_alignment_cost(ee_z, ctx.goal_z_axis) + x_axis_weight * axis_alignment_cost(
            ee_x, ctx.goal_x_axis
        )

    return fn


def _posture(planner, **_):
    def fn(state, inputs, ctx):
        q = state[: planner.nq]
        return jnp.sum(jnp.square(q - ctx.goal_q))

    return fn


def _control_regularization(planner, **_):
    torque_scale = jnp.maximum(planner.torque_limits, 1.0)

    def fn(state, inputs, ctx):
        return jnp.sum(jnp.square((inputs - ctx.goal_tau) / torque_scale))

    return fn


def _joint_velocity(planner, **_):
    nq = planner.nq

    def fn(state, inputs, ctx):
        v = state[nq:]
        return jnp.sum(jnp.square(v))

    return fn


def _mechanical_power(planner, **_):
    """Penalize joint mechanical power tau*omega (opt-in; default weight 0)."""
    nq = planner.nq

    def fn(state, inputs, ctx):
        v = state[nq:]
        return jnp.sum(jnp.square(inputs * v))

    return fn


TERM_BUILDERS: dict[str, Callable[..., TermFn]] = {
    "ee_translation_xy": _ee_translation_xy,
    "ee_translation_z": _ee_translation_z,
    "ee_position_sq": _ee_position_sq,
    "orientation": _orientation,
    "posture": _posture,
    "control_regularization": _control_regularization,
    "joint_velocity": _joint_velocity,
    "mechanical_power": _mechanical_power,
}
