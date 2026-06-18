"""Declarative OCP / cost assembly for SB-MPC.

An OCP's cost is declared as data (a YAML file or :class:`OCPConfig` dataclass):
a list of named terms + weights for the running and terminal stages. This is the
"tune by editing weights" surface — no `sbmpc` code change needed to retune or to
define a new cost. Terms are looked up in :data:`sbmpc.costs.TERM_BUILDERS` and
assembled into a :class:`~sbmpc.costs.CostModel` / :class:`~sbmpc.costs.FactoryObjective`.

YAML schema::

    name: pregrasp
    n_weights: 0            # length of the per-phase weight vector in the reference
    running_terms:
      - {name: ee_translation_xy, weight: 120.0}
      - {name: orientation, weight: 70.0, params: {x_axis_weight: 0.5}}
    terminal_terms:
      - {name: ee_position_sq, weight: 1500.0}
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from sbmpc.costs import (
    TERM_BUILDERS,
    CostModel,
    CostTerm,
    FactoryObjective,
    ReferenceLayout,
)

OCP_CONFIG_DIR = Path(__file__).resolve().parent / "ocp_configs"


@dataclass(frozen=True)
class TermSpec:
    name: str
    weight: float
    ref_weight_index: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MpcSpec:
    """MPPI / solver hyperparameters (the ``mpc:`` yaml section)."""

    horizon: int = 24
    num_samples: int = 4096          # MPC.num_parallel_computations
    num_control_points: int = 12
    dt: float = 0.02
    lambda_mpc: float = 0.05         # yaml key: ``lambda``
    std_dev_scale: float = 1.0       # std_dev_mppi = std_dev_scale * torque_limits
    smoothing: str | None = "Spline"
    initial_guess: str = "zeros"     # warm start: "zeros" | "gravity"
    gains: bool = False
    num_gain_samples: int = 512


@dataclass(frozen=True)
class SimSpec:
    """Closed-loop sandbox settings (the ``sim:`` yaml section)."""

    dt: float = 0.02
    iterations: int = 400
    integrator: str = "si_euler"


@dataclass(frozen=True)
class ReferenceSpec:
    """Reference policy for generic state/control regularization costs."""

    q_ref: str = "goal_ik"  # goal_ik | measured
    v_ref: str = "zero"     # zero | measured
    u_ref: str = "zero"     # zero | gravity_q_ref


@dataclass(frozen=True)
class OCPConfig:
    name: str
    running_terms: tuple[TermSpec, ...]
    terminal_terms: tuple[TermSpec, ...]
    n_weights: int = 0
    mpc: MpcSpec = field(default_factory=MpcSpec)
    sim: SimSpec = field(default_factory=SimSpec)
    references: ReferenceSpec = field(default_factory=ReferenceSpec)


def _term_specs(items: list[dict[str, Any]] | None) -> tuple[TermSpec, ...]:
    specs: list[TermSpec] = []
    for item in items or []:
        if item["name"] not in TERM_BUILDERS:
            valid = ", ".join(sorted(TERM_BUILDERS))
            raise ValueError(f"unknown cost term '{item['name']}'. Known terms: {valid}.")
        specs.append(
            TermSpec(
                name=str(item["name"]),
                weight=float(item["weight"]),
                ref_weight_index=item.get("ref_weight_index"),
                params=dict(item.get("params", {})),
            )
        )
    return tuple(specs)


def _normalize_smoothing(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null", "off"} else text


def _mpc_spec(d: dict[str, Any] | None) -> MpcSpec:
    d = d or {}
    return MpcSpec(
        horizon=int(d.get("horizon", MpcSpec.horizon)),
        num_samples=int(d.get("num_samples", MpcSpec.num_samples)),
        num_control_points=int(d.get("num_control_points", MpcSpec.num_control_points)),
        dt=float(d.get("dt", MpcSpec.dt)),
        lambda_mpc=float(d.get("lambda", MpcSpec.lambda_mpc)),
        std_dev_scale=float(d.get("std_dev_scale", MpcSpec.std_dev_scale)),
        smoothing=_normalize_smoothing(d.get("smoothing", MpcSpec.smoothing)),
        initial_guess=str(d.get("initial_guess", MpcSpec.initial_guess)),
        gains=bool(d.get("gains", MpcSpec.gains)),
        num_gain_samples=int(d.get("num_gain_samples", MpcSpec.num_gain_samples)),
    )


def _sim_spec(d: dict[str, Any] | None) -> SimSpec:
    d = d or {}
    return SimSpec(
        dt=float(d.get("dt", SimSpec.dt)),
        iterations=int(d.get("iterations", SimSpec.iterations)),
        integrator=str(d.get("integrator", SimSpec.integrator)),
    )


def _choice(d: dict[str, Any], key: str, default: str, valid: set[str]) -> str:
    value = str(d.get(key, default)).strip().lower()
    if value not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"references.{key} must be one of: {choices}.")
    return value


def _reference_spec(d: dict[str, Any] | None) -> ReferenceSpec:
    d = d or {}
    return ReferenceSpec(
        q_ref=_choice(d, "q_ref", ReferenceSpec.q_ref, {"goal_ik", "measured"}),
        v_ref=_choice(d, "v_ref", ReferenceSpec.v_ref, {"zero", "measured"}),
        u_ref=_choice(d, "u_ref", ReferenceSpec.u_ref, {"zero", "gravity_q_ref"}),
    )


def ocp_config_from_dict(data: dict[str, Any], *, default_name: str = "ocp") -> OCPConfig:
    return OCPConfig(
        name=str(data.get("name", default_name)),
        running_terms=_term_specs(data.get("running_terms")),
        terminal_terms=_term_specs(data.get("terminal_terms")),
        n_weights=int(data.get("n_weights", 0)),
        mpc=_mpc_spec(data.get("mpc")),
        sim=_sim_spec(data.get("sim")),
        references=_reference_spec(data.get("references")),
    )


def _resolve_source(source: str | Path) -> Path:
    path = Path(source)
    if path.suffix in (".yaml", ".yml") and path.exists():
        return path
    named = OCP_CONFIG_DIR / f"{source}.yaml"
    if named.exists():
        return named
    if path.exists():
        return path
    raise FileNotFoundError(
        f"OCP config '{source}' not found (looked for {named} and {path})."
    )


def load_ocp_config(source: str | Path) -> OCPConfig:
    import yaml

    path = _resolve_source(source)
    data = yaml.safe_load(path.read_text()) or {}
    return ocp_config_from_dict(data, default_name=path.stem)


def with_weight_overrides(ocp: OCPConfig, overrides: dict[str, float]) -> OCPConfig:
    """Return a copy with running/terminal term weights overridden by term name.

    The tuning hook used by the ROS bridge and the offline harness. Unknown names
    are ignored so a generic override map can target any OCP.
    """
    if not overrides:
        return ocp

    def patched(specs: tuple[TermSpec, ...]) -> tuple[TermSpec, ...]:
        return tuple(
            replace(spec, weight=float(overrides[spec.name]))
            if spec.name in overrides
            else spec
            for spec in specs
        )

    return replace(
        ocp,
        running_terms=patched(ocp.running_terms),
        terminal_terms=patched(ocp.terminal_terms),
    )


def build_cost_model(ocp: OCPConfig, planner: Any) -> CostModel:
    layout = ReferenceLayout(nq=planner.nq, nv=planner.nv, n_weights=ocp.n_weights)

    def build(specs: tuple[TermSpec, ...]) -> list[CostTerm]:
        terms: list[CostTerm] = []
        for spec in specs:
            fn = TERM_BUILDERS[spec.name](planner, **spec.params)
            terms.append(
                CostTerm(
                    name=spec.name,
                    weight=spec.weight,
                    fn=fn,
                    ref_weight_index=spec.ref_weight_index,
                )
            )
        return terms

    return CostModel(build(ocp.running_terms), build(ocp.terminal_terms), layout)


def build_objective(ocp: OCPConfig, planner: Any) -> FactoryObjective:
    return FactoryObjective(build_cost_model(ocp, planner))
