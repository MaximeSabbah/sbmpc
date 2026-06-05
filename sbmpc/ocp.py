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
class OCPConfig:
    name: str
    running_terms: tuple[TermSpec, ...]
    terminal_terms: tuple[TermSpec, ...]
    n_weights: int = 0
    # Velocity-pace the receding-horizon warm-start seed at this fraction of the
    # joint velocity limit (None/<=0 = reach within the horizon, the original
    # behavior). This is the dominant lever on reach aggressiveness — cost weights
    # barely move the feedforward because MPPI only perturbs the seed by ~std_dev.
    seed_pace_velocity_fraction: float | None = None


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


def ocp_config_from_dict(data: dict[str, Any], *, default_name: str = "ocp") -> OCPConfig:
    pace = data.get("seed_pace_velocity_fraction")
    return OCPConfig(
        name=str(data.get("name", default_name)),
        running_terms=_term_specs(data.get("running_terms")),
        terminal_terms=_term_specs(data.get("terminal_terms")),
        n_weights=int(data.get("n_weights", 0)),
        seed_pace_velocity_fraction=None if pace is None else float(pace),
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
