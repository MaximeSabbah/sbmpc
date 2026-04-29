"""Chunk C compute-reduction sweep.

Runs a series of short bench_lfc validation runs (timing_mode=immediate) to
quantify how MJX solver options and the buffered-gain knobs trade off compute
latency against controller stability/accuracy. The goal is to find an
operating point whose p99 planning time fits within the 20 ms / 50 Hz budget
while keeping end-effector error, gain norm, and joint velocity within the
Chunk B reference bounds.

Output:
  tests/reference/sweep_compute_<YYYYMMDD>.json     — full per-config metrics
  tests/reference/operating_point.md                — chosen winner + rationale

The sweep has TWO axes, evaluated in order. All configs share horizon=8,
samples=1024, cp=8, gain_method=exact, steps=30, dt=0.02.

  AXIS 1 — MJX solver options (pick fastest that meets accuracy+stability gate)
  AXIS 2 — Buffered-gain knobs  (at axis-1 winner)

Gate per config (matches Chunk B reference):
  error_final          <= 1.5 mm
  gain_norm_final      <= 5.0
  feedback_peak_max    <  50 Nm
  joint_velocity_abs_max <  3 rad/s
  joint_vel_hf_energy  <= 0.13   (1.1 * reference 0.117)
  plan_ms_p99          <  20 ms  (stretch / 50 Hz budget)

Usage:
  PYTHONPATH=/home/msabbah/Desktop/sbmpc \\
    .pixi/envs/cuda/bin/python tests/bench_sweep_compute.py [--steps 30]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import jax
import numpy as np

from tests.bench_lfc import (
    _build_lfc_sim,
    _compile_controller,
    _joint_vel_hf_energy,
    _lfc_step,
)

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")

# Accuracy/stability gates — any config that fails these is considered
# infeasible and cannot be picked as a sweep winner even if it's fast.
GATE_ERROR_FINAL_M = 1.5e-3
GATE_GAIN_NORM_FINAL = 5.0
GATE_FEEDBACK_PEAK_NM = 50.0
GATE_JOINT_VEL_MAX = 3.0
GATE_HF_ENERGY = 0.13
GATE_PLAN_MS_P99 = 20.0  # stretch: 50 Hz budget


@dataclass
class SweepConfig:
    label: str
    mjx_opts: dict | None = None
    gain_samples_per_cycle: int | None = None
    gain_buffer_size: int | None = None
    extra: dict = field(default_factory=dict)


def _make_args(steps: int, cfg: SweepConfig) -> argparse.Namespace:
    return argparse.Namespace(
        visual=False,
        realtime=False,
        print_every=1,
        steps=steps,
        substeps=20,
        timing_mode="immediate",
        retime_initial_state=True,
        desired_state_mode="base",
        clip_torque=False,
        clip_velocity=False,
        dt=0.02,
        horizon=8,
        samples=1024,
        control_points=8,
        gain_samples_per_cycle=cfg.gain_samples_per_cycle,
        gain_buffer_size=cfg.gain_buffer_size,
        mjx_opts=cfg.mjx_opts,
        emit_reference=None,
    )


def _run_one(cfg: SweepConfig, steps: int) -> dict:
    """Run one sweep config. Captures summary metrics and wall-clock timing."""
    args = _make_args(steps, cfg)

    t_build_start = time.perf_counter()
    planner, config, sim = _build_lfc_sim(args)
    _compile_controller(sim, planner, config)
    t_build_ms = (time.perf_counter() - t_build_start) * 1000.0

    errors: list[float] = []
    gain_norms: list[float] = []
    feedback_peaks: list[float] = []
    plan_times_ms: list[float] = []
    q_hist: list[np.ndarray] = []
    v_hist: list[np.ndarray] = []
    previous_control: dict | None = None
    previous_planning_ms = 0.0
    diverged_at: int | None = None

    for step_idx in range(steps):
        metrics, previous_control = _lfc_step(
            sim, planner, config, args, previous_control, previous_planning_ms,
        )
        previous_planning_ms = metrics["planning_ms"]
        state_np = metrics["state"]
        if not np.all(np.isfinite(state_np)):
            diverged_at = step_idx
            break
        errors.append(metrics["error"])
        gain_norms.append(metrics["gain_norm"])
        feedback_peaks.append(metrics["feedback_peak"])
        plan_times_ms.append(metrics["planning_ms"])
        q_hist.append(state_np[: planner.nq])
        v_hist.append(state_np[planner.nq : planner.nq + planner.nv])

    if not errors:
        return {
            "label": cfg.label,
            "mjx_opts": cfg.mjx_opts,
            "gain_samples_per_cycle": cfg.gain_samples_per_cycle,
            "gain_buffer_size": cfg.gain_buffer_size,
            "diverged_at": diverged_at,
            "build_ms": t_build_ms,
            "summary": None,
        }

    errors_arr = np.asarray(errors)
    plan_arr = np.asarray(plan_times_ms)
    v_arr = np.asarray(v_hist)
    tail_start = len(errors) // 2
    hf = _joint_vel_hf_energy(v_arr[tail_start:], args.dt)

    summary = {
        "error_initial": float(errors_arr[0]),
        "error_final": float(errors_arr[-1]),
        "error_min": float(np.min(errors_arr)),
        "tail_error_mean": float(np.mean(errors_arr[tail_start:])),
        "plan_ms_p50": float(np.percentile(plan_arr, 50)),
        "plan_ms_p95": float(np.percentile(plan_arr, 95)),
        "plan_ms_p99": float(np.percentile(plan_arr, 99)),
        "gain_norm_final": float(np.asarray(gain_norms)[-1]),
        "gain_norm_max": float(np.max(gain_norms)),
        "feedback_peak_max": float(np.max(feedback_peaks)),
        "joint_velocity_abs_max": float(np.max(np.abs(v_arr))),
        "joint_vel_hf_energy": float(hf),
    }
    return {
        "label": cfg.label,
        "mjx_opts": cfg.mjx_opts,
        "gain_samples_per_cycle": cfg.gain_samples_per_cycle,
        "gain_buffer_size": cfg.gain_buffer_size,
        "diverged_at": diverged_at,
        "build_ms": t_build_ms,
        "summary": summary,
    }


def _feasible(result: dict) -> tuple[bool, list[str]]:
    """Return (feasible, failing_gate_names). Ignores compute gate."""
    s = result.get("summary")
    if s is None:
        return False, ["diverged"]
    fails = []
    if s["error_final"] > GATE_ERROR_FINAL_M:
        fails.append(f"error_final={s['error_final']*1000:.2f}mm")
    if s["gain_norm_final"] > GATE_GAIN_NORM_FINAL:
        fails.append(f"|K|_f={s['gain_norm_final']:.2f}")
    if s["feedback_peak_max"] >= GATE_FEEDBACK_PEAK_NM:
        fails.append(f"fb_peak={s['feedback_peak_max']:.0f}Nm")
    if s["joint_velocity_abs_max"] >= GATE_JOINT_VEL_MAX:
        fails.append(f"v_max={s['joint_velocity_abs_max']:.2f}")
    if s["joint_vel_hf_energy"] > GATE_HF_ENERGY:
        fails.append(f"hf={s['joint_vel_hf_energy']:.3f}")
    return (len(fails) == 0), fails


def _print_row(r: dict) -> None:
    s = r.get("summary") or {}
    if not s:
        print(f"  {r['label']:42s}  DIVERGED at step {r.get('diverged_at')}")
        return
    feasible, fails = _feasible(r)
    mark = "✓" if feasible else "✗"
    over_budget = "" if s["plan_ms_p99"] < GATE_PLAN_MS_P99 else " [>20ms]"
    fail_str = f"  fails: {', '.join(fails)}" if fails else ""
    print(
        f"  {mark} {r['label']:42s}  "
        f"err_f={s['error_final']*1000:6.2f}mm  "
        f"p99={s['plan_ms_p99']:6.1f}ms{over_budget}  "
        f"|K|_f={s['gain_norm_final']:5.2f}  "
        f"fb={s['feedback_peak_max']:5.1f}Nm"
        f"{fail_str}"
    )


def _pick_winner(results: list[dict]) -> dict | None:
    feasible = [r for r in results if _feasible(r)[0]]
    if not feasible:
        return None
    # Among feasible, prefer minimum plan_ms_p99.
    return min(feasible, key=lambda r: r["summary"]["plan_ms_p99"])


def _axis1_configs() -> list[SweepConfig]:
    # MJX knobs. (None, None, None) = baseline; then progressively aggressive.
    opts = [
        None,
        {"iterations": 4, "ls_iterations": 1},
        {"iterations": 2, "ls_iterations": 1},
        {"iterations": 2, "ls_iterations": 1, "tolerance": 1e-4},
        {"iterations": 1, "ls_iterations": 1, "tolerance": 1e-3},
    ]
    return [
        SweepConfig(label=f"mjx={o}" if o else "mjx=baseline", mjx_opts=o)
        for o in opts
    ]


def _axis2_configs(winner_mjx: dict | None) -> list[SweepConfig]:
    # Buffered-gain knobs at axis-1 winner. (None, None) = baseline (one-shot).
    combos: list[tuple[int | None, int | None]] = [
        (None, None),
        (256, 1024),
        (128, 512),
        (64, 512),
        (64, 256),
        (32, 256),
    ]
    return [
        SweepConfig(
            label=f"gK={k}, gM={m}",
            mjx_opts=winner_mjx,
            gain_samples_per_cycle=k,
            gain_buffer_size=m,
        )
        for (k, m) in combos
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30,
                        help="Validation steps per config. Short is fine — we need summary, not long-horizon convergence.")
    parser.add_argument("--output-dir", type=str, default=REFERENCE_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}\n")

    axis1 = _axis1_configs()
    axis2_pending = True

    all_results = {"axis1": [], "axis2": []}

    print(f"=== AXIS 1 — MJX solver options ({len(axis1)} configs, {args.steps} steps each) ===")
    for cfg in axis1:
        print(f"[{cfg.label}]")
        r = _run_one(cfg, args.steps)
        _print_row(r)
        all_results["axis1"].append(r)

    winner1 = _pick_winner(all_results["axis1"])
    if winner1 is None:
        print("\nAxis 1: NO feasible config. Falling back to baseline MJX for axis 2.")
        winner1_mjx = None
    else:
        winner1_mjx = winner1["mjx_opts"]
        print(f"\nAxis 1 winner: {winner1['label']}  (p99={winner1['summary']['plan_ms_p99']:.1f}ms)")

    print(f"\n=== AXIS 2 — Buffered gain at axis-1 winner ({args.steps} steps each) ===")
    axis2 = _axis2_configs(winner1_mjx)
    for cfg in axis2:
        print(f"[{cfg.label}]")
        r = _run_one(cfg, args.steps)
        _print_row(r)
        all_results["axis2"].append(r)

    winner = _pick_winner(all_results["axis1"] + all_results["axis2"])

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "steps": args.steps,
        "gates": {
            "error_final_m": GATE_ERROR_FINAL_M,
            "gain_norm_final": GATE_GAIN_NORM_FINAL,
            "feedback_peak_nm": GATE_FEEDBACK_PEAK_NM,
            "joint_velocity_max_rad_s": GATE_JOINT_VEL_MAX,
            "hf_energy": GATE_HF_ENERGY,
            "plan_ms_p99_stretch": GATE_PLAN_MS_P99,
        },
        "axis1": all_results["axis1"],
        "axis2": all_results["axis2"],
        "winner": winner,
    }

    day = datetime.now().strftime("%Y%m%d")
    out_json = os.path.join(args.output_dir, f"sweep_compute_{day}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote sweep detail -> {out_json}")

    if winner:
        print(f"\n=== WINNER ===")
        _print_row(winner)
        op_md = os.path.join(args.output_dir, "operating_point.md")
        with open(op_md, "w") as f:
            f.write(f"# Chunk C operating point (selected {datetime.now().isoformat(timespec='seconds')})\n\n")
            f.write(f"- **label**: {winner['label']}\n")
            f.write(f"- **mjx_opts**: {winner['mjx_opts']}\n")
            f.write(f"- **gain_samples_per_cycle**: {winner['gain_samples_per_cycle']}\n")
            f.write(f"- **gain_buffer_size**: {winner['gain_buffer_size']}\n\n")
            f.write("## Summary metrics\n\n")
            for k, v in winner["summary"].items():
                f.write(f"- {k}: {v}\n")
        print(f"wrote operating point -> {op_md}")
    else:
        print("\n=== NO FEASIBLE WINNER across both axes ===")
        print("Inspect the sweep detail JSON for per-config failures.")


if __name__ == "__main__":
    main()
