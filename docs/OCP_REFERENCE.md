# OCP Reference — declaring a controller as a YAML file

An **OCP yaml** is the single source of truth for one SB-MPC controller: the
cost function (running + terminal terms), the MPPI/solver knobs, and the
closed-loop sandbox settings. Retuning or defining a new task variant means
editing a yaml — no `sbmpc` code change. The ROS bridge (`sbmpc_ros`) loads
the **same file** through its `planner_ocp` parameter, so simulation,
benchmarks, and deployment always run the controller you declared here.

This document is the authoritative schema + cost-term catalog. It is written
to be precise enough for automated authoring (scripts, LLM/VLM agents): every
field, constraint, and failure mode is stated, and the implementation pointers
let a tool re-derive the ground truth.

Implementation pointers (ground truth):

| What | Where |
|---|---|
| YAML loading / schema dataclasses | `sbmpc/ocp.py` (`OCPConfig`, `MpcSpec`, `SimSpec`, `load_ocp_config`) |
| Cost-term registry + formulas | `sbmpc/costs.py` (`TERM_BUILDERS`) |
| Reference vector layout | `sbmpc/costs.py` (`ReferenceLayout`) |
| Existing OCPs | `sbmpc/ocp_configs/pregrasp.yaml`, `sbmpc/ocp_configs/pick_and_place.yaml` |
| Consumers | `scripts/panda_pregrasp.py --ocp`, `tests/bench_controller.py`, ROS `planner_ocp` parameter |

---

## 1. File location and loading

- Named OCPs live in `sbmpc/sbmpc/ocp_configs/<name>.yaml` and are loaded with
  `load_ocp_config("<name>")`. An explicit `.yaml` path also works.
- Unknown cost-term names raise `ValueError` listing the valid names, at load
  time — a malformed yaml fails fast, before any JAX compilation.
- Validate a yaml without running anything:

```bash
pixi run -e cuda python -c "from sbmpc.ocp import load_ocp_config; print(load_ocp_config('pregrasp'))"
```

Consumers:

```bash
# Closed-loop sandbox with viewer + PASS/FAIL validation (task error, robot
# limits, gain health, 25 Hz timing):
pixi run -e cuda python scripts/panda_pregrasp.py --ocp <name> [--headless]

# Timing benchmark (blocked GPU calls, p50/p95 vs the 40 ms budget):
pixi run -e cuda python tests/bench_controller.py --gains [--gain-samples N]
```

```yaml
# sbmpc_ros: sbmpc_bringup/config/sbmpc_bridge.yaml
planner_ocp: <name>     # the bridge defers ALL mpc knobs to this OCP yaml
```

The bridge's `planner_*` ROS parameters override individual knobs only when
explicitly set; leaving them unset (the shipped presets do) keeps the OCP yaml
authoritative. `test_bringup_config.py::MPPI_KNOBS_OWNED_BY_OCP_YAML` guards
this, and the bridge warns if `mpc.dt` disagrees with its publish rate.

## 2. Top-level schema

```yaml
name: <string>            # OCP identifier (defaults to the file stem)
n_weights: <int>          # length of the per-phase weight vector carried in the
                          # reference; 0 = no per-phase weights (see §5)

mpc:                      # MPPI / solver knobs (defaults in MpcSpec)
  horizon: 10             # rollout steps
  num_samples: 1024       # parallel MPPI rollouts (= num_parallel_computations)
  num_gain_samples: 128   # top-K lowest-cost samples differentiated for the
                          # F-MPPI feedback gains; must be <= num_samples
  num_control_points: 10  # spline control points; must be <= horizon
  dt: 0.04                # rollout step [s]; ALSO the control period the
                          # deployment must match (0.04 s -> 25 Hz bridge rate)
  lambda: 0.05            # MPPI temperature (yaml key is `lambda`,
                          # field name lambda_mpc); higher = greedier
  std_dev_scale: 0.1      # sampling std dev = std_dev_scale * torque_limits
  smoothing: Spline       # Spline | null  (null = sample raw torques per step)
  initial_guess: gravity  # zeros | gravity  (gravity = hold against gravity at
                          # the home pose; NOT a trajectory to the goal)
  gains: true             # compute F-MPPI feedback gains in the same cycle

sim:                      # closed-loop sandbox only (scripts/panda_pregrasp.py)
  dt: 0.04                # sim integration step [s]
  iterations: 400         # closed-loop steps
  integrator: custom_discrete   # si_euler | euler | rk4 | custom_discrete

references:               # generic state/control refs used by regularization terms
  q_ref: measured          # measured | goal_ik
  v_ref: zero              # zero | measured
  u_ref: gravity_q_ref     # gravity_q_ref | zero
  u_prev_ref: previous_control  # previous_control | u_ref | zero

running_terms:            # weighted sum, integrated as dt * sum(...) per step
  - {name: <term>, weight: <float>, params: {...}, ref_weight_index: <int>}
terminal_terms:           # added once at the end of the horizon
  - {name: <term>, weight: <float>, params: {...}, ref_weight_index: <int>}
```

Constraints enforced at build time:

- `num_control_points <= horizon` (raises otherwise).
- `num_gain_samples <= num_samples` (raises otherwise).
- Every term `name` must exist in `TERM_BUILDERS`.
- `references.q_ref` must be `measured` or `goal_ik`.
- `references.v_ref` must be `zero` or `measured`.
- `references.u_ref` must be `gravity_q_ref` or `zero`; `gravity_q_ref`
  means gravity compensation evaluated at the selected `q_ref`.
- `references.u_prev_ref` must be `previous_control`, `u_ref`, or `zero`.
  It initializes the first step of `command_rate_regularization`; after that,
  the term compares each planned torque with the preceding planned torque.
- All terms must stay JAX-differentiable: the exact-gain path takes a jvp of
  the rollout cost w.r.t. the initial state.

Timing budget (measured on the lab RTX 5000 Ada, h=10, n=1024, fully blocked
`command()` + gains): `num_gain_samples` 128 → p95 ≈ 34 ms (passes the 40 ms /
25 Hz budget); 256 → ≈ 40 ms (fails); 512 → ≈ 43 ms (fails). If you raise the
top-K count, re-run `bench_controller.py` and lower the control rate or shrink
the problem accordingly.

## 3. Cost-term catalog

All terms receive `(state, inputs, ctx)` where `state = [q (nq), v (nv)]`,
`inputs` is the joint-torque sample (running terms only; `None` at the terminal
stage), and `ctx` is the decoded reference (§5). Contribution =
`weight * [ctx.weights[ref_weight_index] if set] * fn(...)`.

| name | formula | reads from reference | params | notes |
|---|---|---|---|---|
| `ee_translation_xy` | `sqrt(‖(ee_pos_ref − ee_pos)[:2]‖² + 1e-8)` | `ee_pos_ref` | — | smooth L2 in the horizontal plane; linear far from reference (steady pull) |
| `ee_translation_z` | `abs((ee_pos_ref − ee_pos)[2])` | `ee_pos_ref` | — | vertical reach component, kept separate so descend/lift phases can weight it differently |
| `ee_position_sq` | `‖ee_pos_ref − ee_pos‖²` | `ee_pos_ref` | — | quadratic; sharp near the reference — typical as a **terminal** term |
| `orientation` | `(1 − ee_z·ee_z_axis_ref) + x_axis_weight·(1 − ee_x·ee_x_axis_ref)` | `ee_x_axis_ref`, `ee_z_axis_ref` | `x_axis_weight` (default 0.5) | axis alignment of the TCP frame; z axis is the tool axis |
| `position_regularization` | `‖q − q_ref‖²` | `q_ref` | — | joint-position regularization; the planner decides whether `q_ref` is the measured state, IK goal, nominal trajectory sample, etc. |
| `control_regularization` | `Σ w_i(τ_i − u_ref_i)²` | `u_ref` | `weights` (optional per joint) | **running-only**; by default pregrasp sets `u_ref = gravity(q_measured_now)` |
| `command_rate_regularization` | `Σ w_i(τ_k,i − τ_{k−1,i})²` | `u_prev_ref` for `k=0` | `weights` (optional per joint) | **running-only**; penalizes command jumps. For `k>0`, `τ_{k−1}` is the previous planned torque inside the same horizon. |
| `velocity_regularization` | `Σ w_i(v_i − v_ref_i)²` | `v_ref` | `weights` (optional per joint) | damps motion; pregrasp uses zero velocity as the reference |
| `mechanical_power` | `‖τ ⊙ v‖²` | — | — | joint mechanical power penalty; opt-in (not used by the shipped OCPs), useful against power-limit violations on the real robot |

End-effector kinematics (`ee_pos`, `ee_x`, `ee_z`) come from the planner's
differentiable `ee_features(q)` (MJX forward kinematics of the gripper site).

To **add a new term**: write a builder `def _my_term(planner, **params) -> fn`
in `sbmpc/costs.py`, register it in `TERM_BUILDERS`, keep it differentiable.
It is then immediately usable from any yaml.

## 4. Running vs terminal stages

- Running cost is accumulated as `cost += dt * Σ term(state_k, τ_k, ref)` along
  the horizon (so weights are per-second; halving `mpc.dt` does not change the
  effective running weight).
- Terminal cost is added once on the final state: `cost += Σ term(state_N, ref)`.
  Terminal terms never see `inputs` — `control_regularization` is invalid there.
- Tuning intuition from the pregrasp task: strong terminal `ee_position_sq`
  (1500) is what actually pins the goal; running translation terms (90–120)
  shape the approach; `velocity_regularization` and `control_regularization` are the
  brakes. If the controller saturates torque/velocity limits during the reach,
  raise the brakes or soften the terminal cost before touching the hardware.

## 5. The reference vector (what terms can "see")

The planner packs a flat reference vector consumed by `ReferenceLayout`:

```
[ ee_pos_ref(3) | q_ref(nq) | ee_x_axis_ref(3) | ee_z_axis_ref(3) |
  u_ref(nv) | u_prev_ref(nv) | v_ref(nv) | weights(n_weights, optional) ]
```

- `ee_pos_ref`: target TCP position.
- `ee_x_axis_ref`/`ee_z_axis_ref`: columns of the reference TCP rotation.
- `q_ref`, `v_ref`, `u_ref`: generic state/control regularization references.
  Cost terms do not prescribe where these values come from; the OCP
  `references:` policy does. In the deployed pregrasp configuration they are
  refreshed at every MPC cycle as `q_measured`, zero velocity, and
  `gravity(q_measured)`.
- `u_prev_ref`: first-step command-rate reference. In deployed pregrasp this is
  the previous `tau_ff` published by the planner, falling back to `u_ref` on the
  first active cycle.
- `weights` (only when `n_weights > 0`): a per-phase scaling vector. A term
  with `ref_weight_index: i` gets its yaml weight multiplied by `weights[i]`.
  This is how one yaml serves a multi-phase task: the planner switches the
  weight vector per phase while the term structure stays fixed.
  `pick_and_place.yaml` uses `n_weights: 6` with the convention
  `0: ee, 1: orientation, 2: position, 3: control, 4: velocity, 5: final_position`.

## 6. Tasks available today

| task | yaml | planner / controller | goal definition |
|---|---|---|---|
| **Pregrasp reach** (deployed) | `pregrasp.yaml` | `PandaPregraspPlanner` / `PandaPregraspController` (`sbmpc/controller/franka_emika_panda/`) | hover above the scene object: `object_pos + [0, 0, half_height + 0.05]`, top-down orientation; or pass `goal_pos=` to the planner |
| **Pick and place** (next phase, kept on the side) | `pick_and_place.yaml` | `PandaPickAndPlacePlanner` / `PandaPickAndPlaceController` | phase machine (PREGRASP → DESCEND → CLOSE → LIFT → TRANSPORT → PLACE → OPEN → RETREAT) re-targeting the same term set via per-phase weights |

A *new task* on the same robot usually needs **only a new yaml** (different
goal weights/terms) plus, if the goal logic differs, a planner that provides:
`nq/nv/nu`, `torque_limits`, differentiable `ee_features(q)`, `dynamics`, and
a packed `reference_vec` matching §5. The cost library and solver are shared.

## 7. Recipe — create and validate a new OCP

1. Copy the closest existing yaml in `sbmpc/ocp_configs/`, rename, edit terms
   and weights. Keep `mpc.dt` consistent with the intended control rate
   (0.04 s ↔ the 25 Hz bridge).
2. Load-check it: `python -c "from sbmpc.ocp import load_ocp_config; load_ocp_config('<name>')"`.
3. Behavior: `pixi run -e cuda python scripts/panda_pregrasp.py --ocp <name> --headless`
   — must print `VERDICT: PASS` (reach < 1 cm, torque/velocity/position
   < 90% of limits, finite stable gains, p95 timing ≤ the control period).
4. Timing: `pixi run -e cuda python tests/bench_controller.py --gains` — p95
   below `1000 * mpc.dt` ms with margin for ~1–3 ms bridge overhead.
5. ROS sim, headless gate:
   `ros2 launch sbmpc_bringup sbmpc_pregrasp_demo.launch.py headless:=true use_rviz:=false max_p95_planning_ms:=40 shutdown_after_validation:=true`
   with `planner_ocp: <name>` in the bridge yaml — must end `verdict: stable`.
6. Only then enable commands on hardware (`sbmpc_franka_lfc_real.launch.py`,
   dry bringup first with `enable_nonzero_control:=false`).

## 8. Runtime weight overrides

`sbmpc.ocp.with_weight_overrides(ocp, {"position_regularization": 20.0})` returns a copy with
matching running/terminal term weights replaced (unknown names ignored) — the
hook for programmatic tuning sweeps without writing yaml files.

## 9. Notes for automated / LLM authoring

- Treat `TERM_BUILDERS` in `sbmpc/costs.py` as the term registry of record;
  regenerate the catalog in §3 from it rather than trusting prose.
- The loader is strict on term names but lenient on omitted sections (`mpc:`
  and `sim:` fall back to `MpcSpec`/`SimSpec` defaults). Always state weights
  as floats.
- The machine-checkable acceptance contract for a generated OCP is the command
  sequence in §7 — steps 2–5 each exit non-zero on failure, so an agent can
  iterate on weights against `VERDICT`/`verdict` without human review.
- Invariants an authoring agent must respect: `num_control_points <= horizon`,
  `num_gain_samples <= num_samples` (top-K F-MPPI gains), `mpc.dt` ↔ control
  rate coherence, `control_regularization` never in `terminal_terms`, and the
  timing budget table in §2 when touching `num_gain_samples`.
