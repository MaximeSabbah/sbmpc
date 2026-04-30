# MuJoCo-ROS2 Deployment Plan — sbmpc + sbmpc_ros

`ASYNC_GAIN_BACKGROUND_PLAN.md` (algorithm) and
> `ROS_DEPLOYMENT_ROADMAP.md` (project arc / real-robot path) remain
> authoritative and unchanged.

---

## 1. Context

`sbmpc/tests/bench_lfc.py` is the trusted reference for SB-MPC + LFC behavior
on the Franka FER. Its `exact-feedback` preset (foreground MPPI at 50 Hz,
exact gains computed in a background thread, LFC law applied
`tau = tau_ff + K @ (x_des - x_meas)`) reaches < 1 mm end-effector error with
stable gains. Several weeks of Gazebo bringup under `sbmpc_ros` have produced
behavior divergence we cannot fully explain (FER gravity model, contact
tuning, inertia workarounds). We are abandoning that path.

This plan does two things, in order:

1. **Clean up.** Remove every Gazebo-related artifact, the deprecated
   finite-difference gain computation path, and any code/configs that exist
   only to support either of those. This applies to **both** `/workspace/sbmpc`
   and `/workspace/sbmpc_ros`.
2. **Replatform sim on `mujoco_ros2_control`.** Wire MuJoCo physics directly
   into `ros2_control` so the ROS sim shares the same physical model, `home`
   keyframe, and PREGRASP reference used by `bench_lfc.py`. Validate with a
   closed-loop ROS run that reproduces the benchmark's success criteria.

After this plan completes, the **real-robot launch topology and safety posture
stay governed by** `docs/ROS_DEPLOYMENT_ROADMAP.md`. One config-level change is
required, though: because the finite-difference gain path is being removed, any
real-robot bridge preset that currently requests `fd_feedback` must be migrated
to the exact background-worker path before it can be used.

### Success criteria (acceptance gates, all measured in ROS)

| Metric | Target | Source |
|---|---|---|
| Foreground (bridge timer + planner) period | 50 Hz, p99 ≤ 20 ms | `lfc_bridge_node._on_timer()` + planner `step()` |
| Background gain worker | Running, dropped-snapshot rate ≤ 1 % over a 30 s run | `BridgeDiagnostics` |
| Steady-state EE position error | < 1 mm (Euclidean) at the pregrasp pose, sustained 5 s | New ROS-side test mirroring `bench_lfc._ee_error()` |
| Gain stability | Velocity HF energy (5–50 Hz band) ≤ value reached by `exact-feedback` preset on the same physical model | New ROS-side check mirroring `bench_lfc._joint_vel_hf_energy()` |
| MuJoCo physics ⟷ bench_lfc parity | Same physical model, same `home` keyframe, and same PREGRASP reference as `bench_lfc.py` | mujoco_ros2_control `<mujoco_model>` param + xacro test |

The two new ROS-side measurements (EE error and gain HF energy) are the
single most important deliverable of this plan: they let any agent verify
parity with `bench_lfc.py` without rerunning the Python benchmark.

---

## 2. North Star — what the new ROS sim must reproduce

Reference command (from `tests/bench_lfc.py`):

```
python -m sbmpc.tests.bench_lfc \
  --preset exact-feedback \
  --steps 250 --dt 0.02 --substeps 20
```

Translates in ROS to: a single launch file that spins up

- mujoco_ros2_control with a model derived from
  **`sbmpc/examples/panda_pick_place/scene.xml`**, because
  `PandaPregraspPlanner` loads `scene.xml` for the `home` keyframe, object
  position, target position, and PREGRASP reference while using `panda.xml`
  for the arm dynamics. If `mujoco_ros2_control` cannot expose the runtime
  `fer_joint*` interfaces against the included `panda.xml` names directly,
  create a ROS-control-specific MJCF wrapper/copy that changes only names and
  actuator types needed by the driver; do **not** mutate the benchmark files.
- `linear_feedback_controller` chainable with `joint_state_estimator`
  (already configured in `sbmpc_ros/sbmpc_bringup/config/franka_controllers.yaml`),
- the existing `lfc_bridge_node` running the planner in
  **`exact_async_feedback`** mode (config exists at
  `sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml`),
- gripper held closed via `gripper_action_controller` (no contact tasks in
  this milestone — pregrasp pose only, matching bench_lfc).

The bridge already publishes `Control` at 50 Hz and consumes `Sensor`. We are
mostly preserving that bridge loop; however, the FD cleanup necessarily touches
the bridge's ROS parameter surface and adapter tests. The runtime joint names
for ROS, `ros2_control`, LFC params, bridge configs, and validation tests are
the FER names exposed by the Franka stack:

```text
fer_joint1 ... fer_joint7
fer_finger_joint1
```

The internal `sbmpc` planner may still use its Panda/Pinocchio names
(`panda_joint*`) inside the algorithm repository. Do not leak those names into
the ROS hardware interfaces unless the active Franka description actually
exports them.

---

## 3. Phase A — Cleanup

Every change in this phase must keep the Python `bench_lfc.py` benchmark
green and the ROS bridge unit tests green. Run after each subphase:

```
# sbmpc
cd /workspace/sbmpc && pytest tests/ -x -q
python -m sbmpc.tests.bench_lfc --preset exact-feedback --steps 50

# sbmpc_ros (no Gazebo, no MuJoCo yet — only the pure modules)
cd /workspace/ros2_ws && colcon build --packages-select sbmpc_ros_bridge sbmpc_bringup
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --pytest-args -x
```

### A1. Remove the finite-difference gain path from `sbmpc/`

Files and call sites to delete (verified via Phase-1 exploration):

- `sbmpc/sbmpc/solvers.py`
  - `_finite_difference_gains()` method (entire definition)
  - branch in `command()` that calls it (lines ~557–565 — `update_gains` +
    `gain_method == "finite_difference"` arm)
- `sbmpc/sbmpc/settings.py`
  - `gain_fd_epsilon`, `gain_fd_scheme`, `gain_fd_num_samples` config fields
    (lines ~155–158) and any property setters.
- `sbmpc/sbmpc/gains.py`
  - Audit: `gains.py` currently only contains `MPPIGain` (async path) and
    helpers — no FD code. Confirm and leave unchanged.
- `sbmpc/tests/bench_lfc.py`
  - Delete `"fast-feedforward"` preset (lines ~73–80) and `"fd-feedback"`
    preset (lines ~81–89).
  - Delete CLI flags `--gain-fd-epsilon`, `--gain-fd-scheme`,
    `--gain-fd-samples` (lines ~886–888) and the corresponding fields on
    `args` / `config.MPC.gain_fd_*` (lines ~169–171).
  - Default `--gain-method` to `exact` and remove the choice list entry for
    `finite_difference`.
- `sbmpc/sbmpc/examples/franka_emika_panda/panda_pregrasp.py` — grep for
  `gain_fd_*`, `finite_difference`; remove any defaults that still set them.
- `sbmpc/sbmpc/examples/franka_emika_panda/planner_api.py`
  - delete `GAIN_MODE_FD_FEEDBACK` and remove it from
    `SUPPORTED_GAIN_MODES`;
  - remove every branch that maps `fd_feedback` to
    `config.MPC.gain_method = "finite_difference"`;
  - make the supported feedback mode `exact_async_feedback`, with
    `gain_samples_per_cycle` and `gain_buffer_size` required when gains are
    enabled.
- `sbmpc/tests/`
  - Any test parametrized over gain methods: drop the FD parametrization,
    keep the exact-async one.
- `sbmpc/docs/ASYNC_GAIN_BACKGROUND_PLAN.md` — leave the doc but add a single
  banner line at the top: *"Finite-difference gain path removed YYYY-MM-DD.
  Only the exact background-worker path is supported."* (Single line, no
  rewriting of the body.)

**Verification:** `pytest -q tests/`, then run the remaining presets
(`exact-feedback`, `exact-phase0`, `exact-feedforward`, `custom`) for 50
steps each; tail EE error must still be < 1 mm on `exact-feedback`. If
`custom` still means "use current defaults", update those defaults to exact
async or make `custom` explicitly require all gain-mode inputs.

### A1b. Remove finite-difference knobs from `sbmpc_ros/`

This is required by §7's `gain_fd` zero-hit gate and by the real launch, whose
default bridge file currently requests `fd_feedback`.

- `sbmpc_ros/sbmpc_ros_bridge/sbmpc_ros_bridge/lfc_bridge_node.py`
  - remove `planner_gain_method`, `planner_gain_fd_epsilon`,
    `planner_gain_fd_scheme`, and `planner_gain_fd_num_samples` parameters;
  - keep only `planner_mode`, `planner_gain_samples_per_cycle`, and
    `planner_gain_buffer_size` for exact async gains.
- `sbmpc_ros/sbmpc_ros_bridge/sbmpc_ros_bridge/planner_adapter.py`
  - remove `gain_method` and `gain_fd_*` fields from
    `PlannerConfigOverrides`, `planner_config_overrides_from_values()`, and
    `apply_config_overrides()`.
- `sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge.yaml`
  - migrate from `planner_mode: fd_feedback` to
    `planner_mode: exact_async_feedback`;
  - add `planner_gain_samples_per_cycle: 128` and
    `planner_gain_buffer_size: 512`;
  - keep `enable_nonzero_control: false` as the safe default for real robot
    bringup.
- `sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml`
  - keep this as the MuJoCo validation preset; it already uses FER joint
    names and exact async settings.
- `sbmpc_ros/sbmpc_ros_bridge/test/` and `sbmpc_ros/sbmpc_bringup/test/`
  - delete FD assertions/fixtures and replace them with exact-async coverage.

### A2. Remove Gazebo from `sbmpc_ros/`

Authoritative deletion checklist (every line below was located in Phase-1
exploration):

- `sbmpc_ros/sbmpc_bringup/package.xml`
  - lines 14, 20, 21 — drop `franka_gazebo_bringup`, `ros_gz_bridge`,
    `ros_gz_sim` exec_depends.
- `sbmpc_ros/sbmpc_bringup/urdf/franka_arm_with_sbmpc_inertials.gazebo.xacro`
  - **Delete the file entirely.** It will be replaced in Phase C by
    `franka_arm_with_sbmpc_mujoco.urdf.xacro`. Do not migrate this file
    in-place — its includes (`franka_gazebo_bringup/urdf/sensors.xacro`,
    `gravity_overrides.xacro`) cannot be carried over.
- `sbmpc_ros/sbmpc_bringup/launch/sbmpc_franka_lfc_sim.launch.py`
  - **Delete the file.** Phase C creates a new
    `sbmpc_franka_lfc_mujoco_sim.launch.py`. Do not edit-in-place — too many
    lines (xacro args, gz spawn, clock_bridge, GZ_SIM_RESOURCE_PATH env,
    declared args) all go away.
- `sbmpc_ros/sbmpc_bringup/sbmpc_bringup/launch_preflight.py`
  - lines 18–30 — remove `/gz_ros_control` and `/ros_gz_bridge` from
    `STALE_SIM_NODE_NAMES`. Add `/mujoco_ros2_control` if a stale-graph
    guard is still desired (decided in Phase C).
- `sbmpc_ros/sbmpc_bringup/sbmpc_bringup/validate_sim.py`
  - line 72 — rename node from `"sbmpc_gazebo_validation_collector"` to
    `"sbmpc_sim_validation_collector"` (one-line edit).
- `sbmpc_ros/sbmpc_bringup/test/test_bringup_config.py`
  - lines 111–117 — delete `test_gazebo_xacro_keeps_gravity_enabled_…`.
- `sbmpc_ros/sbmpc_bringup/test/test_launch_preflight.py`
  - lines 80, 96, 104 — drop `/ros_gz_bridge` mocks and assertions.
- `sbmpc_ros/sbmpc_bringup/test/test_launch_imports.py`
  - drop assertions on `gz_args`, `disable_gazebo_gravity`,
    `GZ_SIM_RESOURCE_PATH` (the new launch file declares neither).
- `sbmpc_ros/sbmpc_bringup/config/franka_lfc_params_sim.yaml`
  - keep a sim-specific LFC params file unless testing proves it can be
    folded away. For MuJoCo direct effort control, LFC must send the absolute
    SB-MPC torque command as-is, so `remove_gravity_compensation_effort` should
    remain `false` in sim. The real robot keeps `true`, because libfranka/FCI
    adds its own gravity compensation downstream.
- `sbmpc_ros/sbmpc_bringup/config/fer_sim_inertials.yaml`
  - **Delete.** Inertia overrides exist only because of Gazebo's link4
    instability; MuJoCo uses the inertias baked into the MJCF.

**Verification (post-A2, before any MuJoCo work):**

```
cd /workspace/ros2_ws && colcon build --packages-select sbmpc_bringup
colcon test --packages-select sbmpc_bringup --pytest-args -x
ros2 launch sbmpc_bringup sbmpc_franka_lfc_real.launch.py --print-description
# Real-robot launch must still parse and import. Sim launch is gone.
```

### A3. Remove unrelated dead code

After A1+A2, audit for orphans (only orphans whose origin we made dead):

- `sbmpc/` — orphaned imports of removed FD symbols (`grep -rn "gain_fd_"
  sbmpc/`).
- `sbmpc_ros/sbmpc_bringup/sbmpc_bringup/pixi_supervisor.py` — confirm no
  Gazebo-specific env vars referenced.
- `sbmpc_ros/sbmpc_ros_bridge/` — pure modules other than the FD-removal
  touch points in §A1b (`planner_adapter.py`, bridge parameter declarations,
  and their tests) should remain untouched.
- `sbmpc_ros/sbmpc_bringup/config/` — drop yamls referenced by zero launch
  files after the sim launch is deleted; keep `sbmpc_bridge.yaml`,
  `sbmpc_bridge_exact_async.yaml`, `sbmpc_bridge_feedforward.yaml`,
  `franka_controllers.yaml`, `franka_lfc_params.yaml`.

`sbmpc_bridge_feedforward.yaml` is kept (still used by tests). Do not delete
pre-existing dead code we did not orphan ourselves — see CLAUDE.md §3.

### A4. Delete superseded plan doc

```
rm /workspace/sbmpc/docs/ASYNC_GAIN_ROS_GAZEBO_PLAN.md
```

`ASYNC_GAIN_BACKGROUND_PLAN.md` and `ROS_DEPLOYMENT_ROADMAP.md` stay.

---

## 4. Phase B — Vendor mujoco_ros2_control

### B1. Pin and clone

Create `/workspace/ros2_ws/src/sbmpc_ros.repos`:

```yaml
repositories:
  mujoco_ros2_control:
    type: git
    url: https://github.com/ros-controls/mujoco_ros2_control.git
    version: <pin specific commit SHA or release tag verified against this ROS distro>
```

Then:

```
cd /workspace/ros2_ws/src
vcs import < sbmpc_ros.repos
cd /workspace/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-up-to mujoco_ros2_control
```

Add `mujoco_ros2_control` to `package.xml` of `sbmpc_bringup` as
`<exec_depend>` (replacing the three Gazebo deps removed in A2). Add explicit
test dependencies for `mujoco_ros2_control_msgs` if the runtime tests call
MuJoCo pause/reset/step services directly.

**Verification:** `colcon test --packages-select mujoco_ros2_control` (only to
confirm build, not to gate on its own tests passing).
At this stage it would be great to also adapt what is in sbmpc_containers to have a clean and reusable installation setup in the Docker.

### B2. Smoke-test the upstream demo

Before integrating with the Franka, run the `mujoco_ros2_control_demos`
launch headless. As of the 0.0.2 docs, the basic demo is:

```
ros2 launch mujoco_ros2_control_demos 01_basic_robot.launch.py headless:=true
```

Confirm `controller_manager` comes up and `/joint_states` ticks. Stop. This
proves the integration layer is functional in this workspace before we layer
Franka complexity on top of it.

---

## 5. Phase C — Wire mujoco_ros2_control into sbmpc_ros

The bridge control law and planner algorithm do **not** change in this phase.
By this point §A1/A1b should already have removed the obsolete FD parameter
surface. Phase C rebuilds the URDF/xacro, launch, MuJoCo model wrapper if
needed, and sim-specific configs that used to feed Gazebo.

### C1. New URDF/xacro

Create `sbmpc_ros/sbmpc_bringup/urdf/franka_arm_with_sbmpc_mujoco.urdf.xacro`.
Required content:

- Includes `franka_description` macros for **visual + collision + transmission**
  (no Gazebo includes).
- `<ros2_control name="MujocoFrankaSystem" type="system">` with
  `<hardware><plugin>mujoco_ros2_control/MujocoSystemInterface</plugin></hardware>`
  and the `<param name="mujoco_model">` pointing to the selected MuJoCo model:
  preferably `sbmpc/examples/panda_pick_place/scene.xml` so the `home`
  keyframe exists, or a generated ROS-control MJCF wrapper that includes the
  same physical model and preserves that keyframe.
- Add `<param name="initial_keyframe">home</param>` if supported by the pinned
  `mujoco_ros2_control` version. The upstream README documents this parameter.
- Add `<param name="headless">$(arg headless)</param>` so container launches do
  not require a GUI.
- For each of `fer_joint{1..7}` expose `command_interface=effort` and
  `state_interface=position,velocity,effort`.
- Do **not** assume `<param name="mujoco_joint">...</param>` is supported. The
  upstream documentation examples map ros2_control joints to MuJoCo
  joints/actuators by name, and effort control is natively supported only for
  MuJoCo `motor`, `general`, or similar actuator types. During implementation,
  verify the pinned source. If explicit ROS-name-to-MJCF-name mapping is not
  supported, create `sbmpc/examples/panda_pick_place/panda_ros2_control.xml`
  as a generated/hand-audited copy whose physical values match `panda.xml` but
  whose arm joints/actuators are named `fer_joint1..7` and use effort-compatible
  actuators.
- For the gripper, expose the single controller joint the current ROS stack
  expects: `fer_finger_joint1`. Keep the second finger coupled internally in
  MJCF via equality/tendon mechanics. `franka_controllers.yaml` uses
  `position_controllers/GripperActionController` on `fer_finger_joint1`, not a
  two-joint gripper controller.

Keep the same FER joint name strings the rest of the ROS stack expects. The
bridge's `joint_names` parameter and LFC `initial_state` snapshot must contain
`fer_joint1..7`, matching `franka_lfc_params*.yaml`.

### C2. New sim launch

Create `sbmpc_ros/sbmpc_bringup/launch/sbmpc_franka_lfc_mujoco_sim.launch.py`.
Skeleton (mirrors the deleted Gazebo launch but minus all gz nodes):

1. `robot_state_publisher` with the new xacro.
2. `mujoco_ros2_control` control node. The current upstream README uses
   `package="mujoco_ros2_control"` and `executable="ros2_control_node"`, with
   `parameters=[{"use_sim_time": True}, controllers_file, optional_plugins_file]`.
   On Humble, remap `("~/robot_description", "/robot_description")` as shown
   in the demos if needed.
3. `joint_state_broadcaster` spawner (unchanged from old launch).
4. `gripper_action_controller` spawner (unchanged).
5. `joint_state_estimator` + `linear_feedback_controller` spawner
   (`--activate-as-group`, unchanged).
6. The `lfc_bridge_node` Python node, parameters from
   `sbmpc_bridge_exact_async.yaml` (the bench_lfc-equivalent preset).
7. Launch argument `enable_nonzero_control`, default `false`, that can override
   the bridge parameter for validation runs. The parity smoke test must set it
   to `true`; the safe default remains silent/PD hold.
8. Event handling adapted to MuJoCo: `Shutdown` on control-node or bridge exit,
   and spawners launched only after `controller_manager` is available. Do not
   blindly reuse the old `spawn_entity`-based event chain because that process
   no longer exists.

**Do not** add `use_sim_time` indirection beyond what mujoco_ros2_control
itself documents. **Do not** carry over `GZ_SIM_RESOURCE_PATH` or
`disable_gazebo_gravity` arguments.

### C3. Configs

- `franka_lfc_params_sim.yaml` — reduce to a single override file containing
  only the keys that legitimately differ from real. Today the important
  difference is `remove_gravity_compensation_effort: false` for direct-effort
  simulation, while real remains `true`.
- The new launch file's default `bridge_params_file` must be
  `sbmpc_bridge_exact_async.yaml` (not `sbmpc_bridge.yaml`). This is the
  config that requests the `exact_async_feedback` planner mode — i.e. the
  bench_lfc reference.
- `sbmpc_bridge.yaml` must also be migrated away from `fd_feedback` during
  §A1b so the real launch remains parseable after FD removal.

### C4. Validation and parity tests (the heart of this plan)

Add these tests under `sbmpc_ros/sbmpc_bringup/test/`:

1. `test_mujoco_launch_imports.py` — import the new launch, assert the node
   set, assert no `gz`/`gazebo` substrings appear anywhere in declared
   args. Pure import-time test; no rclpy runtime.
2. `test_mujoco_xacro.py` — render the xacro to URDF, assert the
   `<plugin>mujoco_ros2_control/MujocoSystemInterface</plugin>` line exists,
   the `mujoco_model` param resolves to a path that exists on disk, and the
   exposed ROS interfaces are exhaustive for `fer_joint1..7` plus the
   configured gripper joint. If a ROS-control-specific MJCF copy is used, assert
   its physical parameters match the benchmark MJCF except for the approved
   name/actuator-type changes.
3. `test_ee_parity_smoke.py` — runtime test (gated behind a
   `pytest.importorskip("rclpy")`). Brings up the launch in a subprocess for
   a 5 s window with `lfc_bridge_node` armed
   (`enable_nonzero_control:=true`), captures `/sbmpc/control` and the FER
   joint states, computes EE position via `PandaPregraspPlanner.ee_position`
   after converting the ordered FER arm vector to the planner's internal
   7-DoF vector, and asserts:
   - 50 Hz cadence on `/sbmpc/control`, p99 ≤ 20 ms inter-message gap;
   - tail EE error (last 100 samples) < 1 mm;
   - velocity HF energy (5–50 Hz band) within 2× the bench_lfc reference.

The reference numbers for (b) and (c) are produced by running the §6
`bench_lfc.py --emit-reference` command and committing the resulting JSON
fixture alongside the test.

---

## 6. Phase D — Run, measure, iterate

Run order:

```
# 1. Algorithm sanity
python -m sbmpc.tests.bench_lfc --preset exact-feedback --steps 250
python -m sbmpc.tests.bench_lfc --preset exact-feedback \
  --timing-mode immediate --steps 250 \
  --emit-reference /tmp/bench_lfc_reference.json

# 2. Build everything
cd /workspace/ros2_ws && colcon build --symlink-install
source install/setup.bash

# 3. ROS sim
ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py \
  enable_nonzero_control:=true

# 4. In another shell, the parity check
pytest /workspace/ros2_ws/src/sbmpc_ros/sbmpc_bringup/test/test_ee_parity_smoke.py -x
```

Acceptance is **all four** numbers from §1 holding simultaneously over a 30 s
sustained run, not just a 5 s window. The 5 s window is the CI gate; the 30 s
run is the human gate before declaring this milestone done.

If the EE error is > 1 mm, do not retune controller gains. Diagnose in this
order:

1. Confirm `mujoco_model` resolves to `scene.xml` or to the approved
   ROS-control MJCF copy/wrapper derived from it.
   If using a ROS-control MJCF copy/wrapper, confirm the only differences from
   the benchmark physical model are approved ROS interface names and actuator
   type changes needed for effort control.
2. Confirm `effort` is being commanded raw in sim:
   `remove_gravity_compensation_effort: false` for MuJoCo, while real remains
   `true`.
3. Confirm the bridge is in `exact_async_feedback` mode and the background
   worker is producing gains (`BridgeDiagnostics.last_gain_worker_running` is
   true, dropped snapshots remain within target, and
   `last_gain_age_cycles * planner_dt < 0.2 s`).
4. Compare the LFC sign convention — bench_lfc explicitly notes the ROS
   `linear_feedback_controller` sign convention (file header comment); make
   sure the bridge's `planner_output_to_control()` has not been edited.

---

## 7. Phase E — Done

Mark done when, in a fresh checkout following only this document:

- The `git grep` for `gazebo`, `gz_sim`, `ros_gz`, `gain_fd`, `fd_feedback`,
  and `finite_difference` in both repositories returns zero hits in production
  code. Scope the grep to production paths so historical docs and the agent log
  do not create false positives.
- `bench_lfc.py --preset exact-feedback --steps 250` succeeds.
- `ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py` brings
  up cleanly.
- `test_ee_parity_smoke.py` passes.

Real-robot deployment topology then proceeds per `ROS_DEPLOYMENT_ROADMAP.md`;
the bridge config has already been migrated away from FD in §A1b.

---

## 8. Critical files (paths a fresh agent will need)

**`sbmpc/`**
- `sbmpc/tests/bench_lfc.py` — the reference (do not edit beyond the FD
  cleanup in §A1 and any reference-output flag alignment needed by §6).
- `sbmpc/sbmpc/solvers.py` — async gain worker (§A1: drop FD branch).
- `sbmpc/sbmpc/settings.py` — drop `gain_fd_*` fields.
- `sbmpc/sbmpc/examples/franka_emika_panda/planner_api.py` — drop
  `fd_feedback` and keep exact async as the supported feedback mode.
- `sbmpc/sbmpc/examples/franka_emika_panda/panda_pregrasp.py:20` —
  `PANDA_XML_PATH`; line 19 is `PANDA_SCENE_PATH`. The ROS sim must preserve
  both the `panda.xml` physical model and the `scene.xml` `home` keyframe /
  PREGRASP reference semantics.
- `sbmpc/examples/panda_pick_place/panda.xml` and `scene.xml` — benchmark
  model/reference inputs. Do not mutate them for ROS naming; add a wrapper/copy
  if needed.

**`sbmpc_ros/`**
- `sbmpc_ros_bridge/sbmpc_ros_bridge/lfc_bridge_node.py` — bridge loop mostly
  unchanged, but FD parameters must be removed.
- `sbmpc_ros_bridge/sbmpc_ros_bridge/planner_adapter.py` — remove FD override
  plumbing; preserve the current planner adapter role.
- `sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml` — the bridge preset
  to use as default in the new launch.
- `sbmpc_bringup/config/sbmpc_bridge.yaml` — real-launch default; migrate from
  `fd_feedback` to exact async during §A1b.
- `sbmpc_bringup/config/franka_controllers.yaml` — shared with real, do not
  edit beyond removing Gazebo-only bits if any (audit; expected: none).

**Created by this plan**
- `sbmpc_ros/sbmpc_bringup/urdf/franka_arm_with_sbmpc_mujoco.urdf.xacro`
- `sbmpc_ros/sbmpc_bringup/launch/sbmpc_franka_lfc_mujoco_sim.launch.py`
- `sbmpc_ros/sbmpc_bringup/test/test_mujoco_launch_imports.py`
- `sbmpc_ros/sbmpc_bringup/test/test_mujoco_xacro.py`
- `sbmpc_ros/sbmpc_bringup/test/test_ee_parity_smoke.py`
- `ros2_ws/src/sbmpc_ros.repos`
- `sbmpc/docs/MUJOCO_ROS2_DEPLOYMENT_PLAN.md` (this file).

---

## 9. Out of scope (explicitly)

- Real-robot deployment (lives in `ROS_DEPLOYMENT_ROADMAP.md`).
- Pick-and-place sequencing (this plan only validates the pregrasp pose
  hold, the same task `bench_lfc.py` runs).
- Algorithm changes (no new gain method, no new objective, no horizon /
  sampling tuning).
- Performance work beyond meeting the four §1 acceptance numbers.
- Any change to `franka_description` or upstream `franka_bringup`.

---

## 10. Resumability — for any agent picking this up cold

A fresh session can resume by:

1. Reading this file end-to-end.
2. Reading §11 newest-first to see what previous agents actually changed,
   verified, skipped, or discovered.
3. Running `git status` in `/workspace/sbmpc` and `/workspace/sbmpc_ros` to
   see how much of §A is already done.
4. Running the §6 build chain — failures point to which phase is incomplete.
5. The checklist at §7 is the definitive "are we done" test.

This document is intentionally written so that every concrete change cites
either a file path or a file path + line number. There are no implicit
dependencies on session memory.

---

## 11. Agent Work Log

Every agent/session that makes progress on this plan must append a short entry
here before ending the turn. This is the durable memory for future Codex or
agentic coding sessions.

Required format:

```markdown
### YYYY-MM-DD — Agent / Session
- Scope:
- Changed:
- Verified:
- Not verified / blockers:
- Next handoff:
```

Keep entries factual and compact. Include command names and relevant file
paths, but do not paste long logs. If an agent changes direction from this plan,
record the reason here and update the plan section itself in the same commit.

### 2026-04-29 — Codex Review / Plan Correction
- Scope: Reviewed `MUJOCO_ROS2_DEPLOYMENT_PLAN.md` against current
  `sbmpc`, `sbmpc_ros`, and upstream `mujoco_ros2_control` documentation.
- Changed: Corrected the plan to use FER runtime joint names, added ROS-side
  finite-difference cleanup, fixed MuJoCo direct-effort gravity guidance,
  replaced the stale demo launch name, corrected the control-node executable,
  added bridge arming requirements for parity tests, and added this work log.
- Verified: Read current configs/code including `franka_lfc_params*.yaml`,
  `franka_controllers.yaml`, `sbmpc_bridge*.yaml`, `lfc_bridge_node.py`,
  `planner_adapter.py`, `planner_api.py`, `panda_pregrasp.py`, `panda.xml`,
  and `scene.xml`; checked upstream docs for `mujoco_ros2_control`
  `MujocoSystemInterface`, `ros2_control_node`, actuator interface support,
  gripper mimic guidance, `initial_keyframe`, and headless launch support.
- Not verified / blockers: Did not build or run ROS/MuJoCo; implementation must
  still confirm whether the pinned `mujoco_ros2_control` supports explicit
  ROS-name-to-MJCF-name mapping. If not, create an audited ROS-control MJCF
  copy/wrapper with FER names and effort-compatible actuators.
- Next handoff: Start implementation at §A1/A1b. Keep `fer_joint*` names in
  all ROS interfaces, and append a new entry here after each completed phase.

### 2026-04-29 — Codex Phase A Implementation
- Scope: Implemented the Phase A cleanup needed before MuJoCo wiring: removed
  obsolete finite-difference gain support from `sbmpc`, migrated ROS bridge
  defaults to exact async feedback, and removed legacy Gazebo bringup assets.
- Changed: `sbmpc/settings.py` and `sbmpc/solvers.py` are exact-gain only;
  Panda planner configs now use exact buffered gains; `planner_api.py` supports
  only `feedforward` and `exact_async_feedback`; bridge parameters no longer
  expose `planner_gain_method` or `planner_gain_fd_*`; `sbmpc_bridge.yaml`
  defaults to `exact_async_feedback` with `128/512` gain buffering; Gazebo
  launch/xacro/inertial workaround files were deleted from `sbmpc_bringup`;
  tests and README were updated to match.
- Verified: `rg` zero-hit checks passed for finite-difference gain symbols in
  non-doc `sbmpc`, for finite-difference bridge symbols in `sbmpc_ros`, and for
  legacy `franka_gazebo` / `ros_gz` / `gz_sim` / `gz_ros_control` hooks in
  `sbmpc_ros`. Tests passed:
  `pixi run python -m pytest tests/test_mppi_gains.py` with ROS env vars
  unset, `pixi run python -m pytest tests/test_planner_api.py` with ROS env
  vars unset, and focused `sbmpc_ros` pytest for planner adapter, bridge config,
  bringup config, launch imports, launch preflight, and validation helpers.
- Not verified / blockers: Did not vendor or build `mujoco_ros2_control`, did
  not add the MuJoCo xacro/launch, and did not run behavior metrics against
  live MuJoCo or hardware yet.
- Next handoff: Continue at §B/§C: vendor/build `mujoco_ros2_control`, add the
  FER-named MuJoCo ros2_control xacro/launch, then run the §6/§7 behavior
  metric validation.

### 2026-04-30 — Codex Phase B/C Implementation
- Scope: Implemented the first MuJoCo ROS2-control integration slice after the
  completed Phase A cleanup: pinned and imported upstream MuJoCo dependencies,
  added the FER-named MuJoCo model/URDF/launch surface, and added tests around
  the new wiring.
- Changed: Created `/workspace/ros2_ws/src/sbmpc_ros.repos` with
  `mujoco_vendor` `0.0.8` (`26187d69dd239adc45c121af780d57590d189686`) and
  `mujoco_ros2_control` `0.0.2`
  (`c277c1d243af4ee81fb114ad47a985c1c540b8d4`); imported both under
  `/workspace/ros2_ws/src`; added the same pinned manifest to
  `sbmpc_containers/repos/mujoco_ros2_control.repos` and wired it into the
  Docker build. Added `sbmpc_bringup/mujoco/panda_ros2_control.xml` and
  `panda_pick_place_ros2_control_scene.xml` as an audited ROS-control MJCF copy
  with `fer_joint1..7`, `fer_finger_joint1`, motor arm actuators, and a
  1 ms timestep matching `bench_lfc.py --dt 0.02 --substeps 20`. Added
  `franka_arm_with_sbmpc_mujoco.urdf.xacro`,
  `sbmpc_franka_lfc_mujoco_sim.launch.py`, MuJoCo launch/xacro/parity tests,
  installed MJCF data files, reduced `franka_lfc_params_sim.yaml` to the
  direct-effort gravity override, and updated the stale-node guard to include
  `/mujoco_ros2_control_node`.
- Verified: `vcs import /workspace/ros2_ws/src <
  /workspace/ros2_ws/src/sbmpc_ros.repos`; `rosdep update --rosdistro jazzy`;
  `rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy`;
  `colcon build --symlink-install --packages-up-to mujoco_ros2_control
  sbmpc_bringup`; `colcon test --packages-select mujoco_ros2_control`;
  `colcon test --packages-select sbmpc_bringup --pytest-args -q`; final
  `colcon test-result --verbose` reported `180 tests, 0 errors, 0 failures,
  2 skipped`. Full local Python suite passed with `77 passed, 2 skipped`.
  Rendered the installed xacro to `/tmp/franka_arm_with_sbmpc_mujoco.urdf` and
  confirmed FER joints plus `mujoco_ros2_control/MujocoSystemInterface`. A
  30 s headless safe launch with `enable_nonzero_control:=false` reached
  MuJoCo hardware initialization, arm effort actuator registration, gripper
  position actuator registration, controller activation, bridge startup, and
  planner JIT warmup; log grep found no errors, only the expected headless GLFW
  camera warning.
- Not verified / blockers: Did not run the armed
  `enable_nonzero_control:=true` parity test or the 30 s acceptance metrics
  for foreground p99 timing, EE error, dropped gain snapshots, or HF velocity
  energy. `test_ee_parity_smoke.py` is present but intentionally skipped unless
  `SBMPC_RUN_MUJOCO_PARITY=1` is set.
- Next handoff: Run the live parity gate from §6 with
  `SBMPC_RUN_MUJOCO_PARITY=1` and
  `ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py
  enable_nonzero_control:=true`; collect the §1 metrics over 5 s and then
  30 s before declaring Phase D/E complete.
