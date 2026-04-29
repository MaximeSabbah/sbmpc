# MuJoCo-ROS2 Deployment Plan — sbmpc + sbmpc_ros

> This document supersedes `ASYNC_GAIN_ROS_GAZEBO_PLAN.md` (to be deleted as
> part of Phase A4). `ASYNC_GAIN_BACKGROUND_PLAN.md` (algorithm) and
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
   into `ros2_control` so the ROS sim shares the exact MJCF used by
   `bench_lfc.py`. Validate with a closed-loop ROS run that reproduces the
   benchmark's success criteria.

After this plan completes, **nothing about the real-robot stack changes** —
real-robot deployment continues to be governed by
`docs/ROS_DEPLOYMENT_ROADMAP.md`.

### Success criteria (acceptance gates, all measured in ROS)

| Metric | Target | Source |
|---|---|---|
| Foreground (bridge timer + planner) period | 50 Hz, p99 ≤ 20 ms | `lfc_bridge_node._on_timer()` + planner `step()` |
| Background gain worker | Running, dropped-snapshot rate ≤ 1 % over a 30 s run | `BridgeDiagnostics` |
| Steady-state EE position error | < 1 mm (Euclidean) at the pregrasp pose, sustained 5 s | New ROS-side test mirroring `bench_lfc._ee_error()` |
| Gain stability | Velocity HF energy (5–50 Hz band) ≤ value reached by `exact-feedback` preset on the same MJCF | New ROS-side check mirroring `bench_lfc._joint_vel_hf_energy()` |
| MuJoCo physics ⟷ bench_lfc parity | Same MJCF (`sbmpc/examples/panda_pick_place/panda.xml`) loaded in both | mujoco_ros2_control `<mujoco_model>` param |

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

- mujoco_ros2_control with **`sbmpc/examples/panda_pick_place/panda.xml`** as
  the loaded MJCF (chosen for strict parity with `PandaPregraspPlanner` —
  `sbmpc/sbmpc/examples/franka_emika_panda/panda_pregrasp.py:20`),
- `linear_feedback_controller` chainable with `joint_state_estimator`
  (already configured in `sbmpc_ros/sbmpc_bringup/config/franka_controllers.yaml`),
- the existing `lfc_bridge_node` running the planner in
  **`exact_async_feedback`** mode (config exists at
  `sbmpc_ros/sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml`),
- gripper held closed via `gripper_action_controller` (no contact tasks in
  this milestone — pregrasp pose only, matching bench_lfc).

The bridge already publishes `Control` at 50 Hz and consumes `Sensor`. We are
**not changing the bridge**; we are only replacing what produces the `Sensor`
stream and consumes the `Control` stream — Gazebo today, mujoco_ros2_control
after this plan.

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
- `sbmpc/tests/`
  - Any test parametrized over gain methods: drop the FD parametrization,
    keep the exact-async one.
- `sbmpc/docs/ASYNC_GAIN_BACKGROUND_PLAN.md` — leave the doc but add a single
  banner line at the top: *"Finite-difference gain path removed YYYY-MM-DD.
  Only the exact background-worker path is supported."* (Single line, no
  rewriting of the body.)

**Verification:** `pytest -q tests/`, then run the four remaining presets
(`exact-feedback`, `exact-phase0`, `exact-feedforward`, `custom`) for 50
steps each; tail EE error must still be < 1 mm on `exact-feedback`.

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
  - lines ~94–105 — delete the gravity-comp comment block and
    `remove_gravity_compensation_effort: false` (Gazebo-only quirk).
    The new sim publishes effort with gravity already included by MuJoCo, so
    this matches the real-robot setting (`remove_gravity_compensation_effort:
    true` in `franka_lfc_params.yaml`). Update the file to be a thin
    sim-specific override or fold it into `franka_lfc_params.yaml` —
    decision deferred to Phase C wiring.
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
- `sbmpc_ros/sbmpc_ros_bridge/` — pure modules
  (`safety.py`, `joint_mapping.py`, `lfc_msg_adapter.py`, `planner_adapter.py`,
  `diagnostics.py`) are unaffected; keep untouched.
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
    version: <pin specific commit SHA — chosen by first agent that runs this>
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
`<exec_depend>` (replacing the three Gazebo deps removed in A2).

**Verification:** `colcon test --packages-select mujoco_ros2_control` (only to
confirm build, not to gate on its own tests passing).

### B2. Smoke-test the upstream demo

Before integrating with the Franka, run the `mujoco_ros2_control_demos`
launch headless:

```
ros2 launch mujoco_ros2_control_demos cart_pole.launch.py headless:=true
```

Confirm `controller_manager` comes up and `/joint_states` ticks. Stop. This
proves the integration layer is functional in this workspace before we layer
Franka complexity on top of it.

---

## 5. Phase C — Wire mujoco_ros2_control into sbmpc_ros

The bridge does **not** change. The planner does **not** change. We rebuild
only the URDF + launch + the one config that fed Gazebo.

### C1. New URDF/xacro

Create `sbmpc_ros/sbmpc_bringup/urdf/franka_arm_with_sbmpc_mujoco.urdf.xacro`.
Required content:

- Includes `franka_description` macros for **visual + collision + transmission**
  (no Gazebo includes).
- `<ros2_control name="MujocoFrankaSystem" type="system">` with
  `<hardware><plugin>mujoco_ros2_control/MujocoSystemInterface</plugin></hardware>`
  and the `<param name="mujoco_model">` pointing to
  `/workspace/sbmpc/examples/panda_pick_place/panda.xml` (the path
  `PandaPregraspPlanner` itself uses — confirmed at
  `sbmpc/sbmpc/examples/franka_emika_panda/panda_pregrasp.py:20`).
- For each of `panda_joint{1..7}`: `command_interface=effort`,
  `state_interface=position,velocity,effort`,
  `<param name="mujoco_joint">actuator{1..7}</param>` mapped to the actuators
  declared in `panda.xml`.
- For the gripper: `panda_finger_joint{1,2}` mapped to the split tendon
  motor in `panda.xml`. The gripper is held closed by
  `gripper_action_controller` for this milestone — gripper actuator may need
  to be re-typed from `<general>` to `<motor>` in `panda.xml` (verify against
  the upstream README; if so, add a single override file
  `sbmpc/examples/panda_pick_place/panda_ros2_control.xml` that includes
  `panda.xml` and overrides actuator types — do **not** mutate the file used
  by bench_lfc).

Keep the same joint name strings the rest of the stack expects
(`panda_joint1..7`, `panda_finger_joint1..2`) — `joint_mapping.py` and the
LFC controllers depend on them.

### C2. New sim launch

Create `sbmpc_ros/sbmpc_bringup/launch/sbmpc_franka_lfc_mujoco_sim.launch.py`.
Skeleton (mirrors the deleted Gazebo launch but minus all gz nodes):

1. `robot_state_publisher` with the new xacro.
2. `mujoco_ros2_control` node (executable name per upstream README;
   typically `mujoco_ros2_control`). It owns the `controller_manager`.
3. `joint_state_broadcaster` spawner (unchanged from old launch).
4. `gripper_action_controller` spawner (unchanged).
5. `joint_state_estimator` + `linear_feedback_controller` spawner
   (`--activate-as-group`, unchanged).
6. The `lfc_bridge_node` Python node, parameters from
   `sbmpc_bridge_exact_async.yaml` (the bench_lfc-equivalent preset).
7. Same `RegisterEventHandler` chain as before, with `Shutdown` on bridge
   exit.

**Do not** add `use_sim_time` indirection beyond what mujoco_ros2_control
itself documents. **Do not** carry over `GZ_SIM_RESOURCE_PATH` or
`disable_gazebo_gravity` arguments.

### C3. Configs

- `franka_lfc_params_sim.yaml` — reduce to a single override file containing
  only the keys that legitimately differ from real (probably none if MuJoCo
  publishes effort with gravity included). If empty, delete and have the
  new launch use `franka_lfc_params.yaml`.
- The new launch file's default `bridge_params_file` must be
  `sbmpc_bridge_exact_async.yaml` (not `sbmpc_bridge.yaml`). This is the
  config that requests the `exact_async_feedback` planner mode — i.e. the
  bench_lfc reference.

### C4. Validation and parity tests (the heart of this plan)

Add these tests under `sbmpc_ros/sbmpc_bringup/test/`:

1. `test_mujoco_launch_imports.py` — import the new launch, assert the node
   set, assert no `gz`/`gazebo` substrings appear anywhere in declared
   args. Pure import-time test; no rclpy runtime.
2. `test_mujoco_xacro.py` — render the xacro to URDF, assert the
   `<plugin>mujoco_ros2_control/MujocoSystemInterface</plugin>` line exists,
   the `mujoco_model` param resolves to a path that exists on disk, and the
   joint/actuator mapping is exhaustive (7 arm + 2 finger).
3. `test_ee_parity_smoke.py` — runtime test (gated behind a
   `pytest.importorskip("rclpy")`). Brings up the launch in a subprocess for
   a 5 s window with `lfc_bridge_node` armed, captures `/sbmpc/control` and
   the joint states, computes EE position via `PandaPregraspPlanner.ee_position`
   (sbmpc is on the `PYTHONPATH`), and asserts:
   - 50 Hz cadence on `/sbmpc/control`, p99 ≤ 20 ms inter-message gap;
   - tail EE error (last 100 samples) < 1 mm;
   - velocity HF energy (5–50 Hz band) within 2× the bench_lfc reference.

The reference numbers for (b) and (c) are produced by running
`bench_lfc.py --preset exact-feedback --steps 250` and recording them in a
JSON fixture committed alongside the test.

---

## 6. Phase D — Run, measure, iterate

Run order:

```
# 1. Algorithm sanity
python -m sbmpc.tests.bench_lfc --preset exact-feedback --steps 250 \
  --json-out /tmp/bench_lfc_reference.json

# 2. Build everything
cd /workspace/ros2_ws && colcon build --symlink-install
source install/setup.bash

# 3. ROS sim
ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py

# 4. In another shell, the parity check
pytest /workspace/ros2_ws/src/sbmpc_ros/sbmpc_bringup/test/test_ee_parity_smoke.py -x
```

Acceptance is **all four** numbers from §1 holding simultaneously over a 30 s
sustained run, not just a 5 s window. The 5 s window is the CI gate; the 30 s
run is the human gate before declaring this milestone done.

If the EE error is > 1 mm, do not retune controller gains. Diagnose in this
order:

1. Confirm `mujoco_model` resolves to the same `panda.xml` bench_lfc loads.
2. Confirm `effort` is being commanded raw (no double-gravity correction).
3. Confirm the bridge is in `exact_async_feedback` mode and the background
   worker is producing gains (`BridgeDiagnostics` `gain_age_ms` < 200 ms).
4. Compare the LFC sign convention — bench_lfc explicitly notes the ROS
   `linear_feedback_controller` sign convention (file header comment); make
   sure the bridge's `planner_output_to_control()` has not been edited.

---

## 7. Phase E — Done

Mark done when, in a fresh checkout following only this document:

- The `git grep` for `gazebo`, `gz_sim`, `ros_gz`, `gain_fd` in both
  repositories returns zero hits in production code.
- `bench_lfc.py --preset exact-feedback --steps 250` succeeds.
- `ros2 launch sbmpc_bringup sbmpc_franka_lfc_mujoco_sim.launch.py` brings
  up cleanly.
- `test_ee_parity_smoke.py` passes.

Real-robot deployment then proceeds per `ROS_DEPLOYMENT_ROADMAP.md` —
unchanged.

---

## 8. Critical files (paths a fresh agent will need)

**`sbmpc/`**
- `sbmpc/tests/bench_lfc.py` — the reference (do not edit beyond the FD
  cleanup in §A1).
- `sbmpc/sbmpc/solvers.py` — async gain worker (§A1: drop FD branch).
- `sbmpc/sbmpc/settings.py` — drop `gain_fd_*` fields.
- `sbmpc/sbmpc/examples/franka_emika_panda/panda_pregrasp.py:20` —
  `PANDA_XML_PATH`. **This is the MJCF mujoco_ros2_control must load.**
- `sbmpc/examples/panda_pick_place/panda.xml` — the model itself.

**`sbmpc_ros/`**
- `sbmpc_ros_bridge/sbmpc_ros_bridge/lfc_bridge_node.py` — unchanged.
- `sbmpc_ros_bridge/sbmpc_ros_bridge/planner_adapter.py` — unchanged.
- `sbmpc_bringup/config/sbmpc_bridge_exact_async.yaml` — the bridge preset
  to use as default in the new launch.
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
2. Running `git status` in `/workspace/sbmpc` and `/workspace/sbmpc_ros` to
   see how much of §A is already done.
3. Running the §6 build chain — failures point to which phase is incomplete.
4. The checklist at §7 is the definitive "are we done" test.

This document is intentionally written so that every concrete change cites
either a file path or a file path + line number. There are no implicit
dependencies on session memory.
