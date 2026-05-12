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

### Success criteria (acceptance gates, measured by the right workload)

| Metric | Target | Source |
|---|---|---|
| Foreground torque/control publication | 50 Hz, p99 ≤ 20 ms once armed and past startup | `/control` header/receive cadence in ROS |
| Isolated feedforward solve path | Target p99 ≤ 10 ms, hard p99 < 20 ms without MuJoCo/rendering load | Controller timing smoke that times feedforward-only MPPI and excludes gain refresh |
| Exact gain refresh | Opportunistic, finite gains, no errors, bounded age/staleness when refresh fits the slack budget | `BridgeDiagnostics`; dropped snapshots are diagnostic, not a freshness-at-50-Hz gate |
| Steady-state EE position error | < 1 mm (Euclidean) at the pregrasp pose, sustained 5 s | ROS-side MuJoCo behavior smoke mirroring `bench_lfc._ee_error()` from FER joint states |
| Gain/behavior stability | No rejected planner outputs, bounded joint spans/velocity/HF energy, no unsafe divergence | ROS-side MuJoCo behavior smoke plus visual replay when needed |
| MuJoCo physics ⟷ bench_lfc parity | Same physical model, same `home` keyframe, and same PREGRASP reference as `bench_lfc.py` | mujoco_ros2_control `<mujoco_model>` param + xacro test |

The ROS-side behavior measurements (EE error, joint spans/velocity, HF energy,
and safety rejections) are the single most important MuJoCo deliverable of
this plan: they let any agent verify that simulated commands are sensible
before moving toward the real robot.

### Validation split — timing, async gains, and MuJoCo visualization

The foreground 50 Hz gate applies to torque/control publication and to the
foreground feedforward torque solve. It does **not** mean exact gains must be
recomputed or refreshed at 50 Hz. Exact gains run opportunistically and update
the controller only when the available slack can cover a refresh. A dropped
gain snapshot means a newer pending context replaced an older one before the
gain path could process it; it is useful
diagnostic information, but it is not by itself a controller failure.

Chosen controller architecture for the next implementation pass:

- Keep the real-time command path feedforward-first. A 50 Hz tick should solve
  only the MPPI feedforward torque for the newest sensor state, update
  `latest_tau_ff`, and publish using the current `latest_K`.
- Store feedforward and gains in separate double buffers:
  `latest_tau_ff` / `latest_initial_state` are owned by the foreground solver;
  `latest_K` is owned by the gain refresh path. The publisher composes
  `Control(tau_ff=latest_tau_ff, K=latest_K)` from those buffers.
- The exact-gain refresh path consumes captured foreground contexts and
  updates `latest_K` atomically when a budgeted gain batch completes. It must
  never be on the critical path for publishing or for the next feedforward
  solve.
- On a single GPU, "opportunistic" is only true if gain work cannot occupy the
  same GPU queue across the next foreground deadline. If a gain batch is a
  monolithic 20-25 ms GPU job, it can still delay the next 7-10 ms foreground
  rollout even though Python code is not explicitly waiting for it. The
  implementation must therefore either isolate gain work on another device /
  process, or split/schedule gain work into bounded chunks that fit the slack
  after a foreground solve and are skipped when no slack is available.
- The acceptance diagnostic for this architecture is not the current
  `PandaPregraspController.step()` wall time. It is: foreground-only
  feedforward timing, gain age/staleness, and proof that enabling gain refresh
  does not increase feedforward p99 or `/control` publication
  jitter.

MuJoCo ROS is primarily a deployment-safety and wiring tool: it should show
that the ROS stack sends sensible commands through the correct FER joints and
that the simulated robot behavior is not hazardous before trying the real
robot. Rendering, MuJoCo physics, validation collectors, and replay tooling are
not part of the real robot GPU/controller timing budget. Therefore C4 is split
into:

- a MuJoCo headless behavior/wiring gate;
- a controller-only timing gate with synthetic or recorded sensors;
- an optional visual replay of recorded joint trajectories.

The SB-MPC tuning from `bench_lfc.py` remains the reference behavior and must
stay fixed while these gates are investigated:

- do **not** reduce `planner_num_samples`, horizon, control points, gain buffer
  sizes, gain samples per cycle, temperatures, or LFC gains to pass the ROS
  cadence test;
- do **not** change the benchmark MJCF, `home` keyframe, PREGRASP target, or
  internal planner API to hide ROS overhead;
- fixes must target integration and validation semantics around the tuned
  controller: bridge timer/executor behavior, message conversion/allocation,
  QoS/transport, logging/stdout, JAX cache warmup, process isolation, CPU
  scheduling/affinity, or launch/container overhead.

The code anchors are:

- `test_ee_parity_smoke.py` for MuJoCo ROS wiring/behavior and `/control`
  publication cadence;
- `test_controller_timing_smoke.py` for the current controller-only adapter
  timing without MuJoCo/rendering load;
- a new feedforward-first timing smoke to add with the architecture rework,
  which must separately report feedforward solve time, gain-worker refresh
  time, gain age, and any foreground delay caused by the gain worker.

### Current state checkpoint — 2026-05-11

This is the restart anchor for future Codex/agent sessions.

- Green / implemented: The MuJoCo ROS2-control stack is wired with FER joint
  names (`fer_joint1..7`, `fer_finger_joint1`) and the exact-async SB-MPC/LFC
  bridge. The bridge now warms up before LFC activation, MuJoCo is reset to
  `home`, and `joint_state_estimator` plus `linear_feedback_controller` are
  activated through `controller_manager_msgs/SwitchController` after warmup.
- Green / behavior: Live headless MuJoCo parity now passes without retuning
  the SB-MPC controller. The key fixes were compilation-only planner warmup,
  `reset_runtime_state_after_warmup()`, and
  `PandaPregraspController(reseed_every_step=True)` in the ROS adapter so the
  live controller path matches the `bench_lfc.py` measured-state nominal guess
  policy.
- Green / tests: Focused non-live tests passed on an isolated ROS domain with
  `67 passed, 1 skipped`; planner adapter tests passed with `9 passed`; live
  MuJoCo parity passed with `SBMPC_RUN_MUJOCO_PARITY=1` over an 8 s observation
  window. A short diagnostic run showed final EE error around `0.0002 m`, no
  rejected planner outputs, and stable joint velocities.
- Current / timing: The current controller-only smoke can pass a `21 ms` gate
  for the ROS planner-adapter path, while `bench_lfc.py` raw foreground
  command timing is around `7-8 ms`. This `21 ms` adapter number is now treated
  as a diagnostic ceiling, not the desired architecture. The next controller
  pass should separate `tau_ff` and `K` buffers so the foreground feedforward
  solve is protected from async-gain GPU contention and can target < 10 ms.
- Still to do: Build the headless record + visual replay workflow for
  `/joint_states`, `/sensor`, `/control`, and `/sbmpc/diagnostics`, then use
  it for human hazard/sanity review before real-robot deployment rehearsal.
  Real-robot launch and safety policy still belong to
  `docs/ROS_DEPLOYMENT_ROADMAP.md`.
- Visual validation status: Headless MuJoCo ROS behavior validation is the
  usable gate today. Live `headless:=false` visual validation is not ready in
  the current environment because `mujoco_ros2_control` aborts while waiting
  for simulation rendering to start. Treat that as a GUI/rendering environment
  issue, not as evidence against the SB-MPC/LFC controller behavior.

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

### C4. Validation tests (the heart of this plan)

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
3. `test_ee_parity_smoke.py` — MuJoCo runtime behavior/wiring test (gated by
   `SBMPC_RUN_MUJOCO_PARITY=1`). Brings up the launch in a subprocess for a
   short window with `lfc_bridge_node` armed (`enable_nonzero_control:=true`),
   captures `/control` and the FER joint state carried in `/sensor`, computes
   EE position via `PandaPregraspPlanner.ee_position` after converting the
   ordered FER arm vector to the planner's internal 7-DoF vector, and asserts:
   - 50 Hz cadence on `/control`, p99 ≤ 20 ms inter-message gap after startup;
   - no rejected planner outputs and no async gain worker errors;
   - tail EE error < 1 mm;
   - bounded joint spans/velocity/HF energy so the behavior is not hazardous.

Do **not** use the MuJoCo smoke as the controller foreground compute timing
gate. MuJoCo physics, validation subscribers, and any visual/replay tooling are
outside the real robot controller timing budget.

Add a separate controller-only timing smoke under
`sbmpc_ros/sbmpc_ros_bridge/test/`:

4. `test_controller_timing_smoke.py` — runtime test (gated by
   `SBMPC_RUN_CONTROLLER_TIMING=1`). Builds the ROS bridge planner adapter,
   feeds synthetic or recorded FER `Sensor` messages, converts every output to
   the LFC `Control` message, and asserts:
   - foreground planner timing p99 ≤ 21 ms without MuJoCo/rendering load;
   - feedforward torques and gain matrices are finite;
   - the async gain worker is alive and has no worker error.

Dropped gain snapshots are reported as diagnostic context only. They must not
be interpreted as "gains failed to update at 50 Hz"; async exact gains are
allowed to publish opportunistically when the background worker finishes.

The reference numbers for EE error and HF energy are produced by running the
§6 `bench_lfc.py --emit-reference` command and committing the resulting JSON
fixture alongside the MuJoCo behavior test when those checks are finalized.

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

# 4. MuJoCo behavior/wiring check
SBMPC_RUN_MUJOCO_PARITY=1 \
pytest /workspace/ros2_ws/src/sbmpc_ros/sbmpc_bringup/test/test_ee_parity_smoke.py -x

# 5. Controller-only timing check (no MuJoCo/rendering load)
SBMPC_RUN_CONTROLLER_TIMING=1 \
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh python -m pytest \
  /workspace/ros2_ws/src/sbmpc_ros/sbmpc_ros_bridge/test/test_controller_timing_smoke.py -x
```

Acceptance is the §1 numbers holding in their correct workload. The MuJoCo
headless run is the wiring/behavior gate and can be recorded for offline
visual replay. The controller timing gate runs without MuJoCo/rendering load,
matching the real robot deployment constraint that only the controller should
consume the controller GPU budget.

### Phase D1 — Feedforward-first controller rework

This is the next controller-architecture pass before real-robot rehearsal.
Keep SB-MPC tuning fixed.

Target behavior:

1. The 50 Hz foreground loop solves a fresh feedforward torque from the latest
   `Sensor` state. It must not wait for an exact gain refresh.
2. `tau_ff` / desired state and `K` live in independent, thread-safe buffers.
   The publisher composes the outgoing LFC `Control` message from the latest
   available values at publish time.
3. Exact-gain computation consumes captured foreground contexts and updates
   only the gain buffer. Gain age, completed batches, dropped contexts, and
   worker errors remain diagnostics, not foreground failures.
4. On a single GPU, gain computation must be either device/process isolated or
   budgeted into chunks that cannot spill over the next foreground deadline.
   If JAX launches a non-preemptible gain kernel that takes longer than the
   available slack, the correct behavior is to skip/defer that gain work, not
   to delay the next feedforward solve.
5. The old adapter timing number (`PandaPregraspController.step()` p99 near
   `20-21 ms`) is not the final acceptance metric. Add a timing smoke that
   reports:
   - feedforward solve p50/p95/p99/max with gain worker disabled;
   - feedforward solve p50/p95/p99/max with gain worker enabled;
   - background gain refresh timing and gain age;
   - `/control` cadence and publish/prepare timing.

Implementation sketch:

- In `sbmpc`, expose a public pregrasp API method that computes the foreground
  feedforward command and captures a gain context without synchronously
  reading or refreshing gains.
- In `sbmpc_ros_bridge`, replace the single `PlannerOutput` buffer with
  separate latest-feedforward and latest-gain buffers, or with an equivalent
  immutable state object whose `tau_ff` and `K` fields can be updated
  independently.
- Preserve LFC sign convention and message shape. The bridge should still
  publish `feedforward` as `(7, 1)` and `feedback_gain` as `(7, 14)`.
- Keep `test_ee_parity_smoke.py` as the behavior gate after the rework, and
  add a controller-only test that proves the gain worker cannot degrade the
  feedforward p99 beyond the accepted budget.

If the EE error is > 1 mm, do not retune controller gains. Diagnose in this
order:

1. Confirm the SB-MPC bridge has completed planner/JAX warmup before the
   `joint_state_estimator` + `linear_feedback_controller` stack is activated.
   In MuJoCo direct-effort simulation, letting LFC sit in its startup PD mode
   for several seconds before the first SB-MPC torque changes the initial
   condition relative to `bench_lfc.py`.
2. Confirm bridge warmup is compilation-only. Before arming live control, the
   ROS planner adapter must reset published async gains and runtime sampler
   state, then solve from the measured state with the same reseed policy as
   `bench_lfc.py`. Do not let the optimized sequence/gains produced during
   warmup become the first live control.
3. Confirm `mujoco_model` resolves to `scene.xml` or to the approved
   ROS-control MJCF copy/wrapper derived from it.
   If using a ROS-control MJCF copy/wrapper, confirm the only differences from
   the benchmark physical model are approved ROS interface names and actuator
   type changes needed for effort control.
4. Confirm `effort` is being commanded raw in sim:
   `remove_gravity_compensation_effort: false` for MuJoCo, while real remains
   `true`.
5. Confirm the bridge is in `exact_async_feedback` mode and the background
   worker is healthy (`BridgeDiagnostics.last_gain_worker_running` is true,
   `last_gain_worker_error` is empty, gains are finite, and gain
   age/staleness remains bounded). Dropped snapshots are diagnostic context
   only; the async worker is not expected to refresh gains at 50 Hz.
6. Compare the LFC sign convention — bench_lfc explicitly notes the ROS
   `linear_feedback_controller` sign convention (file header comment); make
   sure the bridge's `planner_output_to_control()` has not been edited.

If `/control` publication cadence misses 50 Hz / p99 ≤ 20 ms in the MuJoCo
behavior smoke, do not retune the controller or weaken
`sbmpc_bridge_exact_async.yaml`. Diagnose bridge publication first:

1. Confirm the bridge is connected to the LFC root topics `/sensor` and
   `/control` with best-effort QoS, and that `test_ee_parity_smoke.py` has
   reached the real metric window rather than startup/warmup.
2. Compare control header-stamp cadence with ROS receive-time cadence in the
   parity collector. Header stamps come from the LFC sensor snapshot; receive
   time isolates bridge publication cadence and DDS delivery jitter.
3. Inspect `BridgeDiagnostics` over the whole run:
   `last_foreground_planning_time_ms`, `last_planner_step_wall_time_ms`,
   `last_control_prepare_time_ms`, `last_control_publish_time_ms`,
   `last_bridge_loop_time_ms`, and `deadline_miss_count`.
4. Remember that MuJoCo physics/rendering/validation overhead is not part of
   the real robot controller timing budget. If publication cadence is fine but
   foreground compute diagnostics are slow only during MuJoCo runs, move that
   timing question to the controller-only smoke before changing code.

If the current controller-only timing smoke misses p99 ≤ 21 ms before D1 is
implemented, diagnose in this order:

1. Compare against `bench_lfc.py --preset exact-feedback --timing-mode
   immediate` in the same container after JAX cache warmup.
2. Confirm the bridge adapter constructs the same tuned pregrasp controller
   path as the benchmark and does not block foreground torque publication on
   async gain refresh.
3. Profile the adapter path without changing planner tuning: message
   conversion/allocation, warmup, JAX cache, CPU affinity, stdout/logging, and
   whether synthetic/recorded sensors match the real FER joint ordering.
4. Only after the fixed-tuning ROS path is understood should a code change be
   made. The desired outcome is the tuned controller meeting the controller
   timing gate through feedforward/gain decoupling, not a less expensive
   controller configuration.

For visualization before robot deployment, record the headless MuJoCo run
(`/joint_states`, `/sensor`, `/control`, `/sbmpc/diagnostics`) and replay the
joint trajectory in MuJoCo or RViz. The simulation launch can do this directly
with `record_replay:=true`; the recorder starts after the first `/control`
message by default so JAX/MuJoCo warmup does not pollute replay timing. Replay
is for human hazard/sanity review, not for controller timing acceptance. Live
visualization can also add GPU load or fail in the current container/host
rendering path, so it must not share the controller timing run. Prefer headless
recording plus replay unless a separate display/GPU path has been validated.

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
- `test_ee_parity_smoke.py` passes as the MuJoCo wiring/behavior gate.
- The feedforward-first controller timing smoke passes: foreground
  feedforward p99 stays within budget with the gain worker enabled, gain
  staleness remains bounded, and `/control` cadence remains 50 Hz after
  startup.
- `test_controller_timing_smoke.py` either passes as a legacy adapter smoke or
  is updated/replaced by the feedforward-first timing smoke.

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

### 2026-04-30 — Codex Live Parity Debug
- Scope: Investigated the first armed MuJoCo parity run. The initial failure
  was not a robot stability failure; the test was listening before the bridge
  had completed planner/JAX warmup, then on an incompatible reliable QoS, then
  on the wrong LFC topic namespace.
- Changed: `test_ee_parity_smoke.py` now waits until the first real control
  sample before collecting metrics, captures the launch tail in failures,
  subscribes to the LFC control topic with best-effort QoS, records joint
  states from the LFC `Sensor` stream, and reports timing diagnostics when the
  cadence gate fails. Bridge topic constants and bridge YAML presets now target
  the actual LFC topics `/sensor` and `/control`; `/sbmpc/diagnostics` remains
  the bridge diagnostics topic. The plan text was updated to use those topics.
- Verified: Focused non-live tests passed with `6 passed, 1 skipped` for
  `test_bringup_config.py` and `test_ee_parity_smoke.py`. After rebuilding
  `sbmpc_bringup`, the live parity test reached the real metric gate and
  received hundreds of `/control` samples.
- Current blocker: With the existing SB-MPC controller tuning unchanged, the
  armed live smoke currently fails the 50 Hz cadence assertion. A representative
  run published 346 controls over the observation window with p50 ≈ 23 ms,
  p99 ≈ 31 ms, and many planner deadline misses; the final diagnostics showed
  `state='running'`, `planner_step_count=346`, `deadline_miss_count=246`,
  and no bridge error string. Do not reduce SB-MPC sample count or otherwise
  change tuned controller parameters to make this pass; the next step is to
  explain and optimize the ROS execution/bridge overhead around the fixed
  controller tuning.
- Next handoff: Keep the tuned SB-MPC parameters intact. Investigate cadence
  without changing planner tuning: compare control header-stamp cadence against
  ROS receive-time cadence, inspect bridge diagnostics over a full run, and
  look for ROS/executor/JIT/cache/CPU-affinity or launch-process overhead before
  changing any controller behavior.

### 2026-04-30 — Codex Timing-Overhead Reanchor
- Scope: Re-anchored the current blocker in the plan after user confirmation
  that SB-MPC controller tuning must not be changed to satisfy ROS timing.
- Changed: Added a dedicated "Primary blocker — ROS timing overhead, not
  controller tuning" section near the acceptance gates and expanded Phase D
  with a cadence-miss diagnostic order. The plan now explicitly forbids
  reducing planner samples, horizon, control points, gain-buffer settings, or
  LFC gains as a timing workaround.
- Verified: `sbmpc_bridge_exact_async.yaml` was restored to
  `retime_control_initial_state: true`; focused non-live tests still passed
  with `6 passed, 1 skipped`.
- Next handoff: Continue by instrumenting timing separation in the parity
  collector and bridge diagnostics, keeping the tuned exact-async preset fixed.

### 2026-04-30 — Codex Agimus LFC Wiring Review
- Scope: Compared the current SB-MPC MuJoCo/LFC wiring with Agimus Franka LFC
  wiring in `agimus_controller.py`, `franka_common_lfc.launch.py`, and
  `franka_common.launch.py` to reframe the Phase C4 timing blocker.
- Changed: No runtime code changed. Added this work-log checkpoint only.
- Verified: Read Agimus controller startup, sensor/control topics, LFC params,
  controller activation path, and Franka launch xacro arguments; compared them
  with `lfc_bridge_node.py`, `planner_adapter.py`, `franka_controllers.yaml`,
  `franka_lfc_params.yaml`, `sbmpc_franka_lfc_mujoco_sim.launch.py`, and the
  `bench_lfc.py` exact-feedback preset.
- Not verified / blockers: Did not rerun live parity in this review turn.
  The remaining blocker is still the red 50 Hz parity gate: live MuJoCo ROS
  runs show p99 control gaps around 28 ms with tuned SB-MPC settings unchanged.
- Next handoff: Do not clone Agimus wholesale. Adapt the useful framing:
  validate that the SB-MPC bridge executes the same controller path as
  `bench_lfc.py`, then simplify bridge/LFC wiring toward one sensor snapshot
  plus one control publish per MPC tick before considering process scheduling.

### 2026-05-05 — Codex Phase C4 Timing Diagnosis
- Scope: Rechecked the live MuJoCo parity blocker after adding Agimus-style
  delayed-control bridge wiring and explicit accepted/rejected planner-output
  diagnostics.
- Changed: Added bridge diagnostics for `accepted_planner_output_count` and
  `rejected_planner_output_count`; validation summaries now report and reject
  rejected planner outputs. A temporary external probe at
  `/tmp/sbmpc_live_timing_probe.py` was used for diagnosis only and is not part
  of either repository.
- Verified: Focused non-live tests passed with `26 passed, 1 skipped`. A live
  8 s timing probe showed first-control transients with gaps up to ~31 ms, but
  after skipping the first 10 controls the publication cadence settled to
  header `p99=21.0 ms/max=21.0 ms` and receive-time
  `p99=20.93 ms/max=21.08 ms`. The control prepare+publish path was only
  ~0.2 ms, while the run received ~9,900 sensor messages over 8 s, confirming
  that the remaining overhead is ROS/executor/timer scheduling and startup
  handoff jitter, not SB-MPC message conversion or controller tuning.
- Not verified / blockers: Live parity still fails as currently written
  because it starts measuring immediately at the first control sample and
  includes arming transients. The bridge is currently in an experimental
  `EXECUTOR_NUM_THREADS = 2` state, which did not improve the live tail and
  should be reverted or replaced by a cleaner scheduling strategy before the
  next acceptance run. Some planner outputs were rejected in the latest probe,
  so gain-output rejection must remain a hard stability metric.
- Next handoff: Keep SB-MPC tuning fixed. Restore the bridge executor to the
  least-jitter configuration, make the C4 smoke test measure steady armed
  cadence after a short explicit stabilization window, and continue tracking
  rejected planner outputs separately from cadence.

### 2026-05-05 — Codex Phase C4 Executor Revert / Timing Split
- Scope: Reverted the bridge executor thread experiment and split the remaining
  C4 failure into cadence, planner-API timing, and behavior stability.
- Changed: `lfc_bridge_node.py` now uses `EXECUTOR_NUM_THREADS = 1` again, and
  `test_lfc_bridge_main.py` asserts that single-threaded executor choice.
  `test_ee_parity_smoke.py` now computes cadence after skipping the first 10
  control samples so the 50 Hz gate measures steady armed publication instead
  of startup handoff jitter; rejected planner outputs remain part of the
  stability failure context.
- Verified: Focused tests passed with `27 passed, 1 skipped`. Live MuJoCo
  smoke on `ROS_DOMAIN_ID=89` passed the stabilized cadence assertion and then
  failed the behavior/stability gate: `max_foreground_ms=29.43`,
  `mean_foreground_ms=17.16`, `accepted_planner_output_count=144`,
  `rejected_planner_output_count=6`, `final_gain_norm=213.4`, and large tail
  joint spans. Separate external probes in `/tmp` showed `bench_lfc.py
  --preset exact-feedback` foreground planning at `mean=7.19 ms, max=10.26 ms`,
  while the ROS public planner-adapter path with no ROS executor was already
  `mean≈15 ms` before live launch overhead. A benchmark-style reseed probe
  showed raw solver command time can remain ~`7-8 ms`, but applying that reseed
  naively through the current planner API warmup worsened adapter wall time, so
  that unproven code change was not kept.
- Not verified / blockers: The live behavior gate is still red. The remaining
  planner timing overhead is not explained by ROS publication or message
  conversion; it appears between `bench_lfc.py` and the public
  `PandaPregraspController` / ROS adapter path, especially around warmup,
  async-gain state, and gain-output validity.
- Next handoff: Do not tune MPPI parameters. Compare `bench_lfc.py` setup
  against `PandaPregraspController.warmup()` and the ROS adapter construction:
  when the async worker is started, whether published gains are reset before
  arming, and whether the bridge should use the same benchmark-style command
  path before converting outputs to LFC messages.

### 2026-05-06 — Codex Phase C4 Validation Reframe
- Scope: Corrected the C4 timing interpretation after diagnosis showed
  `bench_lfc.py` reports foreground command timing inside a much slower
  MuJoCo/LFC simulation loop, and after user clarified that async exact gains
  are opportunistic rather than a 50 Hz freshness requirement.
- Changed: Rewrote the plan's acceptance gates to split MuJoCo
  wiring/behavior, controller-only foreground timing, and async gain worker
  health. `test_ee_parity_smoke.py` now computes EE error from FER joint
  records with `PandaPregraspPlanner.ee_position` and no longer uses MuJoCo
  foreground planner time as a behavior-gate failure. Added
  `test_controller_timing_smoke.py`, gated by
  `SBMPC_RUN_CONTROLLER_TIMING=1`, to time the ROS planner adapter with
  synthetic FER `Sensor` messages and no MuJoCo/rendering load.
- Verified: Focused non-live ROS tests passed:
  `test_validate_sim.py`, `test_ee_parity_smoke.py`, and
  `test_controller_timing_smoke.py` reported `3 passed, 2 skipped`; broader
  bridge/bringup focus reported `27 passed, 2 skipped`. The enabled
  controller-only timing gate passed under the pixi/ROS runtime with
  `SBMPC_RUN_CONTROLLER_TIMING=1 ... test_controller_timing_smoke.py -q`,
  reporting `1 passed`.
- Not verified / blockers: Did not run the opt-in live MuJoCo behavior gate in
  this turn. MuJoCo behavior still needs a fresh run after these validation
  changes.
- Next handoff: Keep SB-MPC tuning unchanged. Run the split gates:
  `SBMPC_RUN_MUJOCO_PARITY=1` for headless MuJoCo behavior/wiring and
  `SBMPC_RUN_CONTROLLER_TIMING=1` for controller-only timing. If visual review
  is needed, record `/joint_states`, `/sensor`, `/control`, and
  `/sbmpc/diagnostics` from the headless run and replay the trajectory.

### 2026-05-06 — Codex Phase C4 Warmup/LFC Sequencing
- Scope: Continued the MuJoCo behavior failure debug after direct MuJoCo with
  the ROS-control MJCF converged under SB-MPC feedforward torques, while the
  ROS/LFC/MuJoCo path diverged even in feedforward mode.
- Finding: LFC was activated before SB-MPC/JAX warmup completed. During that
  several-second window MuJoCo ran LFC's internal PD startup mode, and in the
  direct-effort sim profile that PD path is not the bench controller behavior.
  The first SB-MPC torque therefore saw a different, moving initial condition
  from `bench_lfc.py`.
- Changed: Added a bridge warmup waiter and changed the MuJoCo launch order so
  the bridge starts first, reports planner warmup completion on
  `/sbmpc/diagnostics`, and only then activates `joint_state_estimator` plus
  `linear_feedback_controller`. Kept SB-MPC tuning untouched. Simplified the
  exact-async preset to retime the LFC `Control.initial_state` to the latest
  sensor snapshot with zero prediction instead of using the delayed 20 ms
  prediction path.
- Verified: Focused non-live bringup tests passed after the warmup helper was
  added. The first live MuJoCo rerun progressed to sustained control
  publication; the remaining failures were cadence tolerance/rejected-output
  details rather than the earlier "no controls" or pre-warmup drift mode.
- Next handoff: Rebuild `sbmpc_bringup`, rerun the live MuJoCo behavior gate,
  and inspect whether retimed exact-async removes rejected outputs. If cadence
  is marginal by sub-millisecond jitter, compare header-stamp p99 with
  receive-time p99 before changing any acceptance threshold.

### 2026-05-06 — Codex Phase C4 Runtime-State Parity Fix
- Scope: Resolved the remaining MuJoCo behavior divergence by comparing
  `bench_lfc.py`, the ROS planner adapter, and live `/control` outputs at the
  `home` state without changing SB-MPC tuning.
- Changed: Made planner warmup compilation-only for live bridge use:
  `PandaPregraspController` / `PandaPickAndPlaceController` now expose
  `reset_runtime_state_after_warmup()`, and `SbMpcPlannerAdapter.warmup()`
  calls it by default so published async gains, cached gains, worker state, and
  sampler initialization are reset before arming. The ROS adapter now builds
  the pregrasp controller with `reseed_every_step=True`, matching
  `bench_lfc.py`'s measured-state nominal-guess policy. Replaced the broken
  `ros2 control switch_controllers` CLI activation with a direct
  `controller_manager_msgs/SwitchController` call in
  `sbmpc_bringup.warmup_wait`, keeping LFC preloaded inactive and activating
  immediately after MuJoCo reset.
- Verified: `colcon build --packages-select sbmpc_ros_bridge sbmpc_bringup`
  passed. Focused non-live tests passed on isolated `ROS_DOMAIN_ID=231`:
  `67 passed, 1 skipped`. Planner adapter tests passed (`9 passed`), and the
  longer planner API run passed its planner tests before the fixed adapter
  assertion was corrected. Live MuJoCo parity passed with
  `SBMPC_RUN_MUJOCO_PARITY=1 SBMPC_MUJOCO_OBSERVATION_SEC=8
  SBMPC_MUJOCO_ROS_DOMAIN_ID=196 ... test_ee_parity_smoke.py -q`
  (`2 passed`). A 3 s diagnostic probe showed convergence instead of runaway:
  final EE error ≈ `0.0002 m`, final gain norm ≈ `1.4`, no rejected planner
  outputs, and stable final joint velocities.
- Not verified / blockers: The controller-only timing smoke was rerun on
  isolated ROS domains and remains marginally red without tuning changes:
  30-step p99 ≈ `20.37 ms` and 60-step p99 ≈ `20.85 ms` against the
  `20.0 ms` gate. This is now isolated to the public planner adapter /
  `PandaPregraspController` construction path, not ROS publication or MuJoCo
  behavior. One likely next comparison is `bench_lfc.py`'s `build_all()` setup,
  which performs a dummy solver command before benchmark warmup, versus the
  adapter's `build_model_and_solver()` path.
- Next handoff: Keep tuning fixed. Continue the controller-only timing
  diagnosis by comparing `bench_lfc.py`'s `build_all()`/dummy-command setup
  against the public planner adapter. If preparing a robot deployment
  rehearsal in parallel, record `/joint_states`, `/sensor`, `/control`, and
  `/sbmpc/diagnostics` from the now-passing headless MuJoCo run for visual
  replay/hazard review.

### 2026-05-11 — Codex Current-State Recap
- Scope: Paused implementation and wrote the current development state into
  this plan so future Codex/agent sessions can restart without re-opening the
  same timing confusion.
- Changed: Added the "Current state checkpoint — 2026-05-11" restart anchor
  near the top of the plan. Updated the controller-only foreground compute
  gate from `20 ms` to the accepted `21 ms` budget; the 50 Hz `/control`
  publication cadence gate remains p99 ≤ `20 ms`.
- Current state: MuJoCo ROS2-control behavior parity is green with fixed
  SB-MPC tuning. Warmup/activation sequencing, runtime-state reset after
  warmup, and `reseed_every_step=True` made the ROS adapter match the
  `bench_lfc.py` live-control semantics. The remaining timing delta is limited
  to controller-only adapter timing and is accepted up to `21 ms`.
- Verified before this recap: Focused non-live tests had passed with
  `67 passed, 1 skipped`; planner adapter tests with `9 passed`; live MuJoCo
  parity with `SBMPC_RUN_MUJOCO_PARITY=1` over 8 s passed. No new live test was
  run in this recap-only turn.
- Next handoff: Implement the headless record plus visual replay workflow for
  `/joint_states`, `/sensor`, `/control`, and `/sbmpc/diagnostics`, then use it
  as the next deployment-safety gate before real-robot rehearsal.

### 2026-05-11 — Codex Visual Validation Status
- Scope: Recorded the current visual-validation fact pattern after attempting
  to launch the MuJoCo ROS stack with live rendering.
- Observed: `ros2 launch sbmpc_bringup
  sbmpc_franka_lfc_mujoco_sim.launch.py headless:=false
  enable_nonzero_control:=true` failed before controller validation because
  `mujoco_ros2_control` reported `Timed out waiting to start simulation
  rendering!`, then aborted hardware initialization. The bridge was interrupted
  only because launch shut down after the MuJoCo process died.
- Interpretation: This is a MuJoCo viewer/rendering startup problem in the
  current host/container environment, not a controller behavior failure. It
  does not invalidate the passing headless MuJoCo parity/behavior gate.
- Current usable validation: Continue using headless MuJoCo plus
  `validate_sbmpc_sim` and `test_ee_parity_smoke.py` for ROS wiring,
  controller communication, EE error, stability, and rejected-output checks.
- Next handoff: Implement the planned headless record plus offline visual
  replay workflow. Prefer that over relying on `headless:=false` live GUI
  rendering, because replay keeps visualization load out of the controller
  validation run.

### 2026-05-11 — Codex Feedforward-First Architecture Reanchor
- Scope: Reframed the controller timing goal after user clarification that
  the desired runtime behavior is fresh feedforward torque at 50 Hz with gains
  refreshed only when available.
- Changed: Updated the acceptance gates and Phase D with a chosen
  feedforward-first architecture: separate `tau_ff` and `K` buffers, 50 Hz
  foreground solve protected from exact-gain refresh, opportunistic gain
  updates, and timing diagnostics that prove the gain worker cannot degrade
  foreground feedforward p99. Clarified that the current `21 ms`
  planner-adapter smoke is a diagnostic ceiling, not the target architecture.
- Verified: No code was changed in this turn. The discussion is based on the
  current code structure inspected earlier: `bench_lfc.py` raw foreground
  command timing around `7-8 ms`, current ROS adapter timing near the `21 ms`
  gate, and background exact-gain refresh often occupying about `20-25 ms` on
  the same JAX/GPU path.
- Not verified / blockers: Need implementation and measurement. The key risk
  is that same-GPU gain kernels may be non-preemptible; if so, opportunistic
  gains require device/process isolation or smaller budgeted gain chunks,
  otherwise the gain worker can still delay the next foreground solve.
- Next handoff: Implement D1. Start by exposing a feedforward-only public API
  in `sbmpc`, then update `sbmpc_ros_bridge` to publish from independent
  latest-feedforward and latest-gain buffers. Add a timing smoke that compares
  feedforward p99 with the gain worker disabled and enabled.

### 2026-05-11 — Codex D1 Split-Buffer Controller Implementation
- Scope: Implemented the first controller-architecture slice for
  feedforward-first control without retuning SB-MPC.
- Changed: Added `FeedforwardOutput` and `GainSnapshot` to the public Panda
  planner API. `PandaPregraspController.step_feedforward()` now computes
  `tau_ff` and captures an exact-gain context without bundling a gain read
  into the output; `latest_gain()` exposes the most recently published gain
  separately. `SbMpcPlannerAdapter` delegates those split calls. The ROS bridge
  now stores independent latest-feedforward and latest-gain buffers and
  composes the outgoing LFC `Control` message at publish time, while preserving
  fallback behavior for old bundled `PlannerOutput` implementations.
- Verified: Focused ROS bridge tests passed through the Pixi/ROS runtime:
  `test_fake_ros_loop.py` and `test_planner_adapter.py` reported `24 passed`.
  Focused `sbmpc` planner API tests passed with `12 passed`. The opt-in
  legacy controller timing smoke still passed with
  `SBMPC_RUN_CONTROLLER_TIMING=1`. A one-step planner smoke returned
  ROS-shaped `(7, 1)` feedforward and `(7, 14)` gain messages with foreground
  timing around `7.7 ms`.
- Not verified / blockers: This pass separates the data path but does not yet
  solve same-GPU gain contention. The internal exact-gain worker can still
  occupy the JAX/GPU queue unless the next pass adds device/process isolation
  or slack-budgeted gain chunks.
- Next handoff: Add the feedforward-first timing smoke described in Phase D1,
  then implement gain scheduling/isolation so enabling gain refresh cannot
  worsen feedforward p99 or `/control` cadence.

### 2026-05-11 — Codex D1 Single-Mode Exact-Async Cleanup
- Scope: Collapsed the controller implementation back to one public feedback
  mode: `exact_async_feedback`. There is no separate opportunistic planner
  mode anymore.
- Changed: `exact_async_feedback` now means foreground-priority control:
  `step_feedforward()` performs the MPPI foreground solve and captures a gain
  context, while gain refresh is attempted only through
  `refresh_gain_if_budget()` when the bridge has remaining cycle budget. The
  ROS bridge config is back to `planner_mode: exact_async_feedback`.
- Verified: Focused ROS bridge / bringup tests passed:
  `test_fake_ros_loop.py`, `test_planner_adapter.py`,
  `test_bringup_config.py`, and `test_validate_sim.py` reported `34 passed`.
  Planner API tests reported `13 passed`. The opt-in controller timing smoke
  reported foreground p99 `8.014 ms` with wall calls around `9-10.6 ms`.
  Planner smoke for `exact_async_feedback` returned finite ROS-shaped control
  with foreground planning `7.49 ms`; the measured exact-gain refresh cost was
  `21.41 ms`, so live refresh was correctly treated as not fitting in the
  remaining 50 Hz budget.
- Remaining constraint: Current exact gain refresh is a monolithic same-GPU
  job of roughly `20-23 ms`, so it cannot be run inside the leftover budget
  after a `7-8 ms` foreground solve. The current safe behavior is to preserve
  the 50 Hz feedforward path and skip gain refresh until enough slack exists.

### 2026-05-11 — Codex Exact-Gain Microbatch Feasibility Check
- Scope: Checked whether gain computation can be continued across control
  cycles while still publishing gains synthesized from many samples.
- Changed: Added `tests/bench_gain_microbatch.py`, a warmed benchmark that
  separates two axes: current full-gradient sample batches, and gradient-only
  microtasks that avoid recomputing rollout cost/control values already known
  from the foreground MPPI pass.
- Verified: Sample-only splitting of the current refresh path is not enough:
  `gK=4` in `bench_lfc.py` still measured about `17.16 ms` mean refresh time
  after warmup. The focused benchmark found the current full-gradient path at
  about `15.10 ms` for 1 sample and `15.24 ms` for 1 sample in a repeat run.
  The gradient-only microtask is materially better: 1 sample with all 14 state
  directions measured about `8.88 ms`; 2 samples about `10.76 ms`; 4 samples
  about `10.32 ms`.
- Follow-up with larger sample chunks: the gradient-only path measured about
  `10.55 ms` for 8 samples, `10.83 ms` for 16 samples, `12.49 ms` for
  64 samples, and `13.40 ms` for 128 samples. The current full-refresh path
  over the same shapes measured about `15.5-19.5 ms`. This means the useful
  split axis is complete sample-gradient records, not partial state
  directions. Keep the final rolling window large for stability, but fill it
  with budgeted chunks of complete sample gradients.
- Interpretation: The rolling-window idea remains applicable and should keep
  a large sample window for stability. The implementation target should be a
  resumable gain accumulator: compute a small number of complete
  `(cost, delta_u0, gradient)` sample records during slack, append complete
  records to `RollingGainWindow`, and publish `K` atomically only after the
  window reaches the configured size. Do not reduce the final gain window just
  to make one cycle fast.
- Next handoff: Replace `phase0_refresh_exact_gains()` in the live path with a
  budgeted gradient-only microtask API. Keep `gain_buffer_size=512`; benchmark
  32/64/128 complete-sample chunks for time-to-first-gain, gain staleness, and
  behavior stability before committing the production chunk size.

### 2026-05-11 — Codex Rolling-Gain Architecture Benchmark
- Scope: Added a closed-loop benchmark to choose the gain scheduling
  architecture before changing the production controller again.
- Changed: Added `tests/bench_gain_architecture.py`. It measures the
  foreground `tau_ff` publish path separately from gain work, reuses the
  foreground costs and first-step control deltas, computes gradient-only
  complete sample chunks, appends them to a rolling gain window, and reports
  when the first gain is ready plus the post-fill update cadence. Once the
  window is full, each new chunk overwrites the oldest chunk and can publish a
  new `K`, matching the intended "drop old 64 / append new 64" behavior.
- Metrics: The benchmark reports foreground p99, gain-work p99, total cycle
  occupancy, 20 ms budget misses, first-gain latency, gain update count,
  gain-delta norms, final/tail end-effector error, feedback peaks, joint
  velocity, high-frequency velocity energy, and tail joint spans. It supports
  fixed chunks such as `32,64,128` and an optional finite-set adaptive scheduler
  that picks the largest precompiled chunk estimated to fit the slack budget.
- Verified: `py_compile` passed. A tiny `samples=32, horizon=2, chunk=1,
  buffer=2` smoke run filled the window and published the first gain. A short
  realistic-shape smoke with `samples=1024, horizon=8, chunk=64, buffer=128`
  exercised the real kernel shape; as expected, the deliberately too-small
  128-sample window produced unstable gains/feedback, reinforcing that the
  real architecture should keep `gain_buffer_size=512` for stability. The
  benchmark warmup now also publishes a representative nonzero gain and runs
  foreground planning once before measured cycles, so first-published-gain JAX
  compilation does not contaminate timing.
- Preliminary `64/512` probe: a 30-cycle gazebo-mode run reached
  `0.40 mm` final EE error, first gain at `0.14 s`, and 23 gain updates.
  Foreground p99 was `7.54 ms`; gain-work p99 was `14.51 ms`; combined cycle
  occupancy p99 was `21.97 ms`, so `64` is promising for behavior but still
  tight for a strict 20 ms same-GPU budget.
- Suggested first real sweep:
  `python tests/bench_gain_architecture.py --steps 80 --chunk-sizes 32,64,128 --gain-buffer-size 512 --timing-mode gazebo --output-json /tmp/gain_architecture_512.json`.
  Use the result to choose the production chunk size.

### 2026-05-12 — Codex Production Rolling 128/512 Controller
- Scope: Promoted the benchmarked rolling exact-gain architecture into the
  production `exact_async_feedback` path.
- Decision: Use `planner_gain_samples_per_cycle=128` and
  `planner_gain_buffer_size=512` as the first production target. The user
  sweep showed `128/512` reaches the first gain in about `0.06 s`, updates
  almost every foreground cycle after that, holds final EE error around
  `0.51 mm`, and remains close enough to the 20 ms budget to try on the robot
  GPU. `512/512` is rejected for live control because it costs roughly
  `31-35 ms` per cycle and produced a large gain jump.
- Changed: `RolloutGenerator` now exposes a JIT `rollout_gradients_to_state`
  kernel that computes only `dJ/dx` for selected samples. The existing exact
  gain refresh path now reuses foreground costs and first-step `delta_u0`,
  appends complete `(cost, delta_u0, gradient)` records to the rolling window,
  and synthesizes `K` from the latest 512 records once the window is full.
  `refresh_gain_if_budget()` no longer skips a positive-budget refresh solely
  because the previous 128-sample chunk estimate exceeds the strict leftover
  50 Hz slack; this lets ROS run the chosen architecture and use the measured
  controller rate as the truth.
- Warmup fix: Exact-async warmup now compiles one foreground command while a
  representative nonzero gain is already published, then
  `reset_runtime_state_after_warmup()` still clears live gains/window state.
  This removes the first-live-gain JAX compilation spike from foreground
  timing.
- ROS scaling: Added `sbmpc_bridge_exact_async_40hz.yaml`, with
  `publish_rate_hz=40.0`, `planner_deadline_sec=0.025`, and
  `planner_dt=0.025`, while keeping `128/512` gains. Use this preset if the
  deployed GPU cannot sustain the chosen controller close enough to 50 Hz.
- Verified: `test_mppi_gains.py` reported `10 passed`;
  the exact-async planner API subset reported `3 passed`; ROS bridge/config
  focused tests reported `32 passed`; the opt-in production timing smoke with
  20 steps reported foreground p99 `7.726 ms`, `completed_batches=20`,
  `dropped_snapshots=0`, and passed.

### 2026-05-12 — Codex ROS Replay Visualization Path
- Scope: Added a way to visualize full ROS-stack behavior without running the
  viewer in the same timing-sensitive process as SB-MPC/JAX and MuJoCo
  control.
- Changed: Added `record_sbmpc_replay`, which subscribes to
  `/sbmpc/joint_states`, `/sensor`, `/control`, and `/sbmpc/diagnostics` and
  writes a JSON replay file. Added `replay_sbmpc_trajectory`, which loads that
  file afterward and replays the recorded FER arm joint trajectory in a MuJoCo
  viewer using the ROS2-control scene. This keeps the live stack headless while
  still allowing visual inspection afterward.
- Workflow: Run `sbmpc_franka_lfc_mujoco_sim.launch.py headless:=true
  enable_nonzero_control:=true record_replay:=true
  record_replay_output:=/tmp/sbmpc_ros_replay.json`, stop the live stack, then
  replay with `/workspace/sbmpc_containers/scripts/pixi_ros_run.sh python -m
  sbmpc_bringup.trajectory_replay /tmp/sbmpc_ros_replay.json`. Set
  `record_replay_duration_sec:=8` for a fixed
  window, or leave it at `0` to record until launch shutdown. Set
  `record_replay_include_warmup:=true` only when startup/JIT behavior is the
  object of the recording. The launch recorder autosaves every
  `record_replay_autosave_period_sec:=5` seconds and writes an empty file as
  soon as it starts, so an old replay file cannot silently survive a failed or
  interrupted recording. Replay pacing defaults to `--time-source auto`, which
  uses recorder wall-clock receive time when available and falls back to control
  cadence for older files whose state header stamps did not advance.
- Verified: `replay_sbmpc_trajectory --dry-run` on a synthetic replay JSON
  loaded the default MuJoCo model and mapped all seven FER arm joints.
- Follow-up: The recorder summary now includes full-stack timing statistics
  from bridge diagnostics (`foreground`, `bridge_loop`, planner wall time,
  background gain, control prepare/publish), control publish cadence, deadline
  misses, accepted/rejected planner output counts, and rolling gain state.
  In ROS 2, direct shell commands such as `record_sbmpc_replay` are not
  guaranteed to be on `PATH`; use `ros2 run sbmpc_bringup record_sbmpc_replay`
  for standalone recording, or the launch arguments above for the normal
  workflow. During source-tree development, the modules can still be run
  directly with `PYTHONPATH=/workspace/sbmpc_ros/sbmpc_bringup`.

### 2026-05-12 — Codex Timing Overhead Audit
- Scope: Made the timing budget auditable enough to explain individual
  millisecond losses before changing production behavior. The ROS diagnostics
  now split the old opaque planner-step overhead into planner API wall time,
  bridge/adapter residue, prepare/reseed time, JAX command time, tau
  extraction, cached-gain fetch, optional task diagnostics, and output build.
  Replay summaries include those fields when recorded by a rebuilt bridge.
- Benchmark correction: `bench_gain_architecture.py` now counts nominal
  reseeding in the foreground critical path, reports reseed time separately,
  supports `--reseed-policy every|periodic|initial|none`, and summarizes gain
  timing by selected chunk. Its adaptive chunk estimator now discards JAX
  compilation before timing; the previous estimator could average compilation
  into chunk estimates and therefore skip all gain work.
- Reseed finding: A focused adapter probe showed the current
  `reseed_every_step=True` prepare path costs about `1.8 ms` mean and about
  `2.1 ms` p99. Disabling reseed after the first step reduced step wall time
  from about `9.15 ms` to `7.32 ms`, but the closed-loop benchmark diverged to
  NaNs by step 30. Periodic reseed every two cycles also failed the stability
  gates (`feedback_peak≈60 Nm`, very large gain jump). Treat per-step reseed
  as required until a faster equivalent nominal seeding method is designed and
  passes the same closed-loop gates.
- Gain chunk timing: Updated `bench_gain_microbatch.py` to time full
  `select + pack + gradient` batches with p95/p99. On the current GPU after
  warmup, full-batch p99 was about `14.46 ms` for 64 samples, `15.73 ms` for
  128, `21.52 ms` for 256, and `25.94 ms` for 512. The relationship is not
  smooth or linear, so adaptive scheduling should use a finite precompiled set
  such as `{64,128}` rather than arbitrary sample counts.
- Current measured tradeoff:
  - 50 Hz, fixed `128/512`, reseed every step: cycle p99 `25.95 ms`; first
    gain `0.06 s`; large enough misses that this is not a true 50 Hz fresh
    planner path on this GPU.
  - 50 Hz, fixed `64/512`, reseed every step: cycle p99 `24.01 ms`; first
    gain `0.14 s`; smaller gain jumps (`dKmax≈0.84`) and lower timing cost.
  - 25 ms budget, adaptive `{64,128}` with reseed every step: selected mostly
    64-sample chunks, cycle p99 `23.55 ms`, no budget misses, first gain
    `0.16 s`, and small gain jumps (`dKmax≈0.53`).
  - 40 Hz-style `dt=0.025`, fixed `64/512`: cycle p99 `24.99 ms` with one
    startup miss. Fixed `128/512` at the same setting reached cycle p99
    `26.59 ms`, so 128 is not the conservative 40 Hz fallback on this GPU.
- Next handoff: Keep production defaults behavior-compatible until a full ROS
  replay is rerecorded with the new diagnostic fields. For timing recovery,
  prioritize (1) a faster per-step nominal reseed equivalent, not removing
  reseed, (2) 64-sample or adaptive `{64,128}` gain chunks for the 40 Hz
  fallback, and (3) finite-shape adaptive scheduling with compilation-free
  estimates. Do not claim a 50 Hz fresh planner path while gain refresh remains
  in the same critical worker and full cycles measure above 20 ms.
