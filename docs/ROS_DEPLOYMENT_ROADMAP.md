# ROS Deployment Roadmap for SB-MPC Panda

This document is a persistent handoff for future Codex instances. It captures the agreed plan for moving the current `sbmpc` controller from MuJoCo validation toward Gazebo and then a real Franka Panda using `linear-feedback-controller` as the low-level torque controller.

## Project Goal

Develop a real-robot pick-and-place stack where:

- `sbmpc` remains the algorithm repository.
- The SB-MPC/MPPI planner outputs feedforward joint torques and Riccati-like feedback gains.
- `linear-feedback-controller` runs the low-level torque loop through `ros2_control`.
- A new ROS repository, tentatively `sbmpc_ros`, bridges between LFC sensor messages and the `sbmpc` planner.
- Gazebo/Ignition validation is a mandatory gate before real-robot execution.

The desired final control split is:

```text
Franka hardware / Gazebo
  -> ros2_control
  -> linear_feedback_controller publishes Sensor at low-level rate
  -> sbmpc_ros_bridge receives Sensor at planner rate, calls sbmpc planner
  -> sbmpc_ros_bridge publishes Control(feedforward, feedback_gain, initial_state)
  -> linear_feedback_controller computes tau = feedforward + K * state_error
  -> ros2_control writes effort commands
```

## Current ROS Workspace Status

The active ROS workspace is now:

```bash
/workspace/ros2_ws
```

Develop the ROS repository from:

```bash
/workspace/ros2_ws/src/sbmpc_ros
```

Implemented packages:

- `sbmpc_ros_bridge`: message adapters, safety profiles, diagnostics, timer loop, planner smoke tooling.
- `sbmpc_bringup`: FER-adapted launch and configuration assets for Gazebo and real Franka bringup.

Important current local convention:

- The installed Franka stack in this environment exposes a `fer` robot description.
- The current bringup defaults therefore use runtime joint names `fer_joint1 ... fer_joint7`.
- Do not assume `panda_joint*` names at runtime. The bridge must use the exact joint names of the active ROS model in its `joint_names` parameter and in the `initial_state` snapshot sent back to LFC.

## Current State of `sbmpc`

Repository path used during development:

```bash
/home/msabbah/Desktop/sbmpc
```

Environment:

```bash
cd /home/msabbah/Desktop/sbmpc
direnv exec . pixi run -e cuda python -m pytest tests/test_mppi_gains.py tests/test_panda_pregrasp.py -q
```

Known current status:

- The algorithm stack is in `sbmpc`.
- The environment uses Nix/direnv to provide Pixi, then Pixi to provide the Python/CUDA stack.
- The controller uses JaxSim dynamics and MPPI-style control sampling.
- Gains are currently computed by a finite-difference approximation for real-time feasibility.
- MuJoCo validation exists for Panda pregrasp and a scripted pick-and-place state machine.
- With gains enabled, the target runtime is around 20 ms per planning call, sufficient for roughly 50 Hz planning.
- The current Gazebo/ROS deployment has not been implemented yet.

Important current files:

```text
sbmpc/panda_pregrasp.py
sbmpc/panda_pick_and_place.py
examples/panda_pregrasp.py
examples/panda_pick_and_place.py
tests/test_mppi_gains.py
tests/test_panda_pregrasp.py
```

Before starting ROS work, inspect these files and confirm the public planner API is stable enough to call from ROS. If it is not stable, create a small adapter in `sbmpc` rather than importing example scripts from ROS.

## External References To Recheck

These references informed the plan. Future Codex instances should re-open them if implementation details become ambiguous.

- `linear-feedback-controller`: https://github.com/loco-3d/linear-feedback-controller
- LFC ROS implementation: https://github.com/loco-3d/linear-feedback-controller/blob/main/src/linear_feedback_controller_ros.cpp
- LFC core control law: https://github.com/loco-3d/linear-feedback-controller/blob/main/src/lf_controller.cpp
- LFC `Control.msg`: https://github.com/loco-3d/linear-feedback-controller-msgs/blob/main/msg/Control.msg
- LFC `Sensor.msg`: https://github.com/loco-3d/linear-feedback-controller-msgs/blob/main/msg/Sensor.msg
- LFC Eigen/ROS conversions: https://github.com/loco-3d/linear-feedback-controller-msgs/blob/main/include/linear_feedback_controller_msgs/eigen_conversions.hpp
- Agimus demo 03 bringup: https://github.com/agimus-project/agimus-demos/blob/f07f6a28127420aeddee5c09d133d08b98a69e9b/agimus_demo_03_mpc_dummy_traj/launch/bringup.launch.py
- Agimus Franka LFC launch: https://github.com/agimus-project/agimus-demos/blob/f07f6a28127420aeddee5c09d133d08b98a69e9b/agimus_demos_common/launch/franka/franka_common_lfc.launch.py
- Agimus Franka LFC params: https://github.com/agimus-project/agimus-demos/blob/f07f6a28127420aeddee5c09d133d08b98a69e9b/agimus_demos_common/config/franka/linear_feedback_controller_params.yaml
- Agimus Franka controllers params: https://github.com/agimus-project/agimus-demos/blob/f07f6a28127420aeddee5c09d133d08b98a69e9b/agimus_demos_common/config/franka/controllers.yaml

Use Agimus as a bringup reference, not as a controller architecture reference. Do not depend on `agimus_controller_ros` for SB-MPC.

Description package decision:

- for this project, prefer `agimus-project/agimus-franka-description` over upstream `frankarobotics/franka_description`
- the user already relies on the Agimus description stack on the real `fer` robot
- the Agimus repository still exports the ROS package name `franka_description`
- because of that, the container image must replace upstream `franka_description` rather than install both at once
- keep launch code referring to `franka_description`; the container build decides which source tree provides that package name

## Milestone 4 Status

Milestone 4 was started with a new `sbmpc_bringup` package and a real planner smoke path.

What is implemented:

- `sbmpc_bringup/launch/sbmpc_franka_lfc_sim.launch.py`
- `sbmpc_bringup/launch/sbmpc_franka_lfc_real.launch.py`
- `sbmpc_bringup/config/franka_controllers.yaml`
- `sbmpc_bringup/config/franka_lfc_params.yaml`
- `sbmpc_bringup/config/sbmpc_bridge.yaml`
- FER-specific controller interface wiring based on the Agimus `joint_state_estimator` plus `linear_feedback_controller` pattern
- `sbmpc_ros_bridge.planner_smoke` for validating real `sbmpc` + JAX planner calls from the ROS-side environment

Verified commands:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```

Current test result:

- `39 tests, 0 errors, 0 failures, 0 skipped`

Verified planner smoke:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke --joint-set fer
```

Observed result:

- planner call succeeds through the ROS bridge adapter
- `Control.initial_state.joint_state.name` is `fer_joint1 ... fer_joint7`
- `feedback_gain` shape is `(7, 14)`
- `feedforward` shape is `(7, 1)`
- measured planning time was about `19.6 ms`

Current Gazebo blocker:

```bash
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch sbmpc_bringup sbmpc_franka_lfc_sim.launch.py \
  gz_args:='empty.sdf -r -s' use_rviz:=false
```

Observed result:

- the FER robot entity spawns successfully
- `/joint_states` publishes `fer_joint1 ... fer_joint7`
- Gazebo reports: `A link named fer_link4 has invalid inertia`
- `/controller_manager/list_controllers` appears in the graph but is not contactable
- `joint_state_broadcaster` spawner times out, so the LFC stack never activates

Interpretation:

- the SB-MPC bridge package, bringup package, and planner import path are working
- the current simulation failure is in the installed FER Gazebo model / controller-manager path, not in the bridge-to-planner integration
- next work should inspect or patch the FER inertial description used by Gazebo before treating Milestone 4 Gazebo validation as complete

## Key LFC Interface Facts

The ROS bridge must publish `linear_feedback_controller_msgs/msg/Control` and subscribe to `linear_feedback_controller_msgs/msg/Sensor`.

`Control` contains:

```text
std_msgs/Header header
std_msgs/Float64MultiArray feedback_gain
std_msgs/Float64MultiArray feedforward
Sensor initial_state
```

`Sensor` contains:

```text
std_msgs/Header header
geometry_msgs/Pose base_pose
geometry_msgs/Twist base_twist
sensor_msgs/JointState joint_state
Contact[] contacts
```

For fixed-base Panda:

- `robot_has_free_flyer: false`.
- Use 7 arm joints only for the LFC-controlled arm.
- Internal state dimension is `nx = 14`, ordered as `[q1..q7, v1..v7]`.
- Control dimension is `nu = 7`, ordered as joint efforts `[tau1..tau7]`.
- `feedback_gain` must be a `(7, 14)` matrix.
- `feedforward` must be a `(7,)` vector.
- `initial_state` must be the exact `Sensor` message used to compute the current control solution.
- Gripper control is separate and should not be included in the LFC gain matrix.

LFC publishes and subscribes on relative topics from the controller namespace:

```text
sensor
control
```

So, depending on the controller namespace, likely full topics are similar to:

```text
/linear_feedback_controller/sensor
/linear_feedback_controller/control
```

Verify with `ros2 topic list` during implementation.

## Gain Sign Convention

This is a non-negotiable validation item.

The LFC computes a state difference in the direction `desired - measured` and applies:

```text
control = feedforward + feedback_gain * diff_state
```

The current SB-MPC finite-difference gain may naturally represent the derivative of the MPPI control with respect to the measured state. If so, the gain sent to LFC may need to be negated.

Before using gains in Gazebo or on hardware, implement an explicit sign test:

1. Choose a nominal state `x0`.
2. Perturb one joint position or velocity in the measured state.
3. Apply the same `desired - measured` convention used by LFC.
4. Verify the resulting feedback torque is stabilizing.
5. Repeat for position and velocity columns.

Do not infer the sign from naming. Test it.

## Recommended Repository Split

Use three repositories, each with a strict responsibility:

```bash
/home/msabbah/Desktop/sbmpc        # algorithm and non-ROS validation
/home/msabbah/Desktop/sbmpc_ros          # ROS bridge, bringup, Gazebo assets
/home/msabbah/Desktop/sbmpc_containers   # Docker/devcontainer/compose deployment
```

Responsibilities:

- `sbmpc`: algorithm, dynamics, costs, sampling, gains, MuJoCo/JaxSim validation. Its Python package is currently named `sbmpc`.
- `sbmpc_ros`: ROS 2 integration, message conversion, launch files, Gazebo validation, robot deployment hooks, safety gates, diagnostics.
- `sbmpc_containers`: Dockerfiles, devcontainer files, compose files, image build scripts, dependency pinning for ROS/LFC/Franka/Gazebo/JAX deployments.

Do not duplicate the planner logic in `sbmpc_ros`. Import `sbmpc` as a Python dependency during development, probably with an editable/path install inside the planner container.

Do not embed Dockerfiles into `sbmpc_ros`. The ROS repo should remain buildable as a normal ROS 2 workspace package. The container repo should decide how to mount/build/install `sbmpc_ros` and `sbmpc`.

Suggested `sbmpc_ros` layout:

```text
sbmpc_ros/
  README.md
  sbmpc_ros_bridge/
    package.xml
    setup.py
    sbmpc_ros_bridge/
      __init__.py
      lfc_bridge_node.py
      planner_adapter.py
      lfc_msg_adapter.py
      joint_mapping.py
      safety.py
      diagnostics.py
  sbmpc_bringup/
    package.xml
    launch/
      sbmpc_franka_lfc_sim.launch.py
      sbmpc_pick_place_gazebo.launch.py
      sbmpc_franka_lfc_real.launch.py
      sbmpc_pick_place_real.launch.py
    config/
      franka_controllers.yaml
      franka_lfc_params.yaml
      sbmpc_bridge.yaml
      safety.yaml
      task.yaml
  sbmpc_gazebo/
    package.xml
    worlds/
      panda_pick_place.world.sdf
    models/
    launch/
      gazebo_pick_place_world.launch.py
  tests/
```

Suggested `sbmpc_containers` layout:

```text
sbmpc_containers/
  README.md
  docker/
    control-dev.Dockerfile
    planner-cuda.Dockerfile
    realtime-control.Dockerfile
  compose/
    gazebo.yaml
    robot.yaml
    dev.yaml
  repos/
    franka_lfc.repos
    sbmpc_ros.repos
  scripts/
    build_control_dev.sh
    build_planner_cuda.sh
    build_realtime_control.sh
    run_gazebo.sh
    run_planner.sh
  devcontainer/
    devcontainer.json
```

This layout can be simplified, but keep container/deployment logic out of `sbmpc_ros`.

## Container Repository Plan

The Agimus dev-container repository was inspected through Git because the GitLab web UI is protected by Anubis. The relevant repository is:

```bash
https://gitlab.laas.fr/agimus-project/agimus_dev_container.git
```

Important files in that repository:

```text
.devcontainer/Dockerfile.base
.devcontainer/control/Dockerfile
.devcontainer/Dockerfile.realtime
compose.yaml
```

Assessment of Agimus images:

- `humble-devel-control` is a useful development/Gazebo/LFC base. It starts from ROS 2 Humble desktop, installs `ros-gz`, builds Pinocchio, Crocoddyl, HPP-related dependencies, imports Franka dependencies from `franka.repos`, and imports LFC dependencies from `agimus_dev.repos`.
- `agimus_dev.repos` includes `linear-feedback-controller` v3.0.1 and `linear-feedback-controller-msgs` v1.1.1, plus Agimus controller packages that we do not want to depend on.
- `franka.repos` includes Agimus forks of `libfranka`, `franka_ros2`, `franka_description`, and `ros2_net_ft_driver`.
- `humble-devel-realtime-control` is much smaller and is aimed at the low-level real-time control side. It builds LFC, LFC messages, libfranka, Franka ROS 2 components, and `agimus_demos_common`.

Conclusion:

- The Agimus `control` image likely suits the Gazebo/LFC/Franka development side, but not the SB-MPC planner side by itself.
- It does not provide the full JAX/JaxSim/CUDA/Pixi or `sbmpc` planner environment we need.
- It is also heavy and Agimus-opinionated, so it should be treated as a base/reference, not as architecture we adopt wholesale.

Recommended image split for `sbmpc_containers`:

1. `sbmpc-control-dev`

   Purpose: Gazebo, Franka ROS 2, LFC, RViz/PlotJuggler, controller-manager debugging.

   Initial practical base:

   ```dockerfile
   FROM gitlab.laas.fr:4567/agimus-project/agimus_dev_container:humble-devel-control
   ```

   Use this first to move quickly. Later, if image size or Agimus coupling becomes a problem, replace it with a minimal Dockerfile inspired by Agimus.

2. `sbmpc-planner-cuda`

   Purpose: run `sbmpc_ros_bridge` and the SB-MPC planner with GPU acceleration.

   Requirements:

   - ROS 2 Humble Python runtime.
   - `linear_feedback_controller_msgs` Python message package.
   - CUDA-compatible JAX.
   - JaxSim and current `sbmpc` dependencies.
   - Editable/path installs of `/workspace/sbmpc` and
     `/workspace/ros2_ws/src/sbmpc_ros` (legacy `/workspace/sbmpc_ros`
     compatibility mount optional).
   - NVIDIA container runtime support.

   This image should not need the full LFC controller or Gazebo stack unless we decide to run everything monolithically for early debugging.

3. `sbmpc-realtime-control`

   Purpose: real robot low-level side if using an auxiliary real-time computer.

   Initial practical base:

   ```dockerfile
   FROM gitlab.laas.fr:4567/agimus-project/agimus_dev_container:humble-devel-realtime-control
   ```

   It should contain only what is needed for Franka hardware, `ros2_control`, LFC, joint-state estimator, and gripper support. It should not contain the JAX planner.

4. Optional `sbmpc-gazebo-monolithic`

   Purpose: one-container debug mode for local Gazebo validation.

   This can combine control-dev and planner dependencies for convenience, but it should not be the final deployment model if it becomes too large or fragile.

Compose strategy:

- `compose/gazebo.yaml`: launches Gazebo/LFC/control services plus planner service on host networking. Use X11/DRI mounts for Gazebo rendering and NVIDIA runtime for planner GPU access.
- `compose/robot.yaml`: launches planner-side container and connects over ROS 2 DDS to the real-time control computer running LFC.
- `compose/dev.yaml`: developer shell with all repositories mounted for iterative work.

Initial recommendation:

- Do not fork Agimus dev-container directly as the long-term source of truth.
- Create our own `sbmpc_containers` repository.
- For the first Gazebo milestone, derive `sbmpc-control-dev` from the Agimus `humble-devel-control` image to avoid spending days rebuilding Franka/LFC/Gazebo dependencies.
- In parallel, make `sbmpc-planner-cuda` explicit and minimal because this is where our custom JAX/JaxSim/Pixi stack matters.
- Once Gazebo works, decide whether to keep deriving from Agimus or replace the control image with a minimal Dockerfile copied conceptually from their `Dockerfile.realtime`/`Dockerfile.control`.

## Roadmap

### Milestone 0: Stabilize `sbmpc` Planner API

Goal: make the controller callable from ROS without importing example scripts.

Expected API shape:

```text
planner.step(
  q: np.ndarray shape (7,),
  v: np.ndarray shape (7,),
  phase: Phase,
  object_pose: optional task context,
  target_pose: optional task context,
) -> PlannerOutput
```

`PlannerOutput` should contain:

```text
tau_ff: shape (7,)
K: shape (7, 14)
phase: current/next phase
gripper_command: open/close/width if needed
diagnostics: timing, cost, gain norm, torque norm, phase metrics
```

Acceptance criteria:

- Existing `sbmpc` tests still pass.
- MuJoCo example still runs.
- Planner call is deterministic enough for repeated ROS calls.
- No ROS dependencies are introduced in `sbmpc`.

### Milestone 1: Create `sbmpc_ros` Skeleton and LFC Message Adapter

Goal: prove that we can convert between LFC messages and SB-MPC arrays correctly.

Implement:

- ROS 2 Python package skeleton for `sbmpc_ros_bridge`.
- `joint_mapping.py` for strict joint-name ordering.
- `lfc_msg_adapter.py` for `Sensor -> PlannerInput` and `PlannerOutput -> Control`.
- Unit tests with synthetic `Sensor` messages.

Acceptance criteria:

- Joint order is tested and fails loudly on missing/extra/shuffled names unless explicitly remapped.
- `feedback_gain` layout is tested as `(7, 14)` with row-major `Float64MultiArray` data.
- `feedforward` layout is tested as either the LFC-accepted vector convention or a `(7, 1)` matrix convention, based on actual LFC conversion behavior.
- `initial_state` is exactly the sensor used for planning.
- NaNs and wrong sizes are rejected.

### Milestone 2: Gain Sign and Safety Unit Tests

Goal: prevent unstable feedback before Gazebo.

Safety philosophy for this milestone and beyond:

- Keep a small `always_on` bridge safety layer in every deployment. This covers
  message validity, non-finite rejection, gain sign convention, and stale
  control checks.
- Treat `bringup_limits` as optional and tunable. Torque clipping, gain-norm
  clipping, and conservative fallback behavior are appropriate during early
  robot testing, but they should not become accidental permanent restrictions on
  the controller.
- Treat deadline and performance signals as `monitoring_only` by default unless
  the deployment explicitly chooses a fail-closed behavior.
- Do not rely on Franka hardware limits alone. Hardware limits protect the
  robot at the low level, while the bridge safety layer is there to catch
  software and integration mistakes before they reach those limits.

Implement tests for:

- LFC sign convention: desired-minus-measured.
- Whether SB-MPC gain must be sent as `K` or `-K`.
- Torque clipping.
- Gain norm clipping.
- Stale control detection.
- Planner deadline miss handling.
- Non-finite output rejection.

Acceptance criteria:

- A perturbed position generates stabilizing feedback torque in the test convention.
- A perturbed velocity generates damping feedback torque in the test convention.
- Unsafe outputs are blocked before publishing.

### Milestone 3: Fake ROS Loop

Goal: test bridge timing without Gazebo.

Current implementation note:

- The fake-loop milestone is implemented in `sbmpc_ros_bridge` with a timer-based
  `sbmpc_lfc_bridge_node`, JSON diagnostics topic, and integration coverage in
  `test/test_fake_ros_loop.py`.

Implement:

- Fake LFC `Sensor` publisher.
- Fake `Control` subscriber.
- `sbmpc_lfc_bridge_node` running at 50 Hz.
- Diagnostics topic or console summary.

Acceptance criteria:

- Bridge publishes valid `Control` at target rate.
- JAX/JaxSim warmup happens before nonzero commands are allowed.
- No nonzero control is published until a valid sensor message is received.
- Missed deadlines are counted.
- The node can run for several minutes without memory or timing degradation.

### Milestone 4: Gazebo LFC Bringup

Goal: reproduce the Agimus-style Franka/LFC stack under our own clean launch files.

Use the Agimus structure conceptually:

- `controller_manager` update rate: 1000 Hz.
- `joint_state_broadcaster` active.
- `gripper_action_controller` active.
- `joint_state_estimator` loaded.
- `linear_feedback_controller` loaded and activated.
- LFC configured with fixed-base Panda, 7 moving joints, effort command interfaces.
- `sbmpc_lfc_bridge_node` starts after LFC sensor topic exists.

Agimus reference mapping that should guide our implementation:

- In `agimus_demos_common/launch/franka/franka_common_lfc.launch.py`, Agimus
  does not implement a custom controller manager. It includes its common Franka
  launch and passes two extra controller names:
  `linear_feedback_controller` and `joint_state_estimator`.
- Agimus injects those controllers through:
  `external_controllers_params` and `external_controllers_names`, rather than by
  changing the LFC controller implementation itself.
- The important LFC controller parameters in
  `agimus_demos_common/config/franka/linear_feedback_controller_params.yaml` are:
  `moving_joint_names`, `chainable_controller.command_interfaces`,
  `joint_velocity_filter_coefficient`, `pd_to_lf_transition_duration`,
  `remove_gravity_compensation_effort`, and `robot_has_free_flyer: false`.
- The important controller-manager parameters in
  `agimus_demos_common/config/franka/controllers.yaml` are:
  `controller_manager.update_rate: 1000`,
  `joint_state_broadcaster`, and `gripper_action_controller`.
- The local stack available in this repository differs slightly from Agimus:
  we have upstream `franka_bringup/launch/franka.launch.py` and
  `franka_gazebo_bringup`, not `franka_common.launch.py`. So our clean-room
  `sbmpc_bringup` should reproduce the same wiring by:
  1. including `franka_bringup` or `franka_gazebo_bringup`,
  2. supplying our own controller YAML that extends Franka's defaults with
     `joint_state_estimator` and `linear_feedback_controller`,
  3. spawning those controllers explicitly, and
  4. starting `sbmpc_lfc_bridge_node` only after the LFC `sensor` topic is live.
- Use Agimus only as a reference for the launch/config split and the order of
  controller activation. Do not depend on `agimus_controller_ros`.

Acceptance criteria:

- `ros2 topic list` shows LFC sensor and control topics.
- LFC publishes `Sensor` with 7 joint positions, velocities, and efforts.
- Bridge publishes `Control` with valid feedforward and gain layout.
- With zero gain and zero feedforward, the system does not crash.
- With safe hold behavior, the robot remains stable.

### Milestone 5: Gazebo PREGRASP Validation

Goal: validate the ROS/LFC/SB-MPC loop on a simple task before full pick-and-place.

Current implementation note:

- The local LFC stack publishes `Sensor` on `/sensor` and consumes `Control` on
  `/control` using best-effort QoS. The bridge must match those endpoints; the
  earlier `/linear_feedback_controller/{sensor,control}` assumption is not
  correct for this stack.
- `sbmpc_bringup/config/sbmpc_bridge_milestone5_feedforward.yaml` and
  `sbmpc_bringup/config/sbmpc_bridge_milestone5_feedback.yaml` start disarmed
  with `enable_nonzero_control: false`. Arm the bridge explicitly with:
  `ros2 param set /sbmpc_lfc_bridge_node enable_nonzero_control true`.
- In one headless validation run, the feedforward PREGRASP path reached
  `state="running"` with zero gain, nonzero `/control` feedforward, and a
  measured position error drop from about `0.43` to about `0.27`.
- In one headless validation run, the finite-gain PREGRASP path also reached
  `state="running"` with nonzero gain norm and a measured position error near
  `0.09`.
- The current bridge is still the simple single-step replanning version. The
  next likely refinement is an Agimus-style buffered receding-horizon layer,
  because the finite-gain run observed planning times around the 20 ms deadline
  and accumulated deadline misses.

Test sequence:

1. PREGRASP with `K = 0`, feedforward only.
2. PREGRASP with clipped finite `K`.
3. PREGRASP repeated from several initial configurations.

Acceptance criteria:

- End-effector reaches pregrasp target within a small tolerance.
- No NaN commands.
- No torque spikes above safety limits.
- Planner runtime remains near target.
- LFC remains active.
- The robot does not oscillate or diverge.

### Milestone 6: Gazebo Full Pick-and-Place Validation

Goal: prove the full phase machine works in Gazebo before hardware.

This must use actual Gazebo object pose for validation. Do not only rely on the scripted object state from MuJoCo.

Test sequence:

- Spawn Panda with gripper.
- Spawn cube.
- Spawn target/place marker.
- Run full phase machine: PREGRASP, DESCEND, CLOSE, LIFT, TRANSPORT, PLACE, OPEN, RETREAT, DONE.
- Control gripper through the ROS gripper action/controller, not through the LFC arm torque path.

Acceptance criteria:

- Cube is actually grasped in Gazebo.
- Cube is actually moved to the target region.
- Final placement error is logged.
- All phase transitions are logged.
- No phase gets stuck.
- 20 repeated trials pass with acceptable final object error.
- Any failure produces enough logs to diagnose: phase, EE pose, object pose, torque norm, gain norm, planner time, latest LFC sensor time.

### Milestone 7: Real Robot Minimal Gate

Gazebo passing does not remove all hardware risk. It should compress the real-robot testing phase, not eliminate it.

Minimum real-robot gate:

1. Launch LFC and bridge with zero nonzero commands disabled.
2. Confirm LFC `Sensor` is received and joint order is correct.
3. Publish zero-control `Control` and verify no unexpected motion.
4. Enable a tiny safe hold or tiny PREGRASP motion with severe torque and gain limits.
5. Run full pick-and-place with reduced speed, reduced gain, and operator ready for emergency stop.
6. Only then move to normal settings.

Acceptance criteria:

- The robot does not move unexpectedly at startup.
- Emergency stop and fallback behavior are known before nonzero commands.
- Joint order is verified against live robot state.
- Torque limits and watchdog are active.
- Full task only runs after the minimal gate passes.

## Safety Requirements

These are mandatory before commanding nonzero torque in Gazebo or on hardware:

- JAX/JaxSim warmup completed before enabling control.
- No nonzero commands before first valid LFC `Sensor` message.
- Strict joint-name and joint-order validation.
- Strict shape validation for `feedforward` and `feedback_gain`.
- Non-finite values are rejected.
- `always_on` checks stay enabled in all deployments.
- `bringup_limits` remain explicitly optional and tunable.
- Torque magnitude limits when conservative bringup limits are enabled.
- Torque rate limits if feasible.
- Gain norm limits when conservative bringup limits are enabled.
- Stale-control timeout.
- Planner deadline miss counter and configurable fail-closed behavior.
- Safe fallback mode: zero gain and safe feedforward/hold or disabled output.
- Explicit enable flag required before publishing nonzero commands.
- Gripper commands gated by phase and safety state.
- Real robot launch must default to conservative limits.

## Development Rules For Future Codex Instances

When continuing this work:

1. Start by reading this document.
2. Inspect the current `sbmpc` status and tests.
3. Do one milestone at a time.
4. Do not jump directly to real robot code before Gazebo validation exists.
5. Do not import Agimus controller code as a dependency unless the user explicitly changes the plan.
6. Use Agimus only as a reference for launch structure and Franka/LFC wiring.
7. Keep algorithm code out of `sbmpc_ros` except for thin adapters.
8. Preserve a clear test command for every milestone.
9. Prefer small, testable files over a monolithic bridge node.
10. Report exact commands run and exact pass/fail results.
11. Preserve the bridge safety split:
    `always_on` for validity/sign/staleness, `bringup_limits` for optional
    conservative caps, and `monitoring_only` for deadline/performance signals.
12. Do not turn temporary bringup limits into permanent hidden restrictions
    unless the user explicitly asks for that tradeoff.

## Suggested First Prompt For A New Codex Instance

The user can paste this into a fresh Codex session:

```text
We are working on SB-MPC for Franka Panda. Read /workspace/sbmpc/docs/ROS_DEPLOYMENT_ROADMAP.md first. The algorithm repo is /workspace/sbmpc. We now want to implement the next milestone only: create /home/msabbah/Desktop/sbmpc_ros with the ROS 2 bridge skeleton and LFC Sensor/Control message adapter tests. Do not implement Gazebo yet. Keep sbmpc as the algorithm dependency and do not depend on agimus_controller_ros. Validate joint order, message shapes, initial_state copying, gain sign convention, and safety rejection of invalid outputs.
```

## Useful Commands

Check current algorithm tests:

```bash
cd /home/msabbah/Desktop/sbmpc
direnv exec . pixi run -e cuda python -m pytest tests/test_mppi_gains.py tests/test_panda_pregrasp.py -q
```

Run the current MuJoCo/GPU examples, if available:

```bash
cd /home/msabbah/Desktop/sbmpc
direnv exec . pixi run -e cuda python examples/panda_pregrasp.py --gains
direnv exec . pixi run -e cuda python examples/panda_pick_and_place.py --gains
```

Inspect future ROS topics once Gazebo/LFC is running:

```bash
ros2 topic list
ros2 topic echo /linear_feedback_controller/sensor --once
ros2 topic echo /linear_feedback_controller/control --once
ros2 control list_controllers
ros2 control list_hardware_interfaces
```

Exact topic names may differ depending on namespace. Verify during implementation.
