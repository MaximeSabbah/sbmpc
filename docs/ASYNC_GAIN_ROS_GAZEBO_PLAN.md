# Async-Gain ROS/Gazebo Deployment Plan

## Goal

Land the Panda SB-MPC controller with MPPI feedforward, background exact gain
updates, and `linear-feedback-controller` feedback in Gazebo through
`sbmpc_ros`, then use the same architecture for real robot deployment.

The first validation target is the existing Panda pregrasp task. This is the
smallest task that exercises the planner, LFC message path, timing, gain
publication, and shutdown behavior without adding gripper sequencing or object
contact as extra variables.

The end goal is the full cube pick-and-place task already present in `sbmpc`:
pregrasp, descend, grasp, lift, transport, place, and release. Pregrasp is only
the first gate. The architecture and diagnostics must be built so the same
foreground MPPI plus background gain plus LFC path extends to the full task.

The MuJoCo benchmark is the algorithm reference. Gazebo is not meant to retune
or re-prove the dynamics model; it is the integration layer used to validate
the full ROS control path:

```text
Gazebo or Franka hardware
  -> ros2_control
  -> linear_feedback_controller Sensor
  -> sbmpc_ros_bridge
  -> sbmpc MPPI foreground planner
  -> background exact-gain worker
  -> linear_feedback_controller Control
  -> ros2_control effort command
```

## References

Keep this plan aligned with:

- `docs/ASYNC_GAIN_BACKGROUND_PLAN.md`
- `docs/ROS_DEPLOYMENT_ROADMAP.md`
- `sbmpc_ros/README.md`
- `sbmpc/tests/bench_lfc.py`
- `sbmpc_ros/sbmpc_ros_bridge/sbmpc_ros_bridge/lfc_bridge_node.py`
- `sbmpc_ros/sbmpc_ros_bridge/sbmpc_ros_bridge/planner_adapter.py`

## Success Criteria

Every validation stage must report these quantities separately.

Validation is staged:

1. Pregrasp first, in MuJoCo and then Gazebo.
2. Full pick-and-place next, in MuJoCo and then Gazebo.
3. Real robot only after both staged validations are understood and clean.

### Controller Timing

The foreground controller path must stay inside the 20 ms budget.

The measured foreground time includes:

- sensor-to-state conversion
- MPPI rollout and action selection
- gain snapshot selection and packaging
- `Control` message construction and publication

The measured foreground time excludes:

- background exact-gradient computation
- rolling gain-window synthesis
- first-call JAX/XLA compilation
- one-time warmup

Pass condition:

- after warmup, foreground controller execution is at or below 20 ms for the
  reference 50 Hz configuration
- any deadline miss must be visible in diagnostics and treated as a blocker
  before robot deployment

Background gain timing is still recorded, but it is not part of the 20 ms
foreground budget. The relevant background criteria are bounded queue depth,
bounded gain age, no worker errors, and stable published gains.

### Task Error

For the pregrasp task, the end-effector task error must be less than 1 mm in
the validated reference run.

Pass condition:

- final task-space position error <= `0.001 m`
- tail task-space position error remains <= `0.001 m`
- no late drift after convergence

For the full pick-and-place task, the same 1 mm position-error target should be
used for the commanded task-space waypoints where it is physically meaningful.
Additional phase-specific checks are required:

- grasp phase reaches a stable gripper/object pose before lift
- object lift clears the table without large joint oscillations
- transport remains stable with the object attached
- place phase reaches the release pose without late drift
- release does not destabilize the arm

The user will also assess the result visually in MuJoCo and Gazebo. The numeric
gate is still required because feedforward alone can look good in the current
pregrasp setup.

### Gain Stability

The published gain must be stable enough for LFC use.

Pass condition:

- gain matrix is always finite
- no NaN or Inf reaches ROS messages
- gain norm has no late spikes after the rolling window is full
- published gain age remains bounded
- dropped snapshot count does not grow without bound
- background worker reports no error
- feedback torque contribution remains small and smooth near convergence

The exact numeric stability threshold can be tightened after the first Gazebo
runs, but the validation output must always include gain norm, feedback torque
norm, gain age, rolling-window fill, completed gain batch count, and dropped
snapshot count.

## Current State

`sbmpc` already has the core algorithm pieces:

- MPPI foreground rollout.
- Buffered exact-gain snapshots.
- Rolling gain window with `K` samples per processed batch and `M` retained
  processed samples.
- Same-process background exact-gain worker.
- MuJoCo LFC benchmark presets for feedforward, finite-difference feedback,
  Phase 0 exact-gain probing, and background exact feedback.

`sbmpc_ros` already has the ROS integration skeleton:

- LFC `Sensor` to planner input adapter.
- planner output to LFC `Control` adapter.
- gain sign conversion for LFC, where LFC applies gains to
  `desired - measured`.
- bridge diagnostics.
- Gazebo and real launch files.
- validation utilities.

Main gap:

- ROS still calls the planner through a synchronous `step()` path and does not
  expose the async exact-gain lifecycle as a clean public API.

## Architecture Decisions

### Public Planner Modes

Expose a small number of planner modes instead of many independent runtime
flags:

- `feedforward`: MPPI feedforward only, zero gain sent to LFC.
- `fd_feedback`: MPPI feedforward plus finite-difference gain.
- `exact_async_feedback`: MPPI feedforward plus background exact gain.

Detailed tuning stays in YAML.

### Async Exact Feedback Semantics

For `exact_async_feedback`:

- the foreground planner runs MPPI and publishes `tau_ff` every cycle
- the foreground planner captures a gain snapshot every cycle
- the background worker computes exact gradients for the selected `K` samples
- the rolling gain window publishes a new gain once `M` processed samples are
  available
- before the first gain is ready, LFC receives zero gain
- the controller never blocks waiting for the background worker

### Gazebo Semantics

Gazebo validation should mirror the MuJoCo LFC benchmark:

- same sign convention
- same feedforward and feedback message shapes
- same planner rate target
- same joint ordering
- same initial-state snapshot semantics

If Gazebo behaves differently from MuJoCo, first suspect ROS integration,
message semantics, timing, controller configuration, or Gazebo model setup. Do
not retune the algorithm before those layers are checked.

## Required `sbmpc` Changes

### 1. Public Async-Gain Lifecycle

Add public methods around the existing controller worker:

- `start_background_gains(reset_published_gain=True)`
- `stop_background_gains(wait=True)`
- `background_gain_status()`
- `reset_published_gains()`
- `close()`

These should wrap the existing exact worker methods and be safe to call even
when async exact gains are disabled.

### 2. Planner API Support

Extend the Panda planner API so ROS does not need private controller details.

The planner controller should support:

- `gain_mode`
- `gain_samples_per_cycle`
- `gain_buffer_size`
- `start()`
- `close()`
- `step()`
- `diagnostics`

For async exact mode, `step()` must call the foreground command path with:

```text
update_gains=False
capture_gain_context=True
```

For finite-difference mode, `step()` keeps the current synchronous gain path.

For feedforward mode, gains are disabled and the returned gain matrix is zero.

### 3. Diagnostics

Extend planner diagnostics with:

- foreground planning time
- gain norm
- feedback torque norm if available
- async worker running flag
- async worker error
- gain age in cycles
- rolling-window fill
- completed gain batch count
- dropped snapshot count
- background gain timing from the latest completed batch

The 20 ms pass/fail decision must use foreground planning time only.

### 4. Tests

Add or keep tests for:

- async worker starts and stops cleanly
- `close()` is idempotent
- async exact step does not compute exact gradients on the foreground path
- zero gain is returned before the rolling window is full
- nonzero finite gain is returned after the rolling window is full
- background worker error is surfaced in diagnostics

## Required `sbmpc_ros` Changes

### 1. YAML-Level Runtime Modes

Add bridge parameters:

- `planner_mode`
- `planner_gain_samples_per_cycle`
- `planner_gain_buffer_size`
- `planner_gain_max_age_cycles` if needed

Keep legacy detailed parameters only where they are still useful for tests.
The default Gazebo configs should be readable without many tuning flags.

### 2. Planner Adapter Lifecycle

`SbMpcPlannerAdapter` should expose:

- `start()`
- `close()`
- `step()`
- `warmup()`
- `diagnostics_snapshot()` if needed

The bridge node must call:

- `planner.start()` after construction or after warmup, depending on the final
  implementation
- `planner.close()` in shutdown, before destroying the ROS node

### 3. Bridge Diagnostics

Publish async-gain fields in `/sbmpc/diagnostics`.

Required fields:

- `planner_mode`
- `last_foreground_planning_time_ms`
- `last_background_gain_time_ms`
- `last_gain_age_cycles`
- `last_gain_window_fill`
- `last_gain_completed_batch_count`
- `last_gain_dropped_snapshot_count`
- `last_gain_worker_running`
- `last_gain_worker_error`

### 4. Clean Shutdown

Shutdown must be deterministic.

Required behavior:

- Ctrl-C stops the bridge node
- bridge calls planner `close()`
- planner stops the background gain worker
- executor shuts down
- node is destroyed
- `rclpy.shutdown()` is called only if the context is still valid
- launch shutdown relies on managed ROS launch actions, not an external cleanup
  command
- stale Gazebo/ROS processes are detected before a new launch as a guardrail

## Gazebo Validation Sequence

All ROS/Gazebo commands are run inside the `sbmpc_containers` ROS environment.

Run this sequence on the pregrasp task first. Once pregrasp passes, repeat the
same structure for the full pick-and-place task.

### 1. Build and Unit Tests

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select sbmpc_ros_bridge sbmpc_bringup
colcon test --packages-select sbmpc_ros_bridge sbmpc_bringup --event-handlers console_direct+
colcon test-result --verbose
```

Pass condition:

- no test errors or failures

### 2. Planner Smoke

Run the planner through the ROS adapter without Gazebo:

```bash
/workspace/sbmpc_containers/scripts/pixi_ros_run.sh \
  python -m sbmpc_ros_bridge.planner_smoke --joint-set fer
```

Pass condition:

- `feedforward` shape is `(7, 1)`
- `feedback_gain` shape is `(7, 14)`
- joint names are `fer_joint1 ... fer_joint7`
- foreground planning time is reported
- async gain diagnostics are present in exact async mode

### 3. Gazebo Feedforward Baseline

Launch Gazebo with feedforward mode first.

Expected result:

- bridge warms up
- bridge remains silent until explicitly armed
- when armed, LFC receives feedforward with zero gain
- robot reaches the pregrasp target with task error <= 1 mm
- no stale process remains after shutdown

### 4. Gazebo Exact Async Feedback

Launch Gazebo with `exact_async_feedback`.

Expected result:

- foreground controller time stays within the 20 ms budget
- background gain worker starts
- rolling gain window fills
- first gain is published after the expected `ceil(M / K)` processed batches
- gain age remains bounded
- gain norm is stable after convergence
- task error remains <= 1 mm
- shutdown stops the worker and leaves no stale ROS/Gazebo graph

### 5. Full Pick-And-Place Extension

After pregrasp passes in Gazebo, extend the same controller path to the full
cube pick-and-place task.

Expected result:

- all phase transitions are driven by the `sbmpc` task logic, not ROS-side
  special cases
- bridge message shapes and gain sign convention remain unchanged
- foreground timing remains within budget
- async gain health remains stable across phase changes
- the cube is picked, transported, placed, and released without visible
  instability

### 6. Visual Assessment

The user validates visually on top of the numeric gates.

Visual checks:

- no shaking near convergence
- no late drift
- no controller transition jump when gains first become nonzero
- feedforward and feedback runs behave consistently with MuJoCo expectations
- for full pick-and-place, phase transitions and object motion look physically
  coherent

## Real Robot Readiness Gates

Do not move to the real robot until all of these are true:

- MuJoCo exact async benchmark passes timing, task error, and gain stability
- MuJoCo full pick-and-place passes with the same controller architecture
- Gazebo pregrasp feedforward passes
- Gazebo pregrasp exact async feedback passes
- Gazebo full pick-and-place passes
- Ctrl-C cleanly tears down the bridge, controller, and Gazebo launch graph
- LFC sign convention is validated
- torque limits and gain limits are configured for bringup
- bridge starts disarmed by default
- operator has an explicit arming step
- diagnostics are visible during the run

## Development Order

1. Stabilize and document the `sbmpc` async-gain public API. Done.
2. Update the Panda planner API to support feedforward, finite-difference, and
   exact async modes. Done for pregrasp and pick-and-place.
3. Add planner diagnostics for foreground timing and async gain health. Done.
4. Add ROS adapter parameters and lifecycle methods. Done.
5. Add bridge diagnostics for async gain state. Done.
6. Make shutdown self-contained. The external cleanup command has been removed;
   bridge planner shutdown is implemented and Gazebo launch teardown still needs
   container validation.
7. Add readable Gazebo YAML presets. Done for feedforward, finite-difference,
   and exact async pregrasp runs.
8. Run MuJoCo `bench_lfc.py` as the algorithm reference.
9. Hand off ROS container build/test commands.
10. Validate Gazebo pregrasp feedforward, then Gazebo pregrasp exact async
    feedback.
11. Extend to MuJoCo full pick-and-place with the same async-gain architecture.
12. Validate Gazebo full pick-and-place.
13. Prepare the real robot launch only after all gates pass.
