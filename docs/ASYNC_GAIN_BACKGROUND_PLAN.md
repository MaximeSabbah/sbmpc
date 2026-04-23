# Async Gain Background Plan

## Goal

Keep the foreground SB-MPC control loop inside the 20 ms budget while moving
exact gain computation to a background path that always prefers the most recent
local motion information.

This note is meant to survive context compaction. It records:

- the exact meaning of `N`, `K`, and `M`
- what data is needed for gain computation
- how the GPU work should be split between foreground and background
- the rolling-buffer / receding-window semantics
- the full implementation and validation plan
- how to time every relevant component so `K` and `M` can be tuned properly

## Core Definitions

Use these meanings consistently.

- `N`: full MPPI batch size used by the foreground controller every control
  cycle.
- `K`: number of samples from the current cycle preserved for gain work.
- `M`: number of newest processed gain samples retained in the rolling gain
  window used to synthesize the currently published gain.

Important:

- `N` is about **control quality now**
- `K` is about **how much new gain evidence enters the rolling window each
  cycle**
- `M` is about **how much recent evidence the published gain is built from**

If `N=1024`, it is still valid to choose:

- `K=1024`: preserve all MPPI samples for gain work
- `K=256`: preserve only 256 selected samples for gain work
- `K=128`: preserve only 128 selected samples for gain work

`K` is therefore **not** necessarily an internal GPU micro-batch size.
At the design level, it should mean:

> how many samples from each control cycle are promoted into the gain pipeline

If the worker later needs to process those `K` samples internally in smaller
micro-batches, that is an implementation detail and should not change the
external meaning of `K`.

## What Is Needed To Compute Exact Gains

For each retained sample `i`, the current gain formula ultimately needs:

- `J_i`: the scalar trajectory cost
- `delta_u0_i`: the first control perturbation associated with sample `i`
- `grad_i = dJ_i / dx`: the state gradient of that sample cost

This is the crucial point:

- `J_i` and `delta_u0_i` are already available from the foreground MPPI rollout
- the expensive missing quantity is mainly `grad_i`

So for exact async gains, the foreground should **not** compute gradients.
If it did, the expensive part would still be on the control path and async
would buy very little.

## What The Foreground Already Produces

For one control cycle `t`, after the foreground MPPI rollout we already have:

- `state_t`
- `reference_t`
- `optimal_samples_t`
- `raw_samples_delta_t` for all `N`
- `costs_t` for all `N`
- `delta_u0_t` for all `N`

The async design should preserve those and let the background worker add:

- `grad_t` for the selected `K` samples

This means the background worker should ideally receive an immutable snapshot
containing enough information so it does not have to recompute what the
foreground already knows.

## What The Gain Formula Actually Uses

The gain formula does **not** use the full control trajectory perturbation at
publication time. It uses only the first-step perturbation `delta_u0_i`.

The whole horizon still matters because:

- `J_i` is a full horizon trajectory cost
- `grad_i = dJ_i / dx` comes from a full horizon rollout

But the published gain maps current state error to the current control
correction, so the control-side retained sample quantity is the first-step
term.

## What Should Stay On The Foreground GPU Path

Allowed on the foreground path:

- MPPI rollout with `N` samples
- action synthesis / update of the nominal control
- extraction of `costs_t`
- extraction of `delta_u0_t`
- selection of the `K` samples to preserve for gain work
- packaging of an immutable snapshot for the background worker

Not allowed on the foreground path:

- exact gain gradient computation
- blocking on background worker completion
- any mutation of worker-owned rolling gain state

## What Should Run On The Background Path

The background path should do only the gain-specific expensive work:

1. receive a snapshot
2. compute `grad_i = dJ_i / dx` for the selected `K` samples
3. combine those gradients with the already-snapshotted `J_i` and `delta_u0_i`
4. insert the resulting `K` processed contributions into the rolling gain
   window
5. when the rolling window is full, synthesize and publish a new gain

## GPU Management: What Is Actually Efficient

### Can the controller send gradients of a subset `K` to the worker?

No. That defeats the purpose.

If the controller computes exact gradients for the subset on the foreground
path, then the expensive part is still on the control loop.

The efficient split is:

- foreground sends the **sample data**
- background computes the **gradients**

### Can we send all `N=1024` samples to the worker?

Yes.

But there is a throughput caveat:

- if the worker only manages to process `K=128` new samples per control period
- and the foreground enqueues all `1024` every control period
- then backlog grows by `896` samples every cycle

That is not sustainable unless:

- the worker processes more than one `K` block per control period
- old unprocessed work is dropped aggressively
- or the foreground promotes only a subset `K` from the `N` samples

That is why `K` is important.

### Recommended interpretation for same-GPU async

On one GPU, the practical first design is:

- foreground runs MPPI with `N`
- foreground selects `K <= N` samples for gain work
- only those `K` are promoted into the background gain pipeline

This keeps the design tunable and avoids guaranteed backlog growth.

If later measurements show that `K=N` is feasible, great. That should be one of
the first probe points.

## How To Select The `K` Samples From `N`

This needs to be explicit because it changes both relevance and cost.

Initial options:

1. `K = N`
   - preserve everything
   - simplest and most faithful baseline

2. `Top-K by MPPI relevance`
   - preserve the `K` lowest-cost / highest-weight trajectories
   - most aligned with the trajectories that dominate the MPPI action

3. `Always keep nominal + top-(K-1)`
   - sample 0 is the nominal trajectory
   - preserve it explicitly
   - fill the rest with the strongest non-nominal trajectories

4. `Weighted resampling`
   - sample `K` trajectories according to MPPI weights
   - more diverse, less deterministic than top-K

Recommended first practical choice:

- keep nominal sample 0
- add the top `K-1` samples by current MPPI relevance

Reason:

- simple
- deterministic
- keeps the local nominal anchor
- prioritizes the trajectories that actually shape the current control action

## Rolling Window / Receding Gain Semantics

This part is correct and should be part of the final design.

Assume:

- `M = 512`
- `K = 128`

Then:

- after the first 4 updates, the rolling gain window is full
- the first nonzero gain is synthesized from those 512 processed samples
- at the next gain update:
  - discard the oldest 128 processed samples
  - add the newest 128 processed samples
  - recompute the gain from the newest 512 processed samples

So yes:

> discard old `K`, add new `K`, recompute immediately

is the correct rolling / receding-window behavior once the window is full.

This is exactly analogous to a receding horizon, but applied to the rolling
sample evidence used for gains.

## Queue Semantics

Two containers are needed and they must not be confused.

### 1. Unprocessed work queue

Policy should be **newest-first / LIFO in effect**.

Rationale:

- if the worker falls behind, old unprocessed work rapidly becomes irrelevant
- the gain should stay local to the current motion
- if something must be dropped, drop older waiting work first

Recommended implementation semantics:

- bounded queue size 1 or 2
- newest snapshot processed first
- oldest waiting snapshot dropped when queue is full

### 2. Processed rolling gain window

This stores the newest processed gain contributions.

Semantics:

- retain the newest `M` processed tuples
  `(J_i, delta_u0_i, grad_i)`
- when a new batch of `K` arrives and the window is already full:
  - evict the oldest `K`
  - append the newest `K`
- recompute gain immediately from the current `M`

Order inside this window does not matter mathematically as long as the retained
set is the newest `M` processed contributions.

## Timing Model

Use these symbols when reasoning about feasibility:

- `P = 20 ms`: control period
- `t_action(N)`: hot steady-state foreground control time without gain work
- `t_select(K)`: subset selection and snapshot packaging time
- `t_grad_exact(K)`: worker time to compute exact gradients for the promoted `K`
  samples
- `t_gain_synth(M)`: time to synthesize a gain from the current rolling window
- `t_publish`: atomic publication time, expected tiny

### Foreground target

The foreground loop must satisfy:

`t_action(N) + t_select(K) < P`

### Background throughput target

The background worker must keep up **on average**.

If it processes one promoted batch per control cycle, a conservative same-GPU
throughput criterion is:

`t_grad_exact(K) + t_gain_synth(M on publish cycles) << P`

But because GPU contention is real, the true criterion is empirical:

- foreground `planning_ms` must remain under budget while the background is
  active
- queue depth must stay bounded
- published gain age must not grow without bound

### First-gain latency

Idealized first-gain latency if exactly one `K` batch is processed per cycle:

`latency_first ~= ceil(M / K) * P`

Examples:

- `K=128`, `M=512` -> first gain after about 4 cycles -> about 80 ms
- `K=128`, `M=1024` -> first gain after about 8 cycles -> about 160 ms
- `K=1024`, `M=1024` -> first gain after about 1 processed batch

In reality this must be measured, not assumed.

## Dummy Numerical Example

Assume:

- `N = 1024`
- `horizon = 8`
- `control_points = 8`
- hypothetical no-gain foreground MPPI time `t_action(1024) = 10 ms`
- exact gradient time for promoted set `t_grad_exact(128) = 13 ms`

Interpretation:

- foreground still has room because `10 ms < 20 ms`
- but same-GPU async does **not** mean the worker gets a clean spare 10 ms slot
- the question is whether the worker can process `K=128` often enough without
  hurting the foreground loop

If `K=128`, `M=512`:

- cycle 0:
  - foreground runs MPPI on 1024 samples
  - foreground keeps `K=128` selected samples for gain work
  - background computes gradients for those 128
  - rolling window fill = 128
- cycle 1:
  - another 128 processed
  - rolling window fill = 256
- cycle 2:
  - rolling window fill = 384
- cycle 3:
  - rolling window fill = 512
  - first gain is synthesized and published
- cycle 4:
  - oldest 128 removed
  - newest 128 added
  - gain is recomputed from the newest 512 processed samples

This is the intended receding-window gain behavior.

## Recommended External Parameters

The tunable exposed parameters should be:

- `N`: already exists, foreground MPPI batch size
- `K`: promoted gain subset per cycle
- `M`: rolling processed gain window size

Useful constraints:

- `1 <= K <= N`
- `M >= K`
- `M % K == 0` for simple rolling-window updates

Recommended initial probe grid:

1. `N=1024`, `K=1024`, `M=1024`
2. `N=1024`, `K=512`, `M=1024`
3. `N=1024`, `K=256`, `M=1024`
4. `N=1024`, `K=128`, `M=512`
5. `N=1024`, `K=128`, `M=1024`
6. `N=512`, `K=512`, `M=512`
7. `N=512`, `K=256`, `M=512`

This grid tests:

- full preservation vs subset promotion
- short vs long first-gain latency
- small vs large rolling evidence windows

## Architecture Proposal

### Foreground API

Introduce a helper with semantics like:

`plan_action_and_snapshot(state, reference) -> (tau_ff, snapshot)`

It should:

- run the normal MPPI action update
- return the feedforward command
- return an immutable `GainSnapshot` containing:
  - `cycle_id`
  - `state`
  - `reference`
  - selected sample indices
  - `costs_K`
  - `delta_u0_K`
  - the selected raw control variables needed for exact gradient evaluation

It should **not** compute gains.

### Background API

`process_snapshot(snapshot) -> ProcessedGainBatch`

It should:

- compute exact `grad_i = dJ_i / dx` for the selected `K` samples
- return:
  - `costs_K`
  - `delta_u0_K`
  - `grads_K`
  - `source_cycle_id`

### Rolling gain window

`RollingGainWindow(M, K)`

It should:

- hold the newest `M` processed tuples
- expose:
  - `append(batch)`
  - `is_full()`
  - `compute_gain()`
- implement:
  - evict oldest `K`
  - add newest `K`
  - recompute on every append once full

### Published gain state

`PublishedGain`

Fields:

- `gain_matrix`
- `source_cycle_id`
- `publish_cycle_id`
- `window_fill`

Foreground reads only the latest `PublishedGain`.

## Synchronization Rules

These are mandatory.

1. `GainSnapshot` is immutable.
2. Worker never touches live `sampler.optimal_samples`.
3. Foreground never mutates the rolling gain window.
4. Gain publication is atomic.
5. Unprocessed work queue is bounded.
6. Older waiting work may be dropped.
7. Very stale completed gains must be discarded.
8. Before first gain publication, controller runs effectively open loop
   (`K_lfc = 0`).

## Recommended Execution Strategy

### Phase 0: No-thread probe first

Before implementing real threads, add a fake-async mode in `bench_lfc.py`.

This mode should:

- execute foreground planning every cycle exactly as intended
- simulate the worker separately and time it separately
- let us study:
  - first gain latency
  - queue growth
  - rolling-window behavior
  - foreground timing under a realistic update schedule

This avoids committing to thread/process design too early.

### Phase 1: Same-process thread if probe is promising

Recommended real implementation first:

- same process
- background thread
- bounded newest-first queue

Reason:

- avoids GPU context duplication
- avoids expensive host serialization of device arrays
- simplest path to a real proof of concept

Do **not** start with a separate process unless measurements prove the thread
model is inadequate.

## What Must Be Timed Properly

Timing must separate compile cost from hot steady-state cost.

### Warmup / compile timing

For every newly introduced helper, record:

- first-call latency
- hot-call latency

Every timing in the benchmark must use `jax.block_until_ready(...)` at the
measurement boundary, otherwise numbers are meaningless.

### Required micro-timers

Add explicit timing helpers for:

1. `t_action(N)`
   - foreground planning with gains disabled / frozen

2. `t_subset_select(K)`
   - selecting the `K` samples from the full `N`

3. `t_snapshot_pack(K)`
   - packing the immutable snapshot for the worker

4. `t_grad_exact(K)`
   - exact gradient computation on the promoted `K`

5. `t_gain_synth(M)`
   - gain synthesis from a full rolling window

6. `t_publish`
   - publication / swap of the new gain

7. `t_total_foreground`
   - measured in `bench_lfc` as the quantity that must stay under budget

8. `t_total_background`
   - one complete worker batch from snapshot to published gain when applicable

### Required probe metrics

The fake-async and later real-async benchmark should report:

- `planning_ms`:
  foreground control time only
- `subset_select_ms`
- `snapshot_pack_ms`
- `gain_grad_ms`
- `gain_synth_ms`
- `gain_refresh_ms`:
  full background update time
- `first_gain_ready_cycle`
- `published_gain_age_cycles`
- `queue_depth_max`
- `dropped_snapshot_count`
- `rolling_window_fill`
- standard closed-loop metrics:
  - final error
  - tail error std
  - gain norm
  - feedback peak
  - joint velocity stats

## Validation Plan

### Unit tests

Add / extend unit tests for:

1. `GainSnapshot` content correctness
   - correct shapes
   - selected `K` samples match intended policy

2. Rolling gain window semantics
   - before full, no gain publication
   - after full, oldest `K` replaced by newest `K`
   - retained set always equals newest `M`

3. Publication logic
   - published gain remains zero before first full window
   - gain updates after every new processed batch once full

4. Staleness policy
   - old waiting work dropped correctly
   - stale completed gains rejected when necessary

5. Sync compatibility
   - if async mode is disabled, behavior matches current synchronous controller

### Low-dimensional integration tests

In `tests/test_mppi_gains.py` or a new async-focused test file:

1. Constant linear system case
   - async exact with `K=N`, `M=N` should recover the same gain as sync exact
     after one full background update

2. Rolling-window update case
   - with `K < M`, verify that first gain appears after `M / K` batches
   - verify oldest `K` are replaced on the next update

3. Open-loop bootstrap case
   - confirm `K_lfc = 0` before first gain publication

### Controller micro-benchmark

Create a small benchmark mode to measure:

- `t_action(N)`
- `t_grad_exact(K)`
- `t_gain_synth(M)`

for candidate values of:

- `N in {256, 512, 1024}`
- `K in {64, 128, 256, 512, 1024}`
- `M in {K, 2K, 4K, 8K}`

Save hot timings and first-call timings separately.

### Real acceptance benchmark

Use `bench_lfc.py --timing-mode gazebo` as the acceptance harness.

Success criteria:

- foreground `planning_ms` below 20 ms with margin
- acceptable final tracking error
- acceptable tail stability
- no unacceptable feedback / gain / velocity spikes
- queue depth bounded
- gain age bounded

## Tuning Workflow

This is the intended practical tuning loop.

1. Measure `t_action(N)` with gains disabled or frozen.
2. Measure `t_grad_exact(K)` for candidate `K`.
3. Measure `t_gain_synth(M)` for candidate `M`.
4. Choose feasible `(N, K, M)` candidates.
5. Run fake-async `bench_lfc --timing-mode gazebo`.
6. Discard candidates where:
   - foreground timing exceeds budget
   - queue depth diverges
   - first gain arrives too late
   - closed-loop behavior is poor
7. Only then implement the real worker.

## Small JAX Cleanups Still Worth Considering

These are second-order improvements, not the main solution:

- pre-tile references outside hot loops
- precompute static subset index helpers
- avoid unnecessary host-device transfers in the foreground path
- keep timing boundaries explicit with `block_until_ready`

They may save a little time, but the main win is expected from the async
architecture and careful `N/K/M` tuning.
