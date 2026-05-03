# Job Contention Modeling — Design Spec

**Date:** 2026-05-03
**Status:** Approved (pending implementation)

## 1. Goal

Model the impact of inter-job network/resource contention on job completion
time in the RFold simulator. When the placement set changes — a job is
admitted into the cluster, or a running job completes — the contention
landscape shifts: some running jobs slow down, others speed up. The estimated
completion time (ETA) of every running job must be re-derived dynamically to
reflect this.

A pluggable function takes the current placement and returns a per-job
slowdown factor. The simulator integrates this into the existing scheduler
+ running-queue + guard machinery without disturbing baseline behavior.

## 2. Slowdown semantics

The slowdown function returns an **instantaneous slowdown multiplier** per
running job. While job J has factor `s > 0`, every wall-clock second
advances `1/s` seconds of "ideal work done" toward J's static
`duration_sec`. The typical contended case has `s ≥ 1` (slower than
ideal); `s = 1` means no contention; `s < 1` (speedup) is mathematically
admitted but unusual. The factor is constant between events and is
recomputed at every event (admit / complete).

### Worked example (the J1/J2 case)

J1 has `duration_sec = 5 min`. J2 arrives 1 min later, doubles J1's
slowdown while co-resident, then completes after 2 wall minutes:

| t (min) | event | J1.work_done | J1.current_slowdown | J1.priority (ETA) |
|---|---|---|---|---|
| 0 | J1 admitted, applySlowdown(1.0, 0)        | 0   | 1.0 | 5 |
| 1 | J2 admitted, applySlowdown(2.0, 1) on J1  | 1.0 | 2.0 | 9 |
| 3 | J2 completes, applySlowdown(1.0, 3) on J1 | 2.0 | 1.0 | 6 |
| 6 | guard fires for J1                        | 5.0 | —   | — |

J1 completes at wall t=6 having done 5 min of ideal work.

## 3. Triggers

Slowdowns are recomputed in exactly these cases:

- **Admit** — `executeOnCluster()` adds the new job to the running set.
- **Complete** — `completeOnCluster()` removes a job from the running set.

`PREEMPT` and `RECONFIGURE` decisions in the scheduler currently route
through `executeOnCluster` after a delay; they pick up recomputation for
free. No additional contention semantics are introduced for the migration
or reconfiguration delays themselves; richer behavior (e.g., partial
progress lost on preemption) is deferred.

Rejection does not change the placement set and does not trigger
recomputation. Failed nodes are out of scope for this spec; the natural
extension is to invoke `recompute()` from a future failure handler.

## 4. Architecture

### File layout

| File | Action |
|---|---|
| `ClusterManager/contention.py` | **New.** Houses `ContentionModel`. Mirrors `ClusterManager/scheduling.py`. |
| `common/job.py` | **Modified.** Three new fields on `Job` and one new method `applySlowdown`. |
| `ClusterManager/manager.py` | **Modified.** Instantiate the model, call it on admit/complete, rebuild the running queue, interrupt the guard. |
| `Cluster/cluster.py` | **Modified.** Remove the stale TODO at lines 155-157 (now resolved). |
| `test/test_contention.py` | **New.** Unit + integration tests. |

### Component responsibilities

| Component | Role |
|---|---|
| `ContentionModel` | Owns the pluggable `slowdown(running_jobs) -> {uuid: factor}` function (identity stub: returns `1.0` for all). Provides `recompute(running_jobs)` which calls the function and pushes each new factor onto the corresponding job. Logs at debug whenever a factor changes. |
| `Job` | Tracks its own progress: `work_done_ideal_sec`, `last_event_time_sec`, `current_slowdown`. Method `applySlowdown(s, t)` accumulates ideal work since the last event, stores the new factor, and updates `self.priority` to the new ETA. |
| `ClusterManager` | Calls `contention_model.recompute(running_jobs)` after every admit and every complete. Rebuilds the `SortedList` so the head reflects the new earliest ETA, and interrupts/triggers the running queue guard accordingly. |
| `runningQueueGuard` | Unchanged in structure. After a recompute it either gets interrupted (if it was sleeping on a timer) or naturally re-reads the queue head on its next iteration. |

## 5. `Job` changes

### New fields

```python
# ----- contention model state -----
# Ideal work completed so far (in seconds of ideal duration). Advances at rate
# 1/current_slowdown per wall-clock second between events. Capped at duration_sec.
work_done_ideal_sec: float = field(default=0.0, compare=False)
# Wall-clock time at which work_done_ideal_sec was last updated. None until the
# job is admitted and its first slowdown is applied. Used to compute the elapsed
# wall time delta when a new slowdown factor arrives.
last_event_time_sec: Optional[float] = field(default=None, compare=False)
# The slowdown factor in effect for this job since last_event_time_sec.
# 1.0 = no contention, > 1.0 = the job runs slower (e.g., 2.0 means 1 wall sec
# advances 0.5 sec of ideal work). Recomputed by ContentionModel on each
# admit/complete event.
current_slowdown: float = field(default=1.0, compare=False)
# ----- end of contention model state -----
```

All three fields use `compare=False` so they do not perturb the
`priority`-based ordering used by `SortedList`.

`duration_sec` keeps its existing meaning (immutable ideal duration). The
existing `Job.slowdown = jct_sec / duration_sec` stays valid as the
realized end-state slowdown.

### New method

```python
def applySlowdown(self, new_slowdown: float, current_time: float):
    """
    Tell this job its slowdown for the next period.
    1) Accrue ideal work done since the last event at the prior factor.
    2) Store the new factor and the current time.
    3) Update self.priority to the projected ETA = now + remaining_ideal * new_s.
    """
    if self.last_event_time_sec is not None:
        elapsed_wall = current_time - self.last_event_time_sec
        self.work_done_ideal_sec += elapsed_wall / self.current_slowdown
    # Clamp against floating-point overshoot.
    self.work_done_ideal_sec = min(self.work_done_ideal_sec, self.duration_sec)
    self.last_event_time_sec = current_time
    self.current_slowdown = new_slowdown
    remaining_ideal = self.duration_sec - self.work_done_ideal_sec
    self.priority = current_time + remaining_ideal * new_slowdown
```

## 6. `ContentionModel`

```python
import logging
import simpy
from typing import Iterable

from common.job import Job
from Cluster.cluster import Cluster


class ContentionModel:
    """
    Models network/resource contention between concurrently running jobs and
    converts it into per-job slowdown factors. The factors are pushed onto each
    Job, which uses them to track its own progress.

    The slowdown function is the pluggable boundary. The default identity
    implementation returns 1.0 for every job, making contention modeling a
    no-op. Replace with a real model when ready.
    """

    def __init__(self, env: simpy.core.Environment, cluster: Cluster):
        self.env = env
        self.cluster = cluster

    def slowdown(self, running_jobs: Iterable[Job]) -> dict[int, float]:
        """
        Pluggable slowdown function. Inputs:
          - running_jobs: every currently placed job. Each Job's .allocation
            field carries the exact job-to-node mapping; cluster topology is
            available via self.cluster.
        Returns: {job.uuid: slowdown_factor}. Factor >= 1.0 means slower; 1.0
        means no contention. Identity stub by default.
        """
        return {job.uuid: 1.0 for job in running_jobs}

    def recompute(self, running_jobs: Iterable[Job]) -> None:
        """
        Recompute slowdowns for all currently running jobs and push the new
        factor onto each. Jobs update their own work-done accumulator and ETA.
        Logs a debug line whenever a factor changes.
        """
        jobs = list(running_jobs)
        factors = self.slowdown(jobs)
        for job in jobs:
            old_s = job.current_slowdown
            new_s = factors[job.uuid]
            job.applySlowdown(new_s, self.env.now)
            if old_s != new_s:
                logging.debug(
                    f"t = {self.env.now}, Job {job.uuid} slowdown "
                    f"{old_s} -> {new_s}, new ETA {job.priority}"
                )
```

### Boundary contract

- `recompute()` is called by `ClusterManager` *after* the placement set has
  changed (i.e., after `cluster.execute()` has allocated the new job, or
  after `cluster.complete()` has freed the completing one). The
  `running_jobs` argument is the up-to-date set.
- `slowdown()` is pure with respect to its inputs — no side effects on
  jobs or cluster. It only inspects the placements/topology and returns
  numbers.
- `recompute()` does have a side effect: it mutates each job's
  `work_done_ideal_sec`, `last_event_time_sec`, `current_slowdown`, and
  `priority`. That mutation is the entire mechanism by which ETAs get
  refreshed.

### No flag gating

With the identity stub returning `1.0`, every `applySlowdown(1.0, t)`
produces the same `priority` as the existing static `arrival + duration`
calculation (modulo floating-point). Behavior matches the no-contention
baseline until the stub is swapped for a real implementation.

## 7. `ClusterManager` wiring

### `__init__` adds

```python
self.contention_model = ContentionModel(env, cluster)
```

alongside the existing `self.scheduler`.

### `executeOnCluster(job)` — modified

```python
def executeOnCluster(self, job: Job):
    job.updateQueueingTime(self.env.now)
    # Initialize contention-tracking state for this brand-new running job.
    # The recompute below will set last_event_time_sec and the real slowdown,
    # then overwrite priority via applySlowdown.
    job.work_done_ideal_sec = 0.0
    job.last_event_time_sec = None
    job.current_slowdown = 1.0
    job.priority = self.env.now + job.duration_sec  # placeholder, overwritten
    self.cluster.execute(job)
    self.new_job_queue.remove(job)
    self.running_job_queue.enqueue(job)
    # Placement set changed: recompute for every running job, including the
    # new one, then re-sort so the head reflects the new earliest ETA.
    self.contention_model.recompute(self.running_job_queue.slist)
    self._rebuildRunningQueue()
    # Wake the guard. If it was on a timer, interrupt it; if it was waiting
    # for a job to be enqueued, signal arrival.
    if self.next_completion > self.env.now:
        self.running_guard_proc.interrupt()
    else:
        self.event_running.trigger()
    self.logClusterStats()
```

The earlier conditional `... and job.priority < self.next_completion` is
dropped — *any* running job's ETA can change on a recompute, so the guard
must always re-read the head.

### `completeOnCluster(job)` — modified

```python
def completeOnCluster(self, job: Job):
    self.cluster.complete(job)
    job.completion_time_sec = self.env.now
    job.jct_sec = self.env.now - job.arrival_time_sec
    job.slowdown = job.jct_sec / job.duration_sec
    self.job_stats[job.uuid] = job
    # Placement set changed: refresh the remaining running jobs.
    self.contention_model.recompute(self.running_job_queue.slist)
    self._rebuildRunningQueue()
    # No guard interrupt needed: the guard called us from inside its loop;
    # it will naturally peek the (re-sorted) head on its next iteration.
    self.event_completion.trigger()
    self.logClusterStats()
```

### New helper `_rebuildRunningQueue`

```python
def _rebuildRunningQueue(self):
    """
    Re-sort the running queue after recompute mutated job.priority.
    SortedList sorts on insert via Job's order=True comparator, so we have
    to dequeue everything and re-insert.
    """
    items = list(self.running_job_queue.slist)
    self.running_job_queue.slist.clear()
    for j in items:
        self.running_job_queue.enqueue(j)
```

### `Cluster.complete` TODO removed

The TODO at `cluster.py:155-157` flagged that the actual completion time
may diverge from the theoretical one once runtime dynamics are modeled.
This is now resolved at the manager layer via dynamic ETAs, so the comment
is removed.

## 8. Edge cases & invariants

**Numerical / boundary**

- **Floating-point overshoot.** `applySlowdown` clamps
  `work_done_ideal_sec ≤ duration_sec`, so `remaining_ideal ≥ 0` always.
  Combined with `new_slowdown > 0`, this guarantees
  `priority ≥ current_time` — the guard never gets a past-due timeout.
- **Slowdown < 1 (speedup).** Math is unchanged; `priority` simply lands
  earlier than the unstretched ideal. Allowed.
- **Slowdown = 0 or negative.** Treated as a contract violation by the
  slowdown function. Not defensively guarded; an identity stub returning
  `1.0` makes this unreachable, and a real model should be tested
  independently.
- **`remaining_ideal` rounds to exactly 0.** `priority = env.now`. Guard
  fires on the next simpy step, completing the job promptly.

**Ordering / queue invariants**

- After `recompute()` + `_rebuildRunningQueue()`, the SortedList is
  consistent with the latest `priority` values on every running job. This
  is the only place priorities are mutated for running jobs.
- `next_completion` is read by the guard, not by
  `executeOnCluster`/`completeOnCluster` for ordering decisions any more.
  Its only role is "is the guard sleeping on a timer?"

**Initialization of a newly admitted job**

- `last_event_time_sec = None` and `current_slowdown = 1.0` going into
  recompute.
- The first `applySlowdown(s, t)` call sees `last_event_time_sec is None`
  → skips the elapsed-work accumulation (correct; no work has been done
  yet) → sets `last_event_time_sec = t`, `current_slowdown = s`,
  `priority = t + duration_sec * s`.

**Interaction with existing flows**

- `closed_loop_threshold` / `totalNewWork` uses `duration_sec` to estimate
  offered load. `duration_sec` remains the immutable ideal duration, so
  this metric keeps its meaning.
- Trace generation (`job_stats_to_trace`) uses arrival time, sched time,
  completion time. All still recorded correctly; emitted JCT now reflects
  realized contention naturally.
- `flushAllQueues` at simulation end is unchanged. Running jobs that did
  not complete still get dumped without realized JCT. This was already the
  case.
- `Job.slowdown = jct_sec / duration_sec` computed in `completeOnCluster`
  is the realized end-state slowdown and remains correct under contention.
- Failed nodes (`Cluster.failNodes`) are out of scope here. The natural
  extension is to invoke `contention_model.recompute()` from the failure
  handler — same pattern.

**`PREEMPT` / `RECONFIGURE`**

- Both currently invoke `executeOnCluster` after a delay. They pick up
  recomputation for free because `executeOnCluster` always recomputes. No
  new semantics added; richer behavior (e.g., partial progress lost on
  preemption) is a later step.

**Concurrency / simpy**

- `recompute()` is synchronous, no `yield`, no scheduling boundary
  crossed. It always runs to completion within the same simpy event step
  that fired it, so the running queue is consistent before any other
  process observes it.

## 9. Observability

- **`Job.slowdown`** continues to be the realized end-state slowdown
  (`jct_sec / duration_sec`).
- **Debug log per slowdown change.** `ContentionModel.recompute()` emits
  one debug-level log line whenever any job's factor differs from its
  prior value. Useful while validating the model; cheap; no schema change.
- **No new persistent stats** (no per-job slowdown trajectory). Can be
  added later if needed for plotting.

## 10. Testing

New file: `test/test_contention.py`. Existing tests use `unittest`
(`uv run python -m unittest`).

### Unit — `Job.applySlowdown` (no simpy)

1. **Identity sequence is a no-op.** `duration_sec=10`, sequence
   `applySlowdown(1.0, 0)` then `applySlowdown(1.0, 5)`. Expected
   `priority == 10`, `work_done_ideal_sec == 5`.
2. **Reproduce the J1/J2 worked example.** `duration_sec=5`, sequence
   `(1.0, 0)`, `(2.0, 1)`, `(1.0, 3)`. Expected priorities `5, 9, 6`;
   work-done `0, 1, 2`.
3. **Floating-point clamp.** Drive `work_done_ideal_sec` past
   `duration_sec` via a deliberately oversized elapsed wall step; assert
   it clamps and `priority == current_time`.
4. **First call with `last_event_time_sec is None` accumulates nothing.**
   First `applySlowdown(2.0, 7)` after admission at `t=7` produces
   `work_done == 0` and `priority == 7 + duration_sec * 2`.

### Unit — `ContentionModel`

5. **Identity stub returns 1.0 for every running job.** Construct a model,
   build a fake `running_jobs` list, call `recompute`. Assert each job's
   `current_slowdown == 1.0`.
6. **Slowdown-change debug log fires.** Subclass with `slowdown` returning
   `2.0`. Call `recompute` twice in a row at the same `env.now`. Assert one
   debug log on the first call (1.0 → 2.0), zero on the second (2.0 → 2.0).
7. **`applySlowdown` is invoked once per running job per recompute.** Spy
   via subclass.

### Integration — full simpy + cluster + manager

8. **Identity contention reproduces the no-contention baseline.** Run a
   small replay end-to-end with the default identity stub. Assert every
   job's `jct_sec == duration_sec` (within FP tolerance) and
   `Job.slowdown == 1.0`. Regression guard: shipping the feature must not
   change existing simulation outputs.
9. **Fixed 2x slowdown doubles JCT.** Inject a `ContentionModel` subclass
   whose `slowdown` returns `2.0`. Run with two non-overlapping jobs.
   Assert each job's JCT is exactly `2 * duration_sec`.
10. **End-to-end J1/J2 example.** Two jobs with the timing from the
    worked example. Custom slowdown function returns `2.0` while both run,
    `1.0` otherwise. Assert J1 completes at `t=6` and J2 at `t=3`.

### Out of scope for this spec

- A real (non-stub) slowdown function.
- `PREEMPT`/`RECONFIGURE` interactions beyond the implicit
  `executeOnCluster` recompute.
- Failed-node mid-run recompute hooks.
