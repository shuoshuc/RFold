# Job Contention Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dynamic per-job slowdown tracking so the simulator's estimated completion times reflect placement-induced contention as jobs are admitted and complete.

**Architecture:** A pluggable `ContentionModel` class (mirroring `SchedulingPolicy`) holds a `slowdown(running_jobs) -> {uuid: factor}` function (identity stub for now). On every admit and every complete, `ClusterManager` calls `ContentionModel.recompute()`, which pushes new factors onto each running `Job` via a new `Job.applySlowdown(s, t)` method. `Job` tracks its own ideal-work accumulator and self-updates `priority` (the running queue's sort key). The manager rebuilds the SortedList and interrupts the running queue guard.

**Tech Stack:** Python 3.9+, simpy 4.1+, `unittest`, `uv` (run tests via `uv run python -m unittest`).

**Reference spec:** `docs/superpowers/specs/2026-05-03-job-contention-modeling-design.md`

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `common/job.py` | Modify | Add three contention-tracking fields and `applySlowdown` method to `Job`. |
| `ClusterManager/contention.py` | Create | New `ContentionModel` class with identity-stub `slowdown` and `recompute` driver. |
| `ClusterManager/manager.py` | Modify | Instantiate the model; recompute on admit/complete; rebuild running queue; clean up guard interrupt logic. |
| `Cluster/cluster.py` | Modify | Remove the stale TODO at lines 155-157 that the new design resolves. |
| `test/test_contention.py` | Create | Unit tests for `Job.applySlowdown`, `ContentionModel`, plus integration tests through `ClusterManager`. |

---

## Task 1: Add contention fields and `applySlowdown` to `Job`

**Files:**
- Modify: `common/job.py` (add fields + method on the `Job` dataclass)
- Test: `test/test_contention.py` (new file; unit tests for `applySlowdown`)

This task is pure data-class math — no simpy, no manager. Keep it that way.

- [ ] **Step 1: Create the test file with the four `applySlowdown` unit tests.**

Create `test/test_contention.py` with the following content:

```python
import unittest

from common.job import Job, TopoType


def make_job(uuid: int, duration_sec: float) -> Job:
    """Construct a Job with the minimum required fields for contention tests."""
    return Job(
        uuid=uuid,
        topology=TopoType.CLOS,
        shape=(1,),
        size=1,
        duration_sec=duration_sec,
        arrival_time_sec=0,
    )


class TestApplySlowdown(unittest.TestCase):
    def test_identity_sequence_no_op(self):
        """Two identity (s=1.0) calls 5 sec apart accumulate exactly 5 sec of work."""
        job = make_job(uuid=1, duration_sec=10)
        job.applySlowdown(1.0, 0)
        self.assertEqual(job.work_done_ideal_sec, 0.0)
        self.assertEqual(job.priority, 10)

        job.applySlowdown(1.0, 5)
        self.assertEqual(job.work_done_ideal_sec, 5.0)
        self.assertEqual(job.priority, 10)
        self.assertEqual(job.current_slowdown, 1.0)
        self.assertEqual(job.last_event_time_sec, 5)

    def test_J1_worked_example(self):
        """The J1 trajectory from the spec: duration 5, contended (s=2) from t=1 to t=3, then alone."""
        job = make_job(uuid=1, duration_sec=5)

        # t=0: J1 admitted alone
        job.applySlowdown(1.0, 0)
        self.assertEqual(job.work_done_ideal_sec, 0.0)
        self.assertEqual(job.priority, 5)

        # t=1: J2 arrives, J1 slows to s=2
        job.applySlowdown(2.0, 1)
        self.assertEqual(job.work_done_ideal_sec, 1.0)
        self.assertEqual(job.priority, 9)  # 1 + (5-1)*2 = 9

        # t=3: J2 completes, J1 back to s=1
        job.applySlowdown(1.0, 3)
        self.assertEqual(job.work_done_ideal_sec, 2.0)  # 1 + (3-1)/2 = 2
        self.assertEqual(job.priority, 6)  # 3 + (5-2)*1 = 6

    def test_floating_point_clamp(self):
        """work_done_ideal_sec is clamped at duration_sec; priority cannot land in the past."""
        job = make_job(uuid=1, duration_sec=5)
        job.applySlowdown(1.0, 0)
        # Drive 100 wall seconds at s=1, which would put work_done at 100 without clamp.
        job.applySlowdown(2.0, 100)
        self.assertEqual(job.work_done_ideal_sec, 5.0)  # clamped
        self.assertEqual(job.priority, 100)             # remaining = 0, ETA = now

    def test_first_call_no_accumulation(self):
        """When last_event_time_sec is None, the first applySlowdown does not accumulate work."""
        job = make_job(uuid=1, duration_sec=5)
        # First call comes at t=7 with s=2.
        job.applySlowdown(2.0, 7)
        self.assertEqual(job.work_done_ideal_sec, 0.0)
        self.assertEqual(job.priority, 7 + 5 * 2)
        self.assertEqual(job.current_slowdown, 2.0)
        self.assertEqual(job.last_event_time_sec, 7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and confirm they fail.**

Run: `uv run python -m unittest test.test_contention -v`

Expected: All four tests in `TestApplySlowdown` fail with `AttributeError: 'Job' object has no attribute 'applySlowdown'` (or similar) — the method does not exist yet.

- [ ] **Step 3: Add the contention-tracking fields to `Job`.**

In `common/job.py`, locate the `Job` dataclass. After the existing `# ----- end of stats -----` block (around line 66) and before the `def __post_init__` method, add a new section. The full insertion (preserving comments) is:

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

All three fields use `compare=False` so they do not perturb the `priority`-based ordering used by `SortedList`.

- [ ] **Step 4: Add the `applySlowdown` method to `Job`.**

In `common/job.py`, append the method to the `Job` class (after the existing `updateQueueingTime` method):

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

- [ ] **Step 5: Run the unit tests and confirm they pass.**

Run: `uv run python -m unittest test.test_contention -v`

Expected: All four tests in `TestApplySlowdown` PASS.

- [ ] **Step 6: Run the full test suite to confirm no regressions.**

Run: `uv run python -m unittest -v`

Expected: All existing tests still pass. The new fields default to `0.0`/`None`/`1.0` and don't affect ordering (`compare=False`), so existing `Job` constructions remain valid.

- [ ] **Step 7: Commit.**

```bash
git add common/job.py test/test_contention.py
git commit -m "$(cat <<'EOF'
Add contention tracking fields and applySlowdown to Job

Job now carries work_done_ideal_sec, last_event_time_sec, and
current_slowdown so it can self-track ideal progress under dynamic
slowdown. applySlowdown(s, t) accrues elapsed work, stores the new
factor, and refreshes priority to the projected ETA.
EOF
)"
```

---

## Task 2: Create `ContentionModel` with identity stub and `recompute`

**Files:**
- Create: `ClusterManager/contention.py`
- Test: `test/test_contention.py` (extend with `ContentionModel` tests)

- [ ] **Step 1: Add `ContentionModel` unit tests to `test/test_contention.py`.**

Append to `test/test_contention.py` (after `TestApplySlowdown` and before the `if __name__` guard):

```python
import logging
import simpy
from unittest.mock import MagicMock

from Cluster.cluster import Cluster
from ClusterManager.contention import ContentionModel


class TestContentionModel(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = MagicMock(spec=Cluster)
        self.model = ContentionModel(self.env, self.cluster)

    def test_identity_returns_one_for_all(self):
        """The default slowdown stub returns 1.0 for every running job."""
        jobs = [make_job(uuid=i, duration_sec=10) for i in (1, 2, 3)]
        factors = self.model.slowdown(jobs)
        self.assertEqual(factors, {1: 1.0, 2: 1.0, 3: 1.0})

    def test_recompute_updates_all_running_jobs(self):
        """recompute calls applySlowdown(1.0, env.now) on every running job."""
        jobs = [make_job(uuid=i, duration_sec=10) for i in (1, 2)]
        self.model.recompute(jobs)
        for j in jobs:
            self.assertEqual(j.current_slowdown, 1.0)
            self.assertEqual(j.last_event_time_sec, self.env.now)
            self.assertEqual(j.priority, self.env.now + 10)

    def test_recompute_logs_only_on_change(self):
        """A debug log fires when a job's factor changes, not when it stays the same."""

        class TwoXModel(ContentionModel):
            def slowdown(self, running_jobs):
                return {j.uuid: 2.0 for j in running_jobs}

        model = TwoXModel(self.env, self.cluster)
        jobs = [make_job(uuid=1, duration_sec=10)]

        # First call: 1.0 -> 2.0, expect a debug log.
        with self.assertLogs(level="DEBUG") as cm:
            model.recompute(jobs)
        self.assertTrue(any("slowdown 1.0 -> 2.0" in m for m in cm.output))

        # Second call at same env.now: 2.0 -> 2.0, expect NO debug log for that job.
        with self.assertLogs(level="DEBUG") as cm:
            # Emit a sentinel so assertLogs has something to capture even if no
            # slowdown-change log fires.
            logging.debug("sentinel")
            model.recompute(jobs)
        self.assertFalse(any("slowdown 2.0 -> 2.0" in m for m in cm.output))
```

- [ ] **Step 2: Run the new tests and confirm they fail.**

Run: `uv run python -m unittest test.test_contention.TestContentionModel -v`

Expected: All three tests fail with `ModuleNotFoundError: No module named 'ClusterManager.contention'`.

- [ ] **Step 3: Create `ClusterManager/contention.py` with the `ContentionModel` class.**

Create `ClusterManager/contention.py`:

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

- [ ] **Step 4: Run the new tests and confirm they pass.**

Run: `uv run python -m unittest test.test_contention.TestContentionModel -v`

Expected: All three tests PASS.

- [ ] **Step 5: Run the full test suite to confirm no regressions.**

Run: `uv run python -m unittest -v`

Expected: Everything still passes; the new module is unused outside tests at this point.

- [ ] **Step 6: Commit.**

```bash
git add ClusterManager/contention.py test/test_contention.py
git commit -m "$(cat <<'EOF'
Add ContentionModel with identity stub and recompute

ContentionModel mirrors SchedulingPolicy in placement: it holds the
pluggable slowdown(running_jobs) -> {uuid: factor} function. The
identity stub returns 1.0 for all jobs so contention modeling is a no-op
until a real implementation is plugged in. recompute() drives
Job.applySlowdown for every running job and logs only when a factor
actually changes.
EOF
)"
```

---

## Task 3: Wire `ContentionModel` into `ClusterManager`

**Files:**
- Modify: `ClusterManager/manager.py`
- Test: `test/test_contention.py` (extend with integration tests)

This is the load-bearing change: `executeOnCluster` and `completeOnCluster` get a recompute call and a SortedList rebuild. The guard interrupt logic is simplified because *any* running job's ETA can shift on a recompute.

- [ ] **Step 1: Add integration tests to `test/test_contention.py`.**

Append to `test/test_contention.py`:

```python
import copy
from unittest.mock import patch

from common.flags import FLAGS
from ClusterManager.manager import ClusterManager
from ClusterManager.scheduling import SchedDecision


class _TwoConcurrentSlow(ContentionModel):
    """Slowdown 2.0 whenever 2+ jobs are running, 1.0 otherwise."""
    def slowdown(self, running_jobs):
        jobs = list(running_jobs)
        s = 2.0 if len(jobs) >= 2 else 1.0
        return {j.uuid: s for j in jobs}


class _AlwaysTwoX(ContentionModel):
    """Slowdown 2.0 for every running job, regardless of count."""
    def slowdown(self, running_jobs):
        return {j.uuid: 2.0 for j in running_jobs}


def _make_alloc_job(uuid: int, duration_sec: float, arrival_time_sec: float, node: str) -> Job:
    job = Job(
        uuid=uuid,
        topology=TopoType.CLOS,
        shape=(1,),
        size=1,
        duration_sec=duration_sec,
        arrival_time_sec=arrival_time_sec,
    )
    job.allocation = {node: 1}
    return job


class TestClusterManagerContention(unittest.TestCase):
    """Integration tests: full simpy + ClusterManager + injected ContentionModel."""

    def _run(self, jobs, model_cls):
        """Drive the manager end-to-end with a mocked Cluster and the given contention model."""
        env = simpy.Environment()
        mock_cluster = MagicMock(spec=Cluster)
        # Total new work check uses closed_loop_threshold=0 (disabled).
        mgr = ClusterManager(env, cluster=mock_cluster, sim_njobs=len(jobs))
        mgr.contention_model = model_cls(env, mock_cluster)

        def feeder():
            for j in jobs:
                yield env.timeout(j.arrival_time_sec - env.now)
                mgr.submitJob(j)

        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=lambda j: (SchedDecision.ADMIT, j),
        ):
            sched_proc = env.process(mgr.schedule())
            env.process(feeder())
            env.run(until=sched_proc)

        return mgr

    def test_identity_baseline(self):
        """Identity stub: jct equals duration_sec for every job."""
        jobs = [
            _make_alloc_job(uuid=1, duration_sec=4, arrival_time_sec=0, node="n1"),
            _make_alloc_job(uuid=2, duration_sec=3, arrival_time_sec=10, node="n2"),
        ]
        mgr = self._run(jobs, ContentionModel)
        self.assertAlmostEqual(mgr.job_stats[1].jct_sec, 4)
        self.assertAlmostEqual(mgr.job_stats[1].slowdown, 1.0)
        self.assertAlmostEqual(mgr.job_stats[2].jct_sec, 3)
        self.assertAlmostEqual(mgr.job_stats[2].slowdown, 1.0)

    def test_fixed_2x_doubles_jct_when_alone(self):
        """A constant 2x slowdown stretches each non-overlapping job's JCT to 2*duration."""
        jobs = [
            _make_alloc_job(uuid=1, duration_sec=4, arrival_time_sec=0, node="n1"),
            # Arrives well after the first one would finish even at 2x.
            _make_alloc_job(uuid=2, duration_sec=3, arrival_time_sec=20, node="n2"),
        ]
        mgr = self._run(jobs, _AlwaysTwoX)
        self.assertAlmostEqual(mgr.job_stats[1].jct_sec, 8)   # 4 * 2
        self.assertAlmostEqual(mgr.job_stats[2].jct_sec, 6)   # 3 * 2

    def test_J1_J2_end_to_end(self):
        """The worked example: J1 dur=5 alone, J2 dur=2 arrives at t=1, both run at 2x while overlapping.
        Expected: J2 completes at t=3, J1 completes at t=6.
        """
        j1 = _make_alloc_job(uuid=1, duration_sec=5, arrival_time_sec=0, node="n1")
        j2 = _make_alloc_job(uuid=2, duration_sec=2, arrival_time_sec=1, node="n2")
        mgr = self._run([j1, j2], _TwoConcurrentSlow)
        self.assertAlmostEqual(mgr.job_stats[2].completion_time_sec, 3)
        self.assertAlmostEqual(mgr.job_stats[1].completion_time_sec, 6)
        self.assertAlmostEqual(mgr.job_stats[1].jct_sec, 6)
        self.assertAlmostEqual(mgr.job_stats[2].jct_sec, 2)
```

- [ ] **Step 2: Run the new integration tests and confirm they fail.**

Run: `uv run python -m unittest test.test_contention.TestClusterManagerContention -v`

Expected: Tests fail. `test_identity_baseline` likely passes (existing static math already produces `jct == duration` when jobs don't overlap), but `test_fixed_2x_doubles_jct_when_alone` and `test_J1_J2_end_to_end` will fail because `ClusterManager` does not yet call `contention_model.recompute()`.

- [ ] **Step 3: Add the `ContentionModel` import and instance to `ClusterManager`.**

In `ClusterManager/manager.py`, add the import near the other manager imports (around line 10):

```python
from ClusterManager.contention import ContentionModel
```

In `ClusterManager.__init__`, immediately after the existing line that creates `self.scheduler` (around line 74), add:

```python
        # The contention model recomputes per-job slowdowns whenever the
        # placement set changes (job admitted or completed) and pushes new
        # factors onto each running Job via Job.applySlowdown.
        self.contention_model = ContentionModel(env, cluster)
```

- [ ] **Step 4: Add the `_rebuildRunningQueue` helper to `ClusterManager`.**

In `ClusterManager/manager.py`, append the following method to the `ClusterManager` class (e.g., right after `logClusterStats`):

```python
    def _rebuildRunningQueue(self):
        """
        Re-sort the running queue after contention_model.recompute mutated
        job.priority. SortedList sorts on insert via Job's order=True
        comparator, so we have to dequeue everything and re-insert.
        """
        items = list(self.running_job_queue.slist)
        self.running_job_queue.slist.clear()
        for j in items:
            self.running_job_queue.enqueue(j)
```

- [ ] **Step 5: Modify `executeOnCluster` to recompute and rebuild on admit.**

In `ClusterManager/manager.py`, replace the entire body of `executeOnCluster` (currently at lines 268-290) with:

```python
    def executeOnCluster(self, job: Job):
        """
        Send the job to the cluster for execution. Timestamp the job with the scheduled
        time, register its initial contention state, and move it to the running queue
        for continuous tracking. Recompute slowdowns for all running jobs (the
        placement set has just changed) and re-sort the running queue accordingly.
        """
        # Scheduled time = time when the job is executed.
        job.updateQueueingTime(self.env.now)
        # Initialize contention-tracking state for this brand-new running job.
        # The recompute below will set last_event_time_sec and the real slowdown,
        # then overwrite priority via applySlowdown.
        job.work_done_ideal_sec = 0.0
        job.last_event_time_sec = None
        job.current_slowdown = 1.0
        # Placeholder priority; recompute below overwrites it.
        job.priority = self.env.now + job.duration_sec
        self.cluster.execute(job)
        # Move the running job into the running queue.
        self.new_job_queue.remove(job)
        self.running_job_queue.enqueue(job)
        # Placement set changed: recompute for every running job (including the
        # new one), then re-sort so the head reflects the new earliest ETA.
        self.contention_model.recompute(self.running_job_queue.slist)
        self._rebuildRunningQueue()
        # Wake the guard. If it was on a timer, interrupt it; if it was waiting
        # for a job to be enqueued, signal arrival. We always interrupt under
        # the recompute model because any running job's ETA may have changed.
        if self.next_completion > self.env.now:
            self.running_guard_proc.interrupt()
        else:
            self.event_running.trigger()
        self.logClusterStats()
```

The earlier conditional `... and job.priority < self.next_completion` is intentionally dropped — *any* running job's ETA can change on a recompute, so the guard must always re-read the head.

- [ ] **Step 6: Modify `completeOnCluster` to recompute and rebuild on complete.**

In `ClusterManager/manager.py`, replace the body of `completeOnCluster` (currently at lines 292-305) with:

```python
    def completeOnCluster(self, job: Job):
        """
        Send the completing job to the cluster to free up resources. Update the
        job's stats. Recompute slowdowns for the remaining running jobs (their
        contention landscape just changed) and re-sort the running queue.
        """
        self.cluster.complete(job)
        # Update job statistics.
        job.completion_time_sec = self.env.now
        job.jct_sec = self.env.now - job.arrival_time_sec
        job.slowdown = job.jct_sec / job.duration_sec
        self.job_stats[job.uuid] = job
        # Placement set changed: refresh the remaining running jobs.
        self.contention_model.recompute(self.running_job_queue.slist)
        self._rebuildRunningQueue()
        # No guard interrupt needed: completeOnCluster is invoked from inside
        # the guard's own loop; the guard naturally peeks the (re-sorted) head
        # on its next iteration.
        # Notify the main schedule loop.
        self.event_completion.trigger()
        self.logClusterStats()
```

- [ ] **Step 7: Run the integration tests and confirm they pass.**

Run: `uv run python -m unittest test.test_contention.TestClusterManagerContention -v`

Expected: All three integration tests PASS:
- `test_identity_baseline` — JCT equals ideal duration for both jobs.
- `test_fixed_2x_doubles_jct_when_alone` — JCTs are exactly `2 * duration`.
- `test_J1_J2_end_to_end` — J2 completes at t=3, J1 at t=6.

- [ ] **Step 8: Run the full test suite to confirm no regressions.**

Run: `uv run python -m unittest -v`

Expected: All existing tests still pass. The identity-stub default keeps behavior identical to the pre-change baseline for any test that doesn't inject a custom contention model.

- [ ] **Step 9: Commit.**

```bash
git add ClusterManager/manager.py test/test_contention.py
git commit -m "$(cat <<'EOF'
Wire ContentionModel into ClusterManager admit/complete paths

executeOnCluster and completeOnCluster now call
contention_model.recompute on the running job set after the placement
changes, then rebuild the SortedList so its head reflects the new
earliest ETA. The admit-path guard interrupt is unconditional because
any running job's ETA can shift on a recompute; the complete path
relies on the guard naturally re-reading the head on its next loop
iteration. Default identity stub preserves existing simulation outputs.
EOF
)"
```

---

## Task 4: Remove the stale `Cluster.complete` TODO

**Files:**
- Modify: `Cluster/cluster.py` (remove lines 155-157)

The TODO comment in `Cluster.complete` flagged that the actual completion time may diverge from the theoretical one once runtime dynamics are modeled. That concern is now resolved at the manager layer: completion fires when the dynamic ETA matures, so the comment is stale.

- [ ] **Step 1: Remove the TODO block from `Cluster.complete`.**

In `Cluster/cluster.py`, locate `Cluster.complete` and delete the three TODO lines (currently lines 155-157):

```python
        # TODO: this method is called when a job completes at the theorectical completion
        # time. The actual completion time may be ahead or behind if we model failures or
        # runtime dynamics. Need to refactor this class to handle such cases.
```

The method body should end at the `for node_id, num_xpu in job.allocation.items(): self.nodes[node_id].free(num_xpu)` loop with no trailing comments.

- [ ] **Step 2: Run the full test suite.**

Run: `uv run python -m unittest -v`

Expected: All tests still pass. This is a comment-only deletion.

- [ ] **Step 3: Commit.**

```bash
git add Cluster/cluster.py
git commit -m "$(cat <<'EOF'
Remove stale TODO in Cluster.complete

The comment flagged that real completion times might diverge from the
theoretical ones once runtime dynamics were modeled. The contention
model now resolves that at the manager layer: completion fires when
the dynamic ETA matures.
EOF
)"
```

---

## Self-Review Notes

Spec coverage:
- Section 2 (semantics) → Task 1 (`applySlowdown` math + worked-example test).
- Section 3 (triggers: admit/complete only) → Task 3 steps 5–6.
- Section 4 (architecture) → all four tasks form the structure.
- Section 5 (Job changes, fields + method, with comments) → Task 1 steps 3–4.
- Section 6 (`ContentionModel` interface, identity stub) → Task 2 steps 3.
- Section 7 (manager wiring, executeOnCluster + completeOnCluster + _rebuildRunningQueue) → Task 3 steps 3–6.
- Section 8 edge cases:
  - FP clamp → Task 1 test 3.
  - First-call no accumulation → Task 1 test 4.
  - Always-interrupt-after-recompute on admit → Task 3 step 5 (comment + dropped conditional).
  - No-interrupt on complete → Task 3 step 6 (comment).
  - `closed_loop_threshold` and trace-generation invariants are unaffected by construction (no field renamed, no semantics changed).
- Section 9 (debug log on slowdown change) → Task 2 step 1 test 3, implementation in step 3.
- Section 10 (10 tests) → all tests are scheduled across Tasks 1–3.
- TODO removal called out in Section 4 / 7 → Task 4.

Method/field name consistency: `applySlowdown`, `recompute`, `slowdown`, `_rebuildRunningQueue`, `contention_model`, `work_done_ideal_sec`, `last_event_time_sec`, `current_slowdown` are spelled identically across spec and plan tasks.

No placeholders. Every step contains the actual code or command an engineer needs.
