import logging
import unittest
from unittest.mock import MagicMock, patch

import simpy

from common.job import Job, TopoType
from common.utils import spec_parser
from Cluster.cluster import Cluster
from ClusterManager.contention import ContentionModel, MinTopoEdge
from ClusterManager.manager import ClusterManager
from ClusterManager.scheduling import SchedDecision


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

    def test_speedup_advances_work_faster_than_wall(self):
        """A slowdown < 1.0 (speedup) advances ideal work faster than wall time.
        Spec Section 8 explicitly admits this case as mathematically valid."""
        job = make_job(uuid=1, duration_sec=10)
        job.applySlowdown(1.0, 0)            # baseline rate
        job.applySlowdown(0.5, 4)            # speedup: 1 wall sec = 2 ideal sec
        self.assertEqual(job.work_done_ideal_sec, 4)
        # Remaining ideal = 6, at s=0.5 → 6*0.5 = 3 wall sec to finish from t=4.
        self.assertEqual(job.priority, 7)
        job.applySlowdown(1.0, 6)            # back to baseline
        # In the [4,6] window at s=0.5, work += 2/0.5 = 4 ideal. Total = 8.
        self.assertEqual(job.work_done_ideal_sec, 8)
        self.assertEqual(job.priority, 8)    # 6 + (10-8)*1.0

    def test_contention_fields_excluded_from_equality(self):
        """The three contention-tracking fields use compare=False, so two jobs
        differing only in those fields must still compare equal. SortedList
        correctness depends on this invariant — silently breakable by removing
        the compare=False annotations."""
        j1 = make_job(uuid=1, duration_sec=10)
        j2 = make_job(uuid=1, duration_sec=10)
        self.assertEqual(j1, j2)
        # Mutate the contention fields on j1 directly (do NOT call
        # applySlowdown, which would also mutate priority — priority IS in
        # the equality comparison).
        j1.work_done_ideal_sec = 3.5
        j1.last_event_time_sec = 7
        j1.current_slowdown = 2.0
        self.assertEqual(
            j1, j2,
            "contention-tracking fields must use compare=False; SortedList "
            "ordering depends on this."
        )


class TestContentionModel(unittest.TestCase):
    """Unit tests for the ContentionModel API surface (slowdown stub,
    onAdmit dispatch, log behavior). These use a MagicMock cluster so the
    findImpactedJobs path is short-circuited via the impacted-set being
    {triggering_job} only — the unit tests pin the per-job dispatch
    invariants, not the cross-job impacted-set detection logic."""

    def setUp(self):
        self.env = simpy.Environment()
        # MagicMock cluster: routeJobPaths returns no edges, so findImpactedJobs
        # short-circuits to {triggering_job} on admit / set() on complete.
        self.cluster = MagicMock(spec=Cluster)
        self.cluster.routeJobPaths.side_effect = lambda job: iter([])
        self.model = ContentionModel(self.env, self.cluster)

    def test_identity_returns_one_for_all_impacted(self):
        """The default slowdown stub returns 1.0 for every impacted job."""
        jobs = [make_job(uuid=i, duration_sec=10) for i in (1, 2, 3)]
        factors = self.model.slowdown(jobs, {j.uuid: [] for j in jobs})
        self.assertEqual(factors, {1: 1.0, 2: 1.0, 3: 1.0})

    def test_onAdmit_updates_admitted_job(self):
        """onAdmit calls applySlowdown(1.0, env.now) on the admitted job
        when the cluster yields no shared paths."""
        admitted = make_job(uuid=1, duration_sec=10)
        running = [admitted]
        self.model.onAdmit(admitted, running)
        self.assertEqual(admitted.current_slowdown, 1.0)
        self.assertEqual(admitted.last_event_time_sec, self.env.now)
        self.assertEqual(admitted.priority, self.env.now + 10)

    def test_onAdmit_logs_only_on_factor_change(self):
        """A debug log fires only when the factor changes."""

        class TwoXModel(ContentionModel):
            def slowdown(self, impacted_jobs, min_topologies):
                return {j.uuid: 2.0 for j in impacted_jobs}

        model = TwoXModel(self.env, self.cluster)
        admitted = make_job(uuid=1, duration_sec=10)

        # First admit: 1.0 -> 2.0, expect a debug log.
        with self.assertLogs(level="DEBUG") as cm:
            model.onAdmit(admitted, [admitted])
        self.assertTrue(any("slowdown 1.0 -> 2.0" in m for m in cm.output))

        # Second admit at the same env.now: 2.0 -> 2.0, expect NO log for it.
        with self.assertLogs(level="DEBUG") as cm:
            logging.debug("sentinel")
            model.onAdmit(admitted, [admitted])
        self.assertFalse(any("slowdown 2.0 -> 2.0" in m for m in cm.output))

    def test_onComplete_empty_impacted_is_noop(self):
        """onComplete on a running set with no overlap (impacted = set())
        does not raise and does not call applySlowdown. This is the path
        exercised by completeOnCluster when the last running job exits."""
        completed = make_job(uuid=1, duration_sec=10)
        with patch.object(Job, "applySlowdown", autospec=True) as spy:
            self.model.onComplete(completed, [])
        self.assertEqual(spy.call_count, 0)

    def test_onAdmit_isclose_guard_suppresses_jitter(self):
        """When the slowdown function returns FP-jittered factors that are
        math.isclose to the prior value, no debug log fires."""

        class JitterModel(ContentionModel):
            def __init__(self, env, cluster):
                super().__init__(env, cluster)
                self.calls = 0

            def slowdown(self, impacted_jobs, min_topologies):
                self.calls += 1
                factor = 2.0 if self.calls == 1 else 2.0 + 1e-15
                return {j.uuid: factor for j in impacted_jobs}

        model = JitterModel(self.env, self.cluster)
        admitted = make_job(uuid=1, duration_sec=10)

        # First admit: 1.0 -> 2.0, log fires.
        with self.assertLogs(level="DEBUG") as cm:
            model.onAdmit(admitted, [admitted])
        self.assertTrue(any("slowdown" in m for m in cm.output))

        # Second admit: 2.0 -> 2.0 + 1e-15, math.isclose default = True, no log.
        with self.assertLogs(level="DEBUG") as cm:
            logging.debug("sentinel")
            model.onAdmit(admitted, [admitted])
        self.assertFalse(any("slowdown" in m for m in cm.output))

    def test_onAdmit_calls_applySlowdown_exactly_once_per_impacted_job(self):
        """onAdmit must invoke Job.applySlowdown exactly once per impacted job."""
        admitted = make_job(uuid=1, duration_sec=10)
        with patch.object(Job, "applySlowdown", autospec=True) as spy:
            self.model.onAdmit(admitted, [admitted])
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args_list[0].args[0].uuid, 1)


class _TwoConcurrentSlow(ContentionModel):
    """Slowdown 2.0 whenever 2+ impacted jobs are present, 1.0 otherwise.

    Note: under the new partial-recompute model, the slowdown function only
    sees the impacted subset of running jobs. The TestClusterManagerContention
    `_run` helper mocks Cluster.routeJobPaths so every job appears to share a
    single synthetic link, which forces every running job into the impacted
    set on every event. With that mock setup, len(impacted_jobs) equals
    len(running_jobs) and this subclass behaves identically to its
    pre-refactor "2x if 2+ jobs running" semantics."""
    def slowdown(self, impacted_jobs, min_topologies):
        jobs = list(impacted_jobs)
        s = 2.0 if len(jobs) >= 2 else 1.0
        return {j.uuid: s for j in jobs}


class _AlwaysTwoX(ContentionModel):
    """Slowdown 2.0 for every impacted job, regardless of count."""
    def slowdown(self, impacted_jobs, min_topologies):
        return {j.uuid: 2.0 for j in impacted_jobs}


def _make_t2d_job(uuid: int, shape, node_ranks: list[str], duration_sec: float = 10.0) -> Job:
    """Build a T2D Job and populate its allocation in the given rank order.
    Mirrors the fixture in test/test_cluster.py; duplicated here to keep
    test files self-contained."""
    job = Job(
        uuid=uuid,
        topology=TopoType.T2D,
        shape=shape,
        size=len(node_ranks),
        duration_sec=duration_sec,
        arrival_time_sec=0,
    )
    for name in node_ranks:
        job.addToAllocation(name)
    return job


def _make_alloc_job(uuid: int, duration_sec: float, arrival_time_sec: float, node: str) -> Job:
    job = Job(
        uuid=uuid,
        topology=TopoType.CLOS,
        shape=(1,),
        size=1,
        duration_sec=duration_sec,
        arrival_time_sec=arrival_time_sec,
    )
    job.allocation = {0: {"node": node, "num_xpu": 1}}
    return job


class TestClusterManagerContention(unittest.TestCase):
    """Integration tests: full simpy + ClusterManager + injected ContentionModel."""

    def _run(self, jobs, model_cls):
        """Drive the manager end-to-end with a mocked Cluster and the given contention model."""
        env = simpy.Environment()
        mock_cluster = MagicMock(spec=Cluster)
        # Configure routeJobPaths so every job appears to share a single
        # synthetic link. This keeps findImpactedJobs returning the full
        # running set on every event, preserving the pre-refactor
        # "every job sees the same contention state" semantics that the
        # JCT-math tests below depend on.
        shared_link = MagicMock()
        shared_link.name = "shared-link"
        shared_link.speed_gbps = 100.0
        shared_link.flow_count = 1
        shared_link.latency_ns = 100.0
        mock_cluster.routeJobPaths.side_effect = lambda job: iter(
            [(0, 1, [shared_link])]
        )
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
            env.process(mgr.schedule())
            env.process(feeder())
            # Note: schedule() blocks on event_arrival once the new-job queue empties,
            # so we cannot use until=sched_proc. simpy exits when no scheduled events
            # remain. The assertion below catches a wrong-reason early termination.
            env.run()

        # Defense against the test passing because env ran out of events before all
        # jobs actually completed. If the running queue is non-empty here, something
        # caused env.run() to return prematurely and the test result is unreliable.
        self.assertEqual(
            len(mgr.running_job_queue), 0,
            "env.run() returned with jobs still in the running queue; test result is invalid.",
        )
        return mgr

    def test_identity_baseline(self):
        """Identity stub with overlapping jobs: JCT still equals duration_sec, but the
        admit and complete onAdmit/onComplete paths are both exercised. If onAdmit/
        onComplete were deleted, this test would still pass for non-overlapping jobs —
        overlap is what makes it a real regression guard."""
        jobs = [
            _make_alloc_job(uuid=1, duration_sec=10, arrival_time_sec=0, node="n1"),
            # Arrives while J1 is still running, so the onAdmit path fires with two
            # jobs in the running queue. J2 finishes first, so the onComplete path
            # also fires while J1 is still alive.
            _make_alloc_job(uuid=2, duration_sec=4, arrival_time_sec=2, node="n2"),
        ]
        mgr = self._run(jobs, ContentionModel)
        # Identity stub means slowdown is always 1.0; JCT equals ideal duration.
        self.assertAlmostEqual(mgr.job_stats[1].jct_sec, 10)
        self.assertAlmostEqual(mgr.job_stats[1].slowdown, 1.0)
        self.assertAlmostEqual(mgr.job_stats[2].jct_sec, 4)
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
        """The worked example from the design spec: J1 (dur=5 ideal) at t=0,
        J2 (dur=1 ideal) arrives at t=1; symmetric 2x slowdown applies while
        both run concurrently. J2 (1 ideal sec at 2x = 2 wall sec) completes
        at t=3, J1 completes at t=6."""
        j1 = _make_alloc_job(uuid=1, duration_sec=5, arrival_time_sec=0, node="n1")
        j2 = _make_alloc_job(uuid=2, duration_sec=1, arrival_time_sec=1, node="n2")  # J2 dur=1 ideal; under 2x slowdown that's 2 wall seconds, completing at t=3 — matches the spec's J1/J2 worked example
        mgr = self._run([j1, j2], _TwoConcurrentSlow)
        self.assertAlmostEqual(mgr.job_stats[2].completion_time_sec, 3)
        self.assertAlmostEqual(mgr.job_stats[1].completion_time_sec, 6)
        self.assertAlmostEqual(mgr.job_stats[1].jct_sec, 6)
        self.assertAlmostEqual(mgr.job_stats[2].jct_sec, 2)

    def test_three_concurrent_jobs_2x(self):
        """Three jobs admitted in sequence, all slowed 2x while ≥2 are running.
        Verifies contention state propagates correctly across multiple admit
        and complete events with more than two jobs in flight.

        Trajectory under _TwoConcurrentSlow (s=2.0 when ≥2 jobs run):
          t=0: J1 admitted alone, s=1, priority=10.
          t=2: J2 admitted, s=2 for both.
              J1.work=2, priority=18.  J2.priority=10.
          t=4: J3 admitted, s=2 for all three.
              J1.work=3, priority=18.  J2.work=1, priority=10.  J3.priority=8.
          t=8: J3 completes. 2 jobs left, s=2 still.
              J1.work=5, priority=18.  J2.work=3, priority=10.
          t=10: J2 completes. 1 job, s=1.
              J1.work=6, priority=14.
          t=14: J1 completes.
        """
        jobs = [
            _make_alloc_job(uuid=1, duration_sec=10, arrival_time_sec=0, node="n1"),
            _make_alloc_job(uuid=2, duration_sec=4, arrival_time_sec=2, node="n2"),
            _make_alloc_job(uuid=3, duration_sec=2, arrival_time_sec=4, node="n3"),
        ]
        mgr = self._run(jobs, _TwoConcurrentSlow)
        self.assertAlmostEqual(mgr.job_stats[3].completion_time_sec, 8)
        self.assertAlmostEqual(mgr.job_stats[2].completion_time_sec, 10)
        self.assertAlmostEqual(mgr.job_stats[1].completion_time_sec, 14)
        self.assertAlmostEqual(mgr.job_stats[3].jct_sec, 4)   # 8 - 4
        self.assertAlmostEqual(mgr.job_stats[2].jct_sec, 8)   # 10 - 2
        self.assertAlmostEqual(mgr.job_stats[1].jct_sec, 14)  # 14 - 0

    def test_admit_reorders_running_queue_head(self):
        """Identity stub: a later-arriving short job ends up at the head of
        the running queue ahead of the earlier-arriving long job. Exercises
        the SortedList rebuild and the guard interrupt path under reordering.
        If the rebuild were skipped (or the interrupt suppressed), J2's
        completion event would be missed and the test would hang or fail."""
        jobs = [
            _make_alloc_job(uuid=1, duration_sec=20, arrival_time_sec=0, node="n1"),
            _make_alloc_job(uuid=2, duration_sec=2, arrival_time_sec=5, node="n2"),
        ]
        mgr = self._run(jobs, ContentionModel)
        # J2 admitted at t=5 with dur=2 → completes at t=7 (head of queue,
        # ahead of J1's t=20 ETA).
        self.assertAlmostEqual(mgr.job_stats[2].completion_time_sec, 7)
        self.assertAlmostEqual(mgr.job_stats[1].completion_time_sec, 20)
        self.assertLess(
            mgr.job_stats[2].completion_time_sec,
            mgr.job_stats[1].completion_time_sec,
            "later-arriving short job must complete before earlier long job",
        )


C1_SPEC = "Cluster/models/c1.json"


class TestComputeMinTopology(unittest.TestCase):
    """Unit tests for ContentionModel.computeMinTopology on the c1.json 4x4 T2D."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster)
        # Per-link latency is uniform from FLAGS.link_latency_ns (default 500.0).
        sample_link = next(iter(self.cluster.links.values()))
        self.link_speed_gbps = sample_link.speed_gbps
        self.link_lat_ns = sample_link.latency_ns

    def test_non_torus_job_returns_empty_list(self):
        """A CLOS Job's min topology is []."""
        job = Job(
            uuid=99,
            topology=TopoType.CLOS,
            shape=(1,),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        job.addToAllocation("x0-y0")
        self.assertEqual(self.model.computeMinTopology(job), [])

    def test_single_uncontended_edge_full_bandwidth_one_hop_latency(self):
        """A (2,1) job alone: forward edge sees full bandwidth and 1-hop latency."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        topo = self.model.computeMinTopology(job)
        # Comm pattern is [(0,1,1.0), (1,0,1.0)] — two edges.
        self.assertEqual(len(topo), 2)
        forward = topo[0]
        self.assertEqual((forward.src_rank, forward.dst_rank), (0, 1))
        self.assertEqual(forward.eff_bw_gbps, self.link_speed_gbps)
        self.assertEqual(forward.eff_lat_ns, self.link_lat_ns)

    def test_wrap_edge_three_hops_cumulative_latency(self):
        """The wrap edge of a (2,1) job traverses 3 hops; latency sums."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        topo = self.model.computeMinTopology(job)
        wrap = topo[1]
        self.assertEqual((wrap.src_rank, wrap.dst_rank), (1, 0))
        self.assertEqual(wrap.eff_bw_gbps, self.link_speed_gbps)  # each link flow_count == 1
        self.assertEqual(wrap.eff_lat_ns, 3 * self.link_lat_ns)

    def test_all_links_shared_between_two_jobs_on_same_row(self):
        """When two (2,1) jobs are placed at [x0-y0,x1-y0] and
        [x2-y0,x3-y0], all 4 +x links of row y=0 are shared between them
        (forward edges and wrap edges interleave around the ring).
        Every link sees flow_count == 2, so every edge of either job's
        min topology has eff_bw_gbps == speed_gbps / 2."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_a)
        self.cluster.execute(job_b)
        topo_a = self.model.computeMinTopology(job_a)
        # Forward edge (rank 0 -> 1) routes x0-y0 -> x1-y0; that link is
        # shared by job_b's wrap, so flow_count == 2.
        forward_a = topo_a[0]
        self.assertEqual(forward_a.eff_bw_gbps, self.link_speed_gbps / 2)
        # Wrap edge (rank 1 -> 0) routes x1->x2->x3->x0; all 3 links shared
        # with job_b, so flow_count == 2 on each.
        wrap_a = topo_a[1]
        self.assertEqual(wrap_a.eff_bw_gbps, self.link_speed_gbps / 2)

    def test_self_contention_two_edges_same_link(self):
        """If two flows of the same job route over the same physical
        link, flow_count on that link is 2 and the edge sees
        eff_bw = speed/2.

        Direct construction of self-contention via the current ring
        comm pattern on c1.json is not achievable for small shapes (the
        forward and wrap edges of a (2,1) ring use disjoint physical
        links). To demonstrate self-contention we manually bump the
        flow_count on a link to simulate a second comm-pattern edge of
        the same job crossing the same link."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        # Simulate a second flow of this job that also crosses x0->x1.
        self.cluster.links["x0-y0-p1:x1-y0-p0"].incFlow()
        topo = self.model.computeMinTopology(job)
        forward = topo[0]
        self.assertEqual(forward.eff_bw_gbps, self.link_speed_gbps / 2)
        # Restore counter to keep cluster state consistent if tests share fixtures.
        self.cluster.links["x0-y0-p1:x1-y0-p0"].decFlow()

    def test_bottleneck_min_with_mixed_flow_counts(self):
        """When the path has mixed flow_count (one bottleneck link at
        flow_count=2 and other links at flow_count=1), eff_bw_gbps equals
        the minimum fair-share — speed/2 — not an average. Pins the
        bottleneck-min semantics versus a sum or average regression."""
        # Use the (2,1) wrap edge: rank 1 -> 0 routes x1->x2->x3->x0 (3 hops).
        # All three links start at flow_count=1 from the job's own admit.
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        # Manually bump only the middle link (x2-y0->x3-y0) so its flow_count
        # is 2 while x1-y0->x2-y0 and x3-y0->x0-y0 remain at 1.
        self.cluster.links["x2-y0-p1:x3-y0-p0"].incFlow()
        topo = self.model.computeMinTopology(job)
        wrap = topo[1]  # the (1, 0) edge
        # min(speed/1, speed/2, speed/1) = speed/2.
        self.assertEqual(wrap.eff_bw_gbps, self.link_speed_gbps / 2)
        # Cumulative latency unaffected by flow_count: 3 hops.
        self.assertEqual(wrap.eff_lat_ns, 3 * self.link_lat_ns)
        # Restore counter so other tests don't see leaked state.
        self.cluster.links["x2-y0-p1:x3-y0-p0"].decFlow()


class TestFindImpactedJobs(unittest.TestCase):
    """Unit tests for ContentionModel.findImpactedJobs on the c1.json 4x4 T2D."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster)

    def test_disjoint_paths_admit_returns_only_trigger(self):
        """Two jobs on rows y=0 and y=2 share no links: impacted={trigger}."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x0-y2", "x1-y2"])
        self.cluster.execute(job_b)  # B running first
        impacted = self.model.findImpactedJobs(job_a, [job_b], is_admit=True)
        self.assertEqual(impacted, {job_a})

    def test_shared_wraparound_admit_returns_both(self):
        """Wraparound paths of two jobs on row y=0 cross all 4 +x links of
        that row; admitting A while B runs returns {A, B}."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_b)  # B already running; its links are bumped
        impacted = self.model.findImpactedJobs(job_a, [job_b], is_admit=True)
        self.assertEqual(impacted, {job_a, job_b})

    def test_shared_wraparound_complete_excludes_trigger(self):
        """Completing A returns only the other job(s) sharing links; A is
        excluded because it's no longer running."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_a)
        self.cluster.execute(job_b)
        # Note: complete() is the caller's responsibility before this is called;
        # we are testing findImpactedJobs alone, so just pass the post-state.
        impacted = self.model.findImpactedJobs(job_a, [job_b], is_admit=False)
        self.assertEqual(impacted, {job_b})

    def test_non_torus_trigger_admit_returns_trigger_only(self):
        """A non-torus triggering job has no paths to share. Admit returns
        only the trigger itself."""
        clos_job = Job(
            uuid=99,
            topology=TopoType.CLOS,
            shape=(1,),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        clos_job.addToAllocation("x0-y0")
        running_job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x2-y2", "x3-y2"])
        self.cluster.execute(running_job)
        impacted = self.model.findImpactedJobs(clos_job, [running_job], is_admit=True)
        self.assertEqual(impacted, {clos_job})

    def test_non_torus_trigger_complete_returns_empty(self):
        """A non-torus triggering job on completion impacts nobody."""
        clos_job = Job(
            uuid=99,
            topology=TopoType.CLOS,
            shape=(1,),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        clos_job.addToAllocation("x0-y0")
        running_job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x2-y2", "x3-y2"])
        self.cluster.execute(running_job)
        impacted = self.model.findImpactedJobs(clos_job, [running_job], is_admit=False)
        self.assertEqual(impacted, set())


class TestOnAdmitOnComplete(unittest.TestCase):
    """End-to-end tests for ContentionModel.onAdmit / onComplete with the
    identity stub slowdown."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster)

    def test_disjoint_second_admit_does_not_touch_first_job(self):
        """Two jobs with disjoint paths: admitting the second job calls
        applySlowdown only on the admitted job (the disjoint earlier job is
        not impacted, so it must not be touched)."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x0-y2", "x1-y2"])
        # Admit Job A first.
        self.cluster.execute(job_a)
        self.model.onAdmit(job_a, [job_a])
        # Now admit Job B; running_jobs at this point is [job_a, job_b]
        # (Cluster.execute is the caller's responsibility before onAdmit).
        self.cluster.execute(job_b)
        with patch.object(Job, "applySlowdown", autospec=True) as spy:
            self.model.onAdmit(job_b, [job_a, job_b])
        called_uuids = [call.args[0].uuid for call in spy.call_args_list]
        # Only Job B (the trigger) should receive applySlowdown; Job A is disjoint.
        self.assertEqual(called_uuids, [2])

    def test_overlapping_second_admit_touches_both_jobs(self):
        """Two jobs whose wraparound paths overlap: admitting the second
        calls applySlowdown on both jobs."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_a)
        self.model.onAdmit(job_a, [job_a])
        self.cluster.execute(job_b)
        with patch.object(Job, "applySlowdown", autospec=True) as spy:
            self.model.onAdmit(job_b, [job_a, job_b])
        called_uuids = sorted(call.args[0].uuid for call in spy.call_args_list)
        # Both Job A (impacted via shared wrap links) and Job B (trigger) are touched.
        self.assertEqual(called_uuids, [1, 2])

    def test_overlapping_complete_touches_only_remaining_job(self):
        """Two jobs with shared wraparound paths: completing one calls
        applySlowdown only on the remaining job (the completed job is
        excluded from the impacted set)."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        # Both jobs running and admitted.
        self.cluster.execute(job_a)
        self.model.onAdmit(job_a, [job_a])
        self.cluster.execute(job_b)
        self.model.onAdmit(job_b, [job_a, job_b])
        # Now complete Job A. Caller (Manager) removes A from running set
        # and decrements link flows BEFORE calling onComplete.
        self.cluster.complete(job_a)
        with patch.object(Job, "applySlowdown", autospec=True) as spy:
            self.model.onComplete(job_a, [job_b])
        called_uuids = [call.args[0].uuid for call in spy.call_args_list]
        # Only Job B (the remaining impacted job) should be touched; Job A is
        # excluded from the impacted set on completion.
        self.assertEqual(called_uuids, [2])


class TestRunAstraSim(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mock_cluster = MagicMock(spec=Cluster)
        self.cm = ContentionModel(self.env, self.mock_cluster)

    def test_run_astra_sim_sets_astra_ideal_dur_nsec(self):
        job = Job(
            uuid=99,
            topology=TopoType.T2D,
            shape=(2, 2),
            size=4,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        self.assertIsNone(job.astra_ideal_dur_nsec)
        with patch(
            "ClusterManager.contention.astra_sim.run_astra",
            return_value=3.14,
        ) as mock_run:
            self.cm.runAstraSim(job)
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["uuid"], 99)
        self.assertEqual(kwargs["shape"], (2, 2))
        self.assertEqual(job.astra_ideal_dur_nsec, 3.14)

    def test_run_astra_sim_coerces_float_shape_to_int(self):
        # Job.shape is typed as Tuple[Union[float, int], ...]; for torus
        # jobs we expect ints, but coerce defensively.
        job = Job(
            uuid=100,
            topology=TopoType.T2D,
            shape=(2.0, 2.0),
            size=4,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        with patch(
            "ClusterManager.contention.astra_sim.run_astra",
            return_value=1.0,
        ) as mock_run:
            self.cm.runAstraSim(job)
        self.assertEqual(mock_run.call_args.kwargs["shape"], (2, 2))

    def test_run_astra_sim_is_idempotent_when_field_already_set(self):
        # If astra_ideal_dur_nsec is already populated, the simulator
        # must not run a second time and the existing value must not
        # be overwritten. Output is deterministic per (uuid, shape),
        # so re-running is pure waste.
        job = Job(
            uuid=77,
            topology=TopoType.T2D,
            shape=(2, 2),
            size=4,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        job.astra_ideal_dur_nsec = 123.0
        with patch(
            "ClusterManager.contention.astra_sim.run_astra",
            return_value=999.0,
        ) as mock_run:
            self.cm.runAstraSim(job)
        mock_run.assert_not_called()
        self.assertEqual(job.astra_ideal_dur_nsec, 123.0)

    def test_run_astra_sim_short_circuits_single_npu_shape(self):
        # astra-sim's analytical backend rejects npus_count <= 1, and a
        # single-rank torus has no collective communication to simulate.
        job = Job(
            uuid=55,
            topology=TopoType.T3D_NT,
            shape=(1, 1, 1),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        with patch(
            "ClusterManager.contention.astra_sim.run_astra",
        ) as mock_run:
            self.cm.runAstraSim(job)
        mock_run.assert_not_called()
        self.assertEqual(job.astra_ideal_dur_nsec, 1.0)


if __name__ == "__main__":
    unittest.main()
