import logging
import unittest
from unittest.mock import MagicMock

import simpy

from Cluster.cluster import Cluster
from ClusterManager.contention import ContentionModel
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


if __name__ == "__main__":
    unittest.main()
