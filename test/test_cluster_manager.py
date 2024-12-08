import copy
import simpy
import unittest
from unittest.mock import patch, call

from common.flags import *
from common.job import Job, TopoType
from ClusterManager.manager import ClusterManager
from ClusterManager.scheduling import SchedDecision

JOB1 = Job(
    uuid=1,
    topology=TopoType.CLOS,
    shape=(1,),
    size=1,
    duration_sec=1,
    arrival_time_sec=0,
)


class TestClusterManager(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mgr = ClusterManager(self.env, cluster=None)

    def wgen_helper(self, jobs: list[Job]):
        for job in jobs:
            yield self.env.timeout(job.arrival_time_sec - self.env.now)
            self.mgr.submitJob(job)

    def test_zero_job(self):
        """
        Verify that there is zero job submitted and scheduled.
        """
        job1 = copy.deepcopy(JOB1)
        job1.arrival_time_sec = 2
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=SchedDecision.ADMIT,
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=1)
            # The job being scheduled never arrives before simulation terminates.
            self.mgr.scheduler.place.assert_not_called()
            # Simulation ends at t = 1, new job queue should be empty.
            self.assertTrue(self.mgr.new_job_queue.empty())

    def test_one_job_admit(self):
        """
        Verify that there is one job submitted and scheduled.
        """
        job1 = copy.deepcopy(JOB1)
        job1.arrival_time_sec = 2
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=SchedDecision.ADMIT,
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=3)
            # The job being scheduled arrives at t = 2.
            self.mgr.scheduler.place.assert_called_once_with(job1)
            # Simulation ends at t = 3, new job queue should be empty since the only
            # job is admitted.
            self.assertTrue(self.mgr.new_job_queue.empty())

    def test_multi_job_admit(self):
        """
        Verify that multiple jobs submitted at different times are all admitted.
        """
        job1 = copy.deepcopy(JOB1)
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 2
        job3 = copy.deepcopy(JOB1)
        job3.arrival_time_sec = 3

        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=SchedDecision.ADMIT,
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2, job3]))
            self.env.run(until=5)
            # Three jobs are submitted.
            self.assertEqual(self.mgr.scheduler.place.call_count, 3)
            # New job queue should be empty since all jobs are admitted.
            self.assertTrue(self.mgr.new_job_queue.empty())

    def test_one_job_reject(self):
        """
        Verify that there is one job submitted and rejected.
        """
        job1 = copy.deepcopy(JOB1)
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=SchedDecision.REJECT,
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=DEFERRED_SCHED_SEC + 1)
            # The job is rejected for the first time at t = 0. It is deferred until
            # t = DEFERRED_SCHED_SEC, when it gets rejected again. And the job is
            # deferred util t = 2 * DEFERRED_SCHED_SEC. So by the time simulation ends,
            # the job is still deferred.
            self.assertTrue(self.mgr.new_job_queue.empty())
            # There should be two scheduling attempts: one for the initial job and
            # one when the first deferral is over.
            self.mgr.scheduler.place.assert_has_calls([call(job1), call(job1)])
            # Deferred job queue should *NOT* be empty since the job is rejected a second
            # time and still deferred when simulation completes.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)

    def test_multi_job_reject(self):
        """
        Verify that multiple jobs are submitted and rejected.
        """
        job1 = copy.deepcopy(JOB1)
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 2
        job3 = copy.deepcopy(JOB1)
        job3.arrival_time_sec = 3
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=SchedDecision.REJECT,
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2, job3]))
            self.env.run(until=DEFERRED_SCHED_SEC + 1)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job1 is rejected twice, job2 and job3 each gets rejected once.
            # Must be in the exact order.
            self.mgr.scheduler.place.assert_has_calls(
                [call(job1), call(job2), call(job3), call(job1)]
            )
            # All three jobs are still deferred when simulation completes.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 3)

    def test_mixed_jobs(self):
        """
        Verify that a mixture of jobs are handled as expected.
        """
        jobs = []
        cnt = 3
        for t in range(cnt):
            job = Job(
                uuid=t + 1,
                topology=TopoType.CLOS,
                shape=(1,),
                size=1,
                duration_sec=1,
                arrival_time_sec=t,
            )
            jobs.append(job)

        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                SchedDecision.ADMIT,
                SchedDecision.REJECT,
                SchedDecision.ADMIT,
                SchedDecision.ADMIT,
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper(jobs))
            # Run simulation until t = 1. Only job1 is processed.
            self.env.run(until=1)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job1 is admitted, deferred queue should be empty.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 0)
            self.mgr.scheduler.place.assert_has_calls([call(jobs[0])])
            # Continue simulation until t = 2. Job2 is processed.
            self.env.run(until=2)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job2 is rejected, it is put into the deferred queue.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.mgr.scheduler.place.assert_has_calls([call(jobs[i]) for i in range(2)])
            # Continue simulation until right before deferral is done. Job3 is processed.
            self.env.run(until=DEFERRED_SCHED_SEC - 1)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job3 is admitted, job2 is still in the deferred queue.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.mgr.scheduler.place.assert_has_calls([call(jobs[i]) for i in range(cnt)])
            # Continue simulation until right after deferral is done. Job2 is retried.
            self.env.run(until=DEFERRED_SCHED_SEC + 2)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job2 is taken out of the deferred queue, retried and admitted.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 0)
            self.mgr.scheduler.place.assert_has_calls(
                [call(jobs[i]) for i in range(cnt)] + [call(jobs[1])]
            )
