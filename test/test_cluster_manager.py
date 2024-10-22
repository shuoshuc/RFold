import copy
import simpy
import unittest
from unittest.mock import patch

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
