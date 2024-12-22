import copy
import simpy
import unittest
from unittest.mock import call, MagicMock

from common.flags import *
from common.job import Job, TopoType
from Cluster.cluster import Cluster
from ClusterManager.scheduling import SchedDecision, SchedulingPolicy

JOB = Job(
    uuid=1,
    topology=TopoType.CLOS,
    shape=(1,),
    size=1,
    duration_sec=1000,
    arrival_time_sec=0,
)


class TestScheduling(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mock_cluster = MagicMock(spec=Cluster)
        self.sched = SchedulingPolicy(self.env, cluster=self.mock_cluster)

    def test_default_decision(self):
        """
        Verify that the default decision for a job is rejection.
        """
        job = copy.deepcopy(JOB)
        # Scheduler invoked with unknown policy.
        decision, job_to_sched = self.sched.place(job, policy="unknown")
        self.assertEqual(decision, SchedDecision.REJECT)
        self.assertEqual(job_to_sched, job)
        self.mock_cluster.getIdleXPU.assert_not_called()

    def test_simple_fit(self):
        """
        Verify the behavior when using simplefit as the policy.
        """
        job = copy.deepcopy(JOB)
        # Job should have an expected allocation like this.
        admitted_job = copy.deepcopy(JOB)
        admitted_job.allocation = {"n1": 1}

        # Simulate one available XPU.
        self.mock_cluster.getIdleXPU.return_value = 1
        # Job should be admitted.
        decision, job_to_sched = self.sched.place(job, policy="simplefit")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(job_to_sched, admitted_job)
        self.mock_cluster.getIdleXPU.assert_has_calls([call("n1")])

        # Job requests 2 XPUs.
        job.shape = (2,)
        job.size = 2
        # Make sure to clear the allocation info. It is set by the previous call.
        job.allocation = {}
        # Job should be rejected.
        decision, job_to_sched = self.sched.place(job, policy="simplefit")
        self.assertEqual(decision, SchedDecision.REJECT)
        # Returned job should be the same as the input job, with no allocation info.
        self.assertEqual(job_to_sched, job)
        self.assertEqual(job_to_sched.allocation, {})
        self.mock_cluster.getIdleXPU.assert_has_calls([call("n1")])
