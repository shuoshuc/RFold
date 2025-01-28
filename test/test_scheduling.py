import copy
import simpy
import unittest
import numpy as np
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
        self.mock_cluster.topo = TopoType.CLOS
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

    def test_first_fit_t2d(self):
        """
        Verify the behavior when using firstfit as the policy in 2D torus.
        """
        job = Job(
            uuid=1,
            topology=TopoType.T2D,
            shape=(2, 2),
            size=4,
            duration_sec=1000,
            arrival_time_sec=0,
        )

        # Job topology and cluster mismatch, expect an exception.
        self.assertRaises(ValueError, self.sched.place, job, "firstfit")

        self.mock_cluster.topo = TopoType.T2D
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
        # Insufficient total XPUs available, expect rejection.
        decision, _ = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleXPU.return_value = job.size
        # Insufficient total nodes available, expect rejection.
        decision, _ = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleNodes.return_value = job.size
        self.mock_cluster.dimx = 4
        self.mock_cluster.dimy = 4
        self.mock_cluster.toArray.return_value = np.zeros((4, 4))
        # All nodes unavailable, expect rejection. Note this is for testing purposes.
        # The array should match total available nodes/XPUs.
        decision, _ = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.REJECT)

        array = np.zeros((4, 4))
        # Feasible allocation wraps around the torus.
        coords = [(0, 1), (0, 2), (3, 1), (3, 2)]
        for x, y in coords:
            array[x, y] = 1
        array[0, 3] = 1
        array[3, 3] = 1
        self.mock_cluster.toArray.return_value = array
        # There are two feasible allocations, expect admission.
        decision, job_to_sched = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 4)
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure firstfit pick the first feasible allocation.
        for x, y in coords:
            self.assertIn(f"x{x}-y{y}", job_to_sched.allocation)

    def test_first_fit_t3d(self):
        """
        Verify the behavior when using firstfit as the policy in 3D torus.
        """
        job = Job(
            uuid=1,
            topology=TopoType.T3D_NT,
            shape=(2, 3, 1),
            size=6,
            duration_sec=1000,
            arrival_time_sec=0,
        )

        # Job topology and cluster mismatch, expect an exception.
        self.assertRaises(ValueError, self.sched.place, job, "firstfit")

        self.mock_cluster.topo = TopoType.T3D_NT
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
        # Insufficient total XPUs available, expect rejection.
        decision, _ = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleXPU.return_value = job.size
        # Insufficient total nodes available, expect rejection.
        decision, _ = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleNodes.return_value = job.size
        self.mock_cluster.dimx = 4
        self.mock_cluster.dimy = 4
        self.mock_cluster.dimz = 4
        self.mock_cluster.toArray.return_value = np.zeros((4, 4, 4))
        # All nodes unavailable, expect rejection. Note this is for testing purposes.
        # The array should match total available nodes/XPUs.
        decision, _ = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.REJECT)

        array = np.zeros((4, 4, 4))
        # Feasible allocation shape is 1x2x3, needs to rotate,
        # and wraps around the torus along dimension z.
        for x in range(1):
            for y in range(3):
                for z in [0, 1, 3]:
                    array[x, y, z] = 1
        self.mock_cluster.toArray.return_value = array
        # There are two feasible allocations, expect admission.
        decision, job_to_sched = self.sched.place(job, policy="firstfit")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 6)
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure firstfit pick the first feasible allocation.
        for x in range(1):
            for y in range(2):
                for z in [0, 1, 3]:
                    self.assertIn(f"x{x}-y{y}-z{z}", job_to_sched.allocation)
