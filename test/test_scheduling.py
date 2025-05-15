import copy
import simpy
import unittest
import numpy as np
from unittest.mock import call, MagicMock

from common.job import Job, TopoType
from Cluster.cluster import Cluster
from Cluster.topology import Node
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
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
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

    def test_slurm_hilbert_t2d(self):
        """
        Verify the behavior when using slurm_hilbert as the policy in 2D torus.
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
        self.assertRaises(ValueError, self.sched.place, job, "slurm_hilbert")

        self.mock_cluster.topo = TopoType.T2D
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
        # Insufficient total XPUs available, expect rejection.
        decision, _ = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleXPU.return_value = job.size
        # Insufficient total nodes available, expect rejection.
        decision, _ = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleNodes.return_value = job.size
        self.mock_cluster.dimx = 4
        self.mock_cluster.dimy = 4
        self.mock_cluster.bits_per_dim = 2
        self.mock_cluster.linearAvail.return_value = np.array([])
        # All nodes unavailable, expect rejection.
        decision, _ = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.REJECT)

        # [Admitted case 1] best fit with smallest range.
        self.mock_cluster.linearAvail.return_value = np.array([1, 3, 4, 5, 6])
        # Corresponding coordinates for nodes with Hilbert index [3, 4, 5, 6].
        # Note that this is an L-shaped allocation.
        coords = [(0, 1), (0, 2), (0, 3), (1, 3)]
        # There are two feasible allocations, expect admission.
        decision, job_to_sched = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 4)
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure slurm_hilbert picks the best feasible allocation, namely [3, 4, 5, 6].
        for x, y in coords:
            self.assertIn(f"x{x}-y{y}", job_to_sched.allocation)

        # [Admitted case 2] first fit with equally small range.
        job_to_sched.allocation.clear()
        self.mock_cluster.linearAvail.return_value = np.array([1, 2, 3, 4, 5])
        # Corresponding coordinates for nodes with Hilbert index [1, 2, 3, 4].
        coords = [(1, 0), (1, 1), (0, 1), (0, 2)]
        # There are two feasible allocations, expect admission.
        decision, job_to_sched = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 4)
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure slurm_hilbert picks the best feasible allocation, namely [1, 2, 3, 4].
        for x, y in coords:
            self.assertIn(f"x{x}-y{y}", job_to_sched.allocation)

    def test_slurm_hilbert_t3d(self):
        """
        Verify the behavior when using slurm_hilbert as the policy in 3D torus.
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
        self.assertRaises(ValueError, self.sched.place, job, "slurm_hilbert")

        self.mock_cluster.topo = TopoType.T3D_NT
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
        # Insufficient total XPUs available, expect rejection.
        decision, _ = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleXPU.return_value = job.size
        # Insufficient total nodes available, expect rejection.
        decision, _ = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleNodes.return_value = job.size
        self.mock_cluster.dimx = 4
        self.mock_cluster.dimy = 4
        self.mock_cluster.dimz = 4
        self.mock_cluster.bits_per_dim = 2
        self.mock_cluster.linearAvail.return_value = np.array([])
        # All nodes unavailable, expect rejection.
        decision, _ = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.REJECT)

        # [Admitted case 1] best fit with smallest range.
        self.mock_cluster.linearAvail.return_value = np.array([1, 3, 4, 5, 6, 7, 8])
        # Corresponding coordinates for nodes with Hilbert index [3, 4, 5, 6, 7, 8].
        coords = [(1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 2)]
        # There are two feasible allocations, expect admission.
        decision, job_to_sched = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 6)
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure slurm_hilbert picks the best feasible allocation, namely [3-8].
        for x, y, z in coords:
            self.assertIn(f"x{x}-y{y}-z{z}", job_to_sched.allocation)

        # [Admitted case 2] first fit with equally small range.
        job_to_sched.allocation.clear()
        self.mock_cluster.linearAvail.return_value = np.array([1, 2, 3, 4, 5, 6, 7])
        # Corresponding coordinates for nodes with Hilbert index [1, 2, 3, 4, 5, 6].
        coords = [
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
        # There are two feasible allocations, expect admission.
        decision, job_to_sched = self.sched.place(job, policy="slurm_hilbert")
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 6)
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure slurm_hilbert picks the first feasible allocation, namely [1-6].
        for x, y, z in coords:
            self.assertIn(f"x{x}-y{y}-z{z}", job_to_sched.allocation)

    def test_reconfig_t2d(self):
        """
        Verify the behavior when using reconfig as the policy in 2D torus.
        """
        job = Job(
            uuid=1,
            topology=TopoType.T2D,
            shape=(15, 4),
            size=60,
            duration_sec=1000,
            arrival_time_sec=0,
        )

        # Job topology and cluster mismatch, expect an exception.
        self.assertRaises(ValueError, self.sched.place, job, "reconfig")

        self.mock_cluster.topo = TopoType.T2D
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
        # Insufficient total XPUs available, expect rejection.
        decision, _ = self.sched.place(job, policy="reconfig")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleXPU.return_value = 64
        # Insufficient total nodes available, expect rejection.
        decision, _ = self.sched.place(job, policy="reconfig")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleNodes.return_value = 64
        self.mock_cluster.dimx = 8
        self.mock_cluster.dimy = 8
        self.mock_cluster.blocks = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        for x in range(8):
            for y in range(8):
                node = MagicMock(spec=Node)
                node.name = f"x{x}-y{y}"
                block_x = x // 4
                block_y = y // 4
                self.mock_cluster.blocks[(block_x, block_y)].append(node)

        self.mock_cluster.toBlockArray.return_value = np.zeros((4, 4))
        decision, _ = self.sched.place(job, policy="reconfig", rsize=4)
        self.assertEqual(decision, SchedDecision.REJECT)

        # Ensure that only shape (4, 15) can be scheduled.
        def side_effect_helper(*args):
            avail = np.ones((4, 4))
            names = [node.name for node in args[0]]
            if "x7-y7" in names:
                avail[:, -1] = 0
            return avail

        self.mock_cluster.toBlockArray.side_effect = side_effect_helper
        decision, job_to_sched = self.sched.place(job, policy="reconfig", rsize=4)
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 60)
        self.assertEqual(job.shape, (4, 15))
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure the four nodes are not allocated.
        for x, y in [(4, 7), (5, 7), (6, 7), (7, 7)]:
            self.assertNotIn(f"x{x}-y{y}", job_to_sched.allocation)

        job.shape = (12, 1)
        job.size = 12

        def side_effect_helper2(*args):
            names = [node.name for node in args[0]]
            avail = np.zeros((4, 4))
            if "x0-y4" in names or "x7-y7" in names:
                avail[-1, :] = 1
            else:
                avail[0, :] = 1
            return avail

        self.mock_cluster.toBlockArray.side_effect = side_effect_helper2
        decision, job_to_sched = self.sched.place(job, policy="reconfig", rsize=4)
        # Available slices at inconsistent locations should lead to rejection.
        self.assertEqual(decision, SchedDecision.REJECT)

        # Now try to schedule a 2x2 job in a 8x8 cluster with 8x8 reconfigurable block.
        # Only the bottom right corner is available.
        job.allocation = {}
        job.shape = (2, 2)
        job.size = 4
        self.mock_cluster.blocks = {(0, 0): []}
        for x in range(8):
            for y in range(8):
                node = MagicMock(spec=Node)
                node.name = f"x{x}-y{y}"
                self.mock_cluster.blocks[(0, 0)].append(node)
        avail = np.zeros((8, 8))
        avail[-2:, -2:] = 1
        self.mock_cluster.toBlockArray.side_effect = None
        self.mock_cluster.toBlockArray.return_value = avail
        decision, job_to_sched = self.sched.place(job, policy="reconfig", rsize=8)
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 4)
        self.assertEqual(job.shape, (2, 2))
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure the four nodes at the bottom right corner are allocated.
        for x, y in [(6, 6), (6, 7), (7, 6), (7, 7)]:
            self.assertIn(f"x{x}-y{y}", job_to_sched.allocation)

    def test_reconfig_t3d(self):
        """
        Verify the behavior when using reconfig as the policy in 3D torus.
        """
        job = Job(
            uuid=1,
            topology=TopoType.T3D_NT,
            shape=(29, 4, 4),
            size=29 * 4 * 4,
            duration_sec=1000,
            arrival_time_sec=0,
        )

        # Job topology and cluster mismatch, expect an exception.
        self.assertRaises(ValueError, self.sched.place, job, "reconfig")

        self.mock_cluster.topo = TopoType.T3D_NT
        self.mock_cluster.totalIdleXPU.return_value = 1
        self.mock_cluster.totalIdleNodes.return_value = 1
        # Insufficient total XPUs available, expect rejection.
        decision, _ = self.sched.place(job, policy="reconfig")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleXPU.return_value = 8 * 8 * 8
        # Insufficient total nodes available, expect rejection.
        decision, _ = self.sched.place(job, policy="reconfig")
        self.assertEqual(decision, SchedDecision.REJECT)

        self.mock_cluster.totalIdleNodes.return_value = 8 * 8 * 8
        self.mock_cluster.dimx = 8
        self.mock_cluster.dimy = 8
        self.mock_cluster.dimz = 8
        self.mock_cluster.blocks = {
            (0, 0, 0): [],
            (0, 0, 1): [],
            (0, 1, 0): [],
            (0, 1, 1): [],
            (1, 0, 0): [],
            (1, 0, 1): [],
            (1, 1, 0): [],
            (1, 1, 1): [],
        }
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    node = MagicMock(spec=Node)
                    node.name = f"x{x}-y{y}-z{z}"
                    block_x = x // 4
                    block_y = y // 4
                    block_z = z // 4
                    self.mock_cluster.blocks[(block_x, block_y, block_z)].append(node)

        self.mock_cluster.toBlockArray.return_value = np.zeros((4, 4, 4))
        decision, _ = self.sched.place(job, policy="reconfig", rsize=4)
        self.assertEqual(decision, SchedDecision.REJECT)

        # Ensure that only shape (29, 4, 4) can be scheduled.
        def side_effect_helper(*args):
            avail = np.ones((4, 4, 4))
            names = [node.name for node in args[0]]
            # Only the x=0 plane is available.
            if "x7-y7-z7" in names:
                avail[1:, :, :] = 0
            return avail

        self.mock_cluster.toBlockArray.side_effect = side_effect_helper
        decision, job_to_sched = self.sched.place(job, policy="reconfig", rsize=4)
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 29 * 4 * 4)
        self.assertEqual(job.shape, (29, 4, 4))
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure the four nodes are not allocated.
        for x in range(5, 8):
            for y in range(4, 8):
                for z in range(4, 8):
                    self.assertNotIn(f"x{x}-y{y}-z{z}", job_to_sched.allocation)

        job.shape = (8, 1, 4)
        job.size = 32

        def side_effect_helper2(*args):
            avail = np.zeros((4, 4, 4))
            names = [node.name for node in args[0]]
            if "x0-y0-z0" in names:
                avail[:, 0, :] = 1
            elif "x4-y4-z4" in names:
                avail[:, -1, :] = 1
            return avail

        self.mock_cluster.toBlockArray.side_effect = side_effect_helper2
        decision, job_to_sched = self.sched.place(job, policy="reconfig", rsize=4)
        # XZ places at y=0 and y=3 do not qualify. Should lead to rejection.
        self.assertEqual(decision, SchedDecision.REJECT)

        # Now try to schedule a 2x2x2 job in a 8x8x8 cluster with 8x8x8 reconfigurable block.
        # Only the (7, 7, 7) corner is available.
        job.allocation = {}
        job.shape = (2, 2, 2)
        job.size = 8
        self.mock_cluster.blocks = {(0, 0, 0): []}
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    node = MagicMock(spec=Node)
                    node.name = f"x{x}-y{y}-z{z}"
                    self.mock_cluster.blocks[(0, 0, 0)].append(node)
        avail = np.zeros((8, 8, 8))
        avail[-2:, -2:, -2:] = 1
        self.mock_cluster.toBlockArray.side_effect = None
        self.mock_cluster.toBlockArray.return_value = avail
        decision, job_to_sched = self.sched.place(job, policy="reconfig", rsize=8)
        self.assertEqual(decision, SchedDecision.ADMIT)
        self.assertEqual(len(job_to_sched.allocation), 8)
        self.assertEqual(job.shape, (2, 2, 2))
        for val in job_to_sched.allocation.values():
            self.assertEqual(val, 1)
        # Make sure the four nodes at the bottom right corner are allocated.
        for x in range(6, 8):
            for y in range(6, 8):
                for z in range(6, 8):
                    self.assertIn(f"x{x}-y{y}-z{z}", job_to_sched.allocation)
        # Other nodes should not be allocated.
        for x, y, z in [(0, 0, 0), (1, 1, 1)]:
            self.assertNotIn(f"x{x}-y{y}-z{z}", job_to_sched.allocation)
