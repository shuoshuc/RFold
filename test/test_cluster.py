import copy
import simpy
import unittest
import numpy as np
from hilbert import decode as hdecode

from common.job import Job, TopoType
from common.utils import spec_parser, failure_sampling
from Cluster.cluster import Cluster

JOB = Job(
    uuid=1,
    topology=TopoType.T2D,
    shape=(1,),
    size=1,
    duration_sec=1000,
    arrival_time_sec=0,
)

# Path to cluster C1's spec file. C1 is 2D torus.
C1_SPEC = "Cluster/models/c1.json"

# Some nodes in C1.
C1_NODE1 = "x0-y0"
C1_NODE2 = "x3-y3"
C1_NODE3 = "x2-y1"


class TestClusterSimple(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))

    def test_node_creation(self):
        """
        Verify that expected number of nodes are created.
        """
        # Thre are 16 nodes in the cluster.
        self.assertEqual(self.cluster.numNodes(), 16)
        self.assertIn(C1_NODE1, self.cluster.allNodes())
        self.assertIn(C1_NODE2, self.cluster.allNodes())
        self.assertIn(C1_NODE3, self.cluster.allNodes())
        self.assertNotIn("x0-y4", self.cluster.allNodes())
        self.assertNotIn("x4-y0", self.cluster.allNodes())
        self.assertNotIn("", self.cluster.allNodes())
        self.assertNotIn(" ", self.cluster.allNodes())
        # Each node has 1 XPU.
        nodes = self.cluster.allNodes().values()
        for node in nodes:
            self.assertEqual(node.numXPU(), 1)
            self.assertEqual(node.numIdleXPU(), 1)
            self.assertIsNotNone(node.getHilbertIndex())
            self.assertEqual(self.cluster.getIdleXPU(node.name), 1)

    def test_node_hilbert_index(self):
        """
        Verify that nodes have correct Hilbert indices.
        """
        for node in self.cluster.allNodes().values():
            self.assertIsNone(node.dimz)
            # Decode the Hilbert index and check against the node's coordinates.
            x, y = hdecode(node.getHilbertIndex(), 2, 2)[0]
            self.assertEqual(node.dimx, x)
            self.assertEqual(node.dimy, y)

    def test_node_xpu_alloc_free(self):
        """
        Verify that XPUs are allocated and freed correctly on a node.
        """
        nodes = self.cluster.allNodes().values()
        for node in nodes:
            # Start with 1 idle XPU.
            self.assertEqual(node.numXPU(), 1)
            self.assertEqual(node.numIdleXPU(), 1)
            # Allocate 1, now 0 idle left.
            node.alloc(1)
            self.assertEqual(node.numIdleXPU(), 0)
            # Allocate 1, this would trigger an exception.
            self.assertRaises(ValueError, node.alloc, 1)
            # Free 1, 0 idle -> 1 idle.
            node.free(1)
            self.assertEqual(node.numIdleXPU(), 1)
            # Free 1, exceeds total XPU, triggers an exception.
            self.assertRaises(ValueError, node.free, 1)

    def test_job_execution(self):
        """
        Verify that job execution changes the cluster and node resources.
        """
        job = copy.deepcopy(JOB)
        # Job's allocation field is empty, should trigger an exception.
        self.assertRaises(ValueError, self.cluster.execute, job)
        # Node 1, 2, 3 each has 1 idle XPU.
        self.assertIn(C1_NODE1, self.cluster.allNodes())
        self.assertIn(C1_NODE2, self.cluster.allNodes())
        self.assertIn(C1_NODE3, self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE1), 1)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE2), 1)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE3), 1)
        # The cluster right now is completely idle.
        self.assertTrue((self.cluster.toArray() == np.full((4, 4), 1)).all())
        # Allocate 1 XPU from node 1 and 2 to the job.
        job.allocation = {C1_NODE1: 1, C1_NODE2: 1}
        # Fix and execute the job with correct shape.
        job.shape = (1, 1)
        job.size = 2
        self.cluster.execute(job)
        # Node 1 and 2 should have 0 idle XPU left, node 3 still has 1.
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE1), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE2), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE3), 1)
        truth = np.full((4, 4), 1)
        truth[0, 0] = 0
        truth[3, 3] = 0
        self.assertTrue((self.cluster.toArray() == truth).all())

    def test_job_completion(self):
        """
        Verify that job completion changes the cluster and node resources.
        """
        job = copy.deepcopy(JOB)
        job.shape = (1, 1)
        job.size = 2
        job.allocation = {C1_NODE1: 1, C1_NODE2: 1}
        self.cluster.execute(job)
        # Node 1 and 2 each has 0 idle XPU, node 3 has 1 idle XPU.
        self.assertIn(C1_NODE1, self.cluster.allNodes())
        self.assertIn(C1_NODE2, self.cluster.allNodes())
        self.assertIn(C1_NODE3, self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE1), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE2), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE3), 1)
        truth = np.full((4, 4), 1)
        truth[0, 0] = 0
        truth[3, 3] = 0
        self.assertTrue((self.cluster.toArray() == truth).all())
        # Now job is completed, free up the resources.
        self.cluster.complete(job)
        # Each node should have 1 idle XPU.
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE1), 1)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE2), 1)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE3), 1)
        self.assertTrue((self.cluster.toArray() == np.full((4, 4), 1)).all())

    def test_job_completion_exception(self):
        """
        Verify that a job with malicious allocation info is caught in the completion call.
        """
        job = copy.deepcopy(JOB)
        job.shape = (1, 1)
        job.size = 2
        job.allocation = {C1_NODE1: 1, C1_NODE2: 1}
        self.cluster.execute(job)
        # Node 1 and 2 each has 0 idle XPU, node 3 has 1 idle XPU.
        self.assertIn(C1_NODE1, self.cluster.allNodes())
        self.assertIn(C1_NODE2, self.cluster.allNodes())
        self.assertIn(C1_NODE3, self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE1), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE2), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE3), 1)
        # Malicious allocation info tries to over-free resources.
        job.allocation = {C1_NODE1: 3, C1_NODE2: 3}
        # This triggers an exception from the underlying nodes.
        self.assertRaises(ValueError, self.cluster.complete, job)

    def test_node_failure(self):
        """
        Verify that failed nodes are marked as unavailable.
        """
        failed_nodes = failure_sampling(self.cluster, 4)
        self.assertEqual(len(failed_nodes), 4)
        # Make sure the failed nodes are in the cluster.
        for node_name in failed_nodes:
            self.assertIn(node_name, self.cluster.allNodes())
            self.assertEqual(self.cluster.getIdleXPU(node_name), 1)
        self.cluster.failNodes(failed_nodes)
        # After marking nodes as failed, they should not be available.
        for node_name in failed_nodes:
            self.assertEqual(self.cluster.getIdleXPU(node_name), 0)
