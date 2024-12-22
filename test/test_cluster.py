import copy
import simpy
import unittest

from common.flags import *
from common.job import Job, TopoType
from Cluster.cluster import Cluster

JOB = Job(
    uuid=1,
    topology=TopoType.CLOS,
    shape=(1,),
    size=1,
    duration_sec=1000,
    arrival_time_sec=0,
)


class TestClusterSimple(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, num_nodes=2, num_xpu=2)

    def test_node_creation(self):
        """
        Verify that expected number of nodes are created.
        """
        # Thre are 2 nodes in the cluster.
        self.assertEqual(self.cluster.numNodes(), 2)
        self.assertIn("n1", self.cluster.allNodes())
        self.assertIn("n2", self.cluster.allNodes())
        self.assertNotIn("n0", self.cluster.allNodes())
        self.assertNotIn("n3", self.cluster.allNodes())
        self.assertNotIn("", self.cluster.allNodes())
        self.assertNotIn(" ", self.cluster.allNodes())
        # Each node has 2 XPUs.
        nodes = self.cluster.allNodes().values()
        for node in nodes:
            self.assertEqual(node.numXPU(), 2)
            self.assertEqual(node.numIdleXPU(), 2)
            self.assertEqual(self.cluster.getIdleXPU(node.name), 2)

    def test_node_xpu_alloc_free(self):
        """
        Verify that XPUs are allocated and freed correctly on a node.
        """
        nodes = self.cluster.allNodes().values()
        for node in nodes:
            # Start with 2 idle XPUs.
            self.assertEqual(node.numXPU(), 2)
            self.assertEqual(node.numIdleXPU(), 2)
            # Allocate 1, now 1 idle left.
            node.alloc(1)
            self.assertEqual(node.numIdleXPU(), 1)
            # Allocate 1, now 0 idle left.
            node.alloc(1)
            self.assertEqual(node.numIdleXPU(), 0)
            # Allocate 1, this would trigger an exception.
            self.assertRaises(ValueError, node.alloc, 1)
            # Free 1, 0 idle -> 1 idle.
            node.free(1)
            self.assertEqual(node.numIdleXPU(), 1)
            # Free 1, 1 idle -> 2 idle.
            node.free(1)
            self.assertEqual(node.numIdleXPU(), 2)
            # Free 1, exceeds total XPU, triggers an exception.
            self.assertRaises(ValueError, node.free, 1)

    def test_job_execution(self):
        """
        Verify that job execution changes the cluster and node resources.
        """
        job = copy.deepcopy(JOB)
        # Job's allocation field is empty, should trigger an exception.
        self.assertRaises(ValueError, self.cluster.execute, job)
        # Node n1 and n2 each has 2 idle XPUs.
        self.assertIn("n1", self.cluster.allNodes())
        self.assertIn("n2", self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU("n1"), 2)
        self.assertEqual(self.cluster.getIdleXPU("n2"), 2)
        # Allocate 1 XPU from each node to the job.
        job.allocation = {"n1": 1, "n2": 1}
        # Job's shape is (1), mismatching the allocation shape.
        self.assertRaises(ValueError, self.cluster.execute, job)
        # Fix and execute the job with correct shape.
        job.shape = (1, 1)
        self.cluster.execute(job)
        # Each node should only have 1 idle XPU left.
        self.assertEqual(self.cluster.getIdleXPU("n1"), 1)
        self.assertEqual(self.cluster.getIdleXPU("n2"), 1)

    def test_job_completion(self):
        """
        Verify that job completion changes the cluster and node resources.
        """
        job = copy.deepcopy(JOB)
        job.shape = (1, 1)
        job.allocation = {"n1": 1, "n2": 1}
        self.cluster.execute(job)
        # Node n1 and n2 each has 1 idle XPU.
        self.assertIn("n1", self.cluster.allNodes())
        self.assertIn("n2", self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU("n1"), 1)
        self.assertEqual(self.cluster.getIdleXPU("n2"), 1)
        # Now job is completed, free up the resources.
        self.cluster.complete(job)
        # Each node should have 2 idle XPUs.
        self.assertEqual(self.cluster.getIdleXPU("n1"), 2)
        self.assertEqual(self.cluster.getIdleXPU("n2"), 2)

    def test_job_completion_exception(self):
        """
        Verify that a job with malicious allocation info is caught in the completion call.
        """
        job = copy.deepcopy(JOB)
        job.shape = (1, 1)
        job.allocation = {"n1": 1, "n2": 1}
        self.cluster.execute(job)
        # Node n1 and n2 each has 1 idle XPU.
        self.assertIn("n1", self.cluster.allNodes())
        self.assertIn("n2", self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU("n1"), 1)
        self.assertEqual(self.cluster.getIdleXPU("n2"), 1)
        # Malicious allocation info tries to over-free resources.
        job.allocation = {"n1": 3, "n2": 3}
        # This triggers an exception from the underlying nodes.
        self.assertRaises(ValueError, self.cluster.complete, job)
