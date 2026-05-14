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


def _make_t2d_job(uuid: int, shape, node_ranks: list[str], duration_sec: float = 10.0):
    """Build a T2D Job and populate its allocation in the given rank order."""
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
        job.allocation = {
            0: {"node": C1_NODE1, "num_xpu": 1},
            1: {"node": C1_NODE2, "num_xpu": 1},
        }
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
        job.allocation = {
            0: {"node": C1_NODE1, "num_xpu": 1},
            1: {"node": C1_NODE2, "num_xpu": 1},
        }
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
        job.allocation = {
            0: {"node": C1_NODE1, "num_xpu": 1},
            1: {"node": C1_NODE2, "num_xpu": 1},
        }
        self.cluster.execute(job)
        # Node 1 and 2 each has 0 idle XPU, node 3 has 1 idle XPU.
        self.assertIn(C1_NODE1, self.cluster.allNodes())
        self.assertIn(C1_NODE2, self.cluster.allNodes())
        self.assertIn(C1_NODE3, self.cluster.allNodes())
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE1), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE2), 0)
        self.assertEqual(self.cluster.getIdleXPU(C1_NODE3), 1)
        # Malicious allocation info tries to over-free resources.
        job.allocation = {
            0: {"node": C1_NODE1, "num_xpu": 3},
            1: {"node": C1_NODE2, "num_xpu": 3},
        }
        # This triggers an exception from the underlying nodes.
        self.assertRaises(ValueError, self.cluster.complete, job)

    def test_coord_to_node_lookup_is_built_for_torus(self):
        """T2D cluster builds _coord_to_node mapping every coord -> Node."""
        # 4x4 grid -> 16 entries, each (dimx, dimy) maps to the node at that coord.
        self.assertEqual(len(self.cluster._coord_to_node), 16)
        node_x0_y0 = self.cluster._coord_to_node[(0, 0)]
        self.assertEqual(node_x0_y0.name, "x0-y0")
        self.assertEqual(node_x0_y0.dimx, 0)
        self.assertEqual(node_x0_y0.dimy, 0)
        node_x3_y3 = self.cluster._coord_to_node[(3, 3)]
        self.assertEqual(node_x3_y3.name, "x3-y3")

    def test_getLinkFlows_returns_zero_for_all_links_initially(self):
        """getLinkFlows snapshots every link's flow_count; freshly built cluster is all zero."""
        flows = self.cluster.getLinkFlows()
        self.assertEqual(len(flows), len(self.cluster.links))
        self.assertTrue(all(v == 0 for v in flows.values()))
        # Spot-check one known link name from c1.json.
        self.assertIn("x0-y0-p1:x1-y0-p0", flows)
        self.assertEqual(flows["x0-y0-p1:x1-y0-p0"], 0)

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
        self.assertEqual(failed_nodes, self.cluster.getFailedNodes())


class TestClusterRouting(unittest.TestCase):
    """Unit tests for Cluster._routePath on the c1.json 4x4 T2D topology."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))

    def _node(self, name: str):
        return self.cluster.getNodeByName(name)

    def _link_names(self, links):
        return [link.name for link in links]

    def test_same_node_returns_empty_path(self):
        """Routing from a node to itself returns no links."""
        n = self._node("x0-y0")
        self.assertEqual(self.cluster._routePath(n, n), [])

    def test_adjacent_plus_x_one_hop(self):
        """Adjacent +x neighbor: x0-y0 -> x1-y0 traverses 1 link."""
        path = self.cluster._routePath(self._node("x0-y0"), self._node("x1-y0"))
        self.assertEqual(self._link_names(path), ["x0-y0-p1:x1-y0-p0"])

    def test_two_hop_plus_x_in_order(self):
        """Two-hop +x: x0-y0 -> x2-y0 traverses x0->x1 then x1->x2 in order."""
        path = self.cluster._routePath(self._node("x0-y0"), self._node("x2-y0"))
        self.assertEqual(
            self._link_names(path),
            ["x0-y0-p1:x1-y0-p0", "x1-y0-p1:x2-y0-p0"],
        )

    def test_single_hop_plus_x_wraparound(self):
        """+x wrap from x3-y0 -> x0-y0 is exactly one link (the wrap)."""
        path = self.cluster._routePath(self._node("x3-y0"), self._node("x0-y0"))
        self.assertEqual(self._link_names(path), ["x3-y0-p1:x0-y0-p0"])

    def test_three_hop_plus_x_never_takes_backward_wrap(self):
        """x0-y0 -> x3-y0 walks +x three times, never the 1-hop -x wrap."""
        path = self.cluster._routePath(self._node("x0-y0"), self._node("x3-y0"))
        self.assertEqual(
            self._link_names(path),
            [
                "x0-y0-p1:x1-y0-p0",
                "x1-y0-p1:x2-y0-p0",
                "x2-y0-p1:x3-y0-p0",
            ],
        )

    def test_diagonal_routes_x_first_then_y(self):
        """x0-y0 -> x2-y3 takes 2 +x hops followed by 3 +y hops, in that order."""
        path = self.cluster._routePath(self._node("x0-y0"), self._node("x2-y3"))
        self.assertEqual(
            self._link_names(path),
            [
                "x0-y0-p1:x1-y0-p0",
                "x1-y0-p1:x2-y0-p0",
                "x2-y0-p3:x2-y1-p2",
                "x2-y1-p3:x2-y2-p2",
                "x2-y2-p3:x2-y3-p2",
            ],
        )

    def test_pure_plus_y_wraparound(self):
        """+y wrap from x0-y3 -> x0-y0 is one link (the wrap)."""
        path = self.cluster._routePath(self._node("x0-y3"), self._node("x0-y0"))
        self.assertEqual(self._link_names(path), ["x0-y3-p3:x0-y0-p2"])

    def test_missing_link_raises_value_error(self):
        """_routePath raises ValueError naming the offending nodes if the
        expected next-hop link is absent (simulates e.g. T3D_T twist or a
        topology gap)."""
        src = self._node("x0-y0")
        dst = self._node("x1-y0")
        # Synthetically remove the expected link to trigger the raise path.
        del self.cluster.links["x0-y0-p1:x1-y0-p0"]
        with self.assertRaises(ValueError) as cm:
            self.cluster._routePath(src, dst)
        self.assertIn("x0-y0", str(cm.exception))
        self.assertIn("x1-y0", str(cm.exception))

    def test_two_hop_plus_x_from_non_origin(self):
        """Mid-lattice same-axis routing: x1-y2 -> x3-y2 traverses 2 +x links
        from a non-zero starting coordinate (guards against off-by-one in the
        modular hop-count formula that would only manifest when start != 0)."""
        path = self.cluster._routePath(self._node("x1-y2"), self._node("x3-y2"))
        self.assertEqual(
            self._link_names(path),
            ["x1-y2-p1:x2-y2-p0", "x2-y2-p1:x3-y2-p0"],
        )


class TestClusterLinkFlows(unittest.TestCase):
    """End-to-end flow accounting via Cluster.execute / Cluster.complete."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))

    def _flow(self, link_name: str) -> int:
        return self.cluster.links[link_name].flow_count

    def test_clos_job_is_noop_for_link_flows(self):
        """A non-torus job goes through execute/complete without changing any flow_count."""
        # Build a CLOS-topology Job; topology check in _updateJobLinkFlows short-circuits
        # before getCommPattern is called. We still need a valid allocation for execute()
        # to run alloc/free on the c1.json T2D nodes.
        job = Job(
            uuid=99,
            topology=TopoType.CLOS,
            shape=(1,),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        job.addToAllocation("x0-y0")
        self.cluster.execute(job)
        self.assertTrue(all(v == 0 for v in self.cluster.getLinkFlows().values()))
        self.cluster.complete(job)
        self.assertTrue(all(v == 0 for v in self.cluster.getLinkFlows().values()))

    def test_single_2x1_job_increments_forward_and_wrap_path(self):
        """Shape (2,1) at [x0-y0, x1-y0]: forward edge = 1 link, wrap edge = 3 links."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        # Forward edge (rank 0 -> 1): x0-y0 -> x1-y0.
        self.assertEqual(self._flow("x0-y0-p1:x1-y0-p0"), 1)
        # Wrap edge (rank 1 -> 0): x1 -> x2 -> x3 -> x0.
        self.assertEqual(self._flow("x1-y0-p1:x2-y0-p0"), 1)
        self.assertEqual(self._flow("x2-y0-p1:x3-y0-p0"), 1)
        self.assertEqual(self._flow("x3-y0-p1:x0-y0-p0"), 1)
        # Total non-zero links: 4. Everything else stays at 0.
        nonzero = {k: v for k, v in self.cluster.getLinkFlows().items() if v > 0}
        self.assertEqual(len(nonzero), 4)

    def test_execute_then_complete_returns_all_flows_to_zero(self):
        """Symmetric: execute then complete leaves every link at flow_count == 0."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        self.cluster.complete(job)
        self.assertTrue(all(v == 0 for v in self.cluster.getLinkFlows().values()))

    def test_two_nonoverlapping_jobs_share_links_via_wraparound(self):
        """Job A on [x0,x1] and Job B on [x2,x3] (y=0): wraparound paths cross all 4 +x links on row y=0."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_a)
        self.cluster.execute(job_b)
        shared = [
            "x0-y0-p1:x1-y0-p0",
            "x1-y0-p1:x2-y0-p0",
            "x2-y0-p1:x3-y0-p0",
            "x3-y0-p1:x0-y0-p0",
        ]
        for name in shared:
            self.assertEqual(self._flow(name), 2, f"{name} should be 2 after both executes")
        # After completing Job A only: every shared link drops to 1.
        self.cluster.complete(job_a)
        for name in shared:
            self.assertEqual(self._flow(name), 1, f"{name} should be 1 after A complete")
        # After completing Job B: all return to 0.
        self.cluster.complete(job_b)
        self.assertTrue(all(v == 0 for v in self.cluster.getLinkFlows().values()))

    def test_2x2_job_exercises_both_axes(self):
        """Shape (2,2) at the corner: spot-check one +x and one +y link each see 1 flow."""
        job = _make_t2d_job(
            uuid=1,
            shape=(2, 2),
            node_ranks=["x0-y0", "x1-y0", "x0-y1", "x1-y1"],
        )
        self.cluster.execute(job)
        # The +x edge (rank 0 -> 1) routes x0-y0 -> x1-y0 directly; no other comm edge
        # in this allocation routes through x0-y0->x1-y0.
        self.assertEqual(self._flow("x0-y0-p1:x1-y0-p0"), 1)
        # The +y edge (rank 0 -> 2) routes x0-y0 -> x0-y1 directly; no other comm edge
        # routes through x0-y0->x0-y1.
        self.assertEqual(self._flow("x0-y0-p3:x0-y1-p2"), 1)

    def test_complete_without_execute_raises_underflow(self):
        """complete on a never-executed job triggers decFlow on a 0-count link -> ValueError."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        with self.assertRaises(ValueError) as cm:
            self.cluster.complete(job)
        # Error message names the offending link (the first one walked).
        self.assertIn("flow_count is already 0", str(cm.exception))

    def test_getLinkFlows_matches_direct_field_reads(self):
        """The accessor snapshots flow_count consistently with reading link.flow_count."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        snapshot = self.cluster.getLinkFlows()
        for name, link in self.cluster.links.items():
            self.assertEqual(snapshot[name], link.flow_count)


class TestRouteJobPaths(unittest.TestCase):
    """Unit tests for Cluster.routeJobPaths."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))

    def test_non_torus_job_yields_nothing(self):
        """A CLOS-topology job has no comm pattern, so routeJobPaths yields no edges."""
        job = Job(
            uuid=99,
            topology=TopoType.CLOS,
            shape=(1,),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        job.addToAllocation("x0-y0")
        edges = list(self.cluster.routeJobPaths(job))
        self.assertEqual(edges, [])

    def test_torus_job_yields_one_entry_per_comm_edge(self):
        """A (2, 1) T2D job has 2 comm-pattern edges; routeJobPaths yields 2 tuples
        whose third element is the list of links traversed."""
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        edges = list(self.cluster.routeJobPaths(job))
        # Comm pattern for (2,1) is [(0, 1, 1.0), (1, 0, 1.0)] — two edges.
        self.assertEqual(len(edges), 2)
        # First edge: rank 0 -> 1, 1 link forward.
        src_rank, dst_rank, links = edges[0]
        self.assertEqual((src_rank, dst_rank), (0, 1))
        self.assertEqual([l.name for l in links], ["x0-y0-p1:x1-y0-p0"])
        # Second edge: rank 1 -> 0, 3 links forward (wrap).
        src_rank, dst_rank, links = edges[1]
        self.assertEqual((src_rank, dst_rank), (1, 0))
        self.assertEqual(
            [l.name for l in links],
            [
                "x1-y0-p1:x2-y0-p0",
                "x2-y0-p1:x3-y0-p0",
                "x3-y0-p1:x0-y0-p0",
            ],
        )

    def test_missing_rank_in_allocation_raises_value_error(self):
        """If a rank referenced by the comm pattern is missing from
        job.allocation, routeJobPaths raises ValueError naming the rank."""
        # Build a torus job whose shape generates rank 1 in its comm pattern
        # but leave its allocation empty (no addToAllocation calls).
        job = Job(
            uuid=42,
            topology=TopoType.T2D,
            shape=(2, 1),
            size=2,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        # Allocate only rank 0, leaving rank 1 missing.
        job.addToAllocation("x0-y0")
        with self.assertRaises(ValueError) as cm:
            list(self.cluster.routeJobPaths(job))
        # The error message should include the job uuid and the missing rank.
        msg = str(cm.exception)
        self.assertIn("42", msg)
        self.assertIn("1", msg)


if __name__ == "__main__":
    unittest.main()
