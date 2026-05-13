import unittest
from collections import Counter

from common.job import (
    Job,
    TopoType,
    compute_ring_comm_pattern,
    enumerate_xmajor,
)


class TestEnumerateXMajor(unittest.TestCase):

    def test_1d_no_origin(self):
        self.assertEqual(
            list(enumerate_xmajor((3,))),
            [(0,), (1,), (2,)],
        )

    def test_2d_no_origin_x_is_fastest(self):
        # shape (2, 3): x fastest, then y.
        # Expected order: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2).
        self.assertEqual(
            list(enumerate_xmajor((2, 3))),
            [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        )

    def test_3d_no_origin_x_then_y_then_z(self):
        # shape (2, 2, 2): full cube enumerated x fastest, then y, then z.
        self.assertEqual(
            list(enumerate_xmajor((2, 2, 2))),
            [
                (0, 0, 0), (1, 0, 0),
                (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1),
                (0, 1, 1), (1, 1, 1),
            ],
        )

    def test_2d_with_origin(self):
        # Same iteration order, offset by origin.
        self.assertEqual(
            list(enumerate_xmajor((2, 2), origin=(10, 20))),
            [(10, 20), (11, 20), (10, 21), (11, 21)],
        )

    def test_count_matches_product_of_extents(self):
        # Spot-check: total yielded coords = product of extents.
        coords = list(enumerate_xmajor((3, 4, 5)))
        self.assertEqual(len(coords), 3 * 4 * 5)
        # All unique.
        self.assertEqual(len(set(coords)), 3 * 4 * 5)


class TestComputeRingCommPattern(unittest.TestCase):

    def test_1d_size_four_is_single_forward_ring(self):
        self.assertEqual(
            compute_ring_comm_pattern((4,)),
            [(0, 1), (1, 2), (2, 3), (3, 0)],
        )

    def test_1d_size_one_is_empty(self):
        # Degenerate axis (size 1) contributes no edges; no self-loops.
        self.assertEqual(compute_ring_comm_pattern((1,)), [])

    def test_1d_size_two_is_bidirectional_ring(self):
        # Forward ring of size 2 naturally emits both directions:
        # (0 -> 1) and (1 -> 0).
        self.assertEqual(
            compute_ring_comm_pattern((2,)),
            [(0, 1), (1, 0)],
        )

    def test_2d_shape_4x2_matches_spec_example(self):
        # Worked example from the spec. Ranks are x-fastest:
        #   rank(c0, c1) = c0 + c1 * shape[0]
        # Order: all axis-0 edges first (fibers y=0, then y=1), then all
        # axis-1 edges (fibers x=0, x=1, x=2, x=3).
        self.assertEqual(
            compute_ring_comm_pattern((4, 2)),
            [
                # axis 0 (x), fiber y=0
                (0, 1), (1, 2), (2, 3), (3, 0),
                # axis 0 (x), fiber y=1
                (4, 5), (5, 6), (6, 7), (7, 4),
                # axis 1 (y), fiber x=0
                (0, 4), (4, 0),
                # axis 1 (y), fiber x=1
                (1, 5), (5, 1),
                # axis 1 (y), fiber x=2
                (2, 6), (6, 2),
                # axis 1 (y), fiber x=3
                (3, 7), (7, 3),
            ],
        )

    def test_2d_shape_3x3_axis0_first_fiber(self):
        edges = compute_ring_comm_pattern((3, 3))
        # 2 axes * 9 ranks = 18 edges total.
        self.assertEqual(len(edges), 18)
        # First three edges = axis-0 fiber y=0: (0,1),(1,2),(2,0).
        self.assertEqual(edges[:3], [(0, 1), (1, 2), (2, 0)])

    def test_2d_shape_4x1_skips_degenerate_axis(self):
        # Axis 1 has size 1; only axis-0 ring edges are emitted.
        self.assertEqual(
            compute_ring_comm_pattern((4, 1)),
            [(0, 1), (1, 2), (2, 3), (3, 0)],
        )

    def test_3d_shape_3x1x5_middle_degenerate_axis(self):
        # Axis 1 is degenerate (size 1); only axis 0 and axis 2 emit edges.
        # 2 non-degenerate axes * 15 ranks = 30 edges.
        edges = compute_ring_comm_pattern((3, 1, 5))
        self.assertEqual(len(edges), 30)
        # No self-loops introduced by the degenerate axis.
        self.assertFalse(any(src == dst for src, dst in edges))

    def test_3d_shape_2x2x2_first_axis2_edge(self):
        edges = compute_ring_comm_pattern((2, 2, 2))
        # 3 axes * 8 ranks = 24 edges total.
        self.assertEqual(len(edges), 24)
        # 8 axis-0 edges + 8 axis-1 edges precede axis-2, so edges[16] is the
        # first axis-2 edge. First axis-2 fiber is at (x=0, y=0): rank 0 ->
        # rank at (0,0,1) = 0 + 0*2 + 1*4 = 4.
        self.assertEqual(edges[16], (0, 4))

    def test_3d_shape_3x4x5_total_edge_count(self):
        edges = compute_ring_comm_pattern((3, 4, 5))
        # Every rank emits one outgoing edge per non-degenerate axis.
        # 3 axes, 60 ranks => 180 edges.
        self.assertEqual(len(edges), 3 * 4 * 5 * 3)

    def test_all_ranks_within_shape(self):
        # On a fully non-degenerate 3D shape, every src/dst falls inside
        # [0, prod(shape)); every rank appears as a source exactly ndim
        # times (one outgoing edge per axis); and no edge is duplicated.
        shape = (3, 4, 5)
        size = 3 * 4 * 5
        ndim = len(shape)
        edges = compute_ring_comm_pattern(shape)
        for src, dst in edges:
            self.assertGreaterEqual(src, 0)
            self.assertLess(src, size)
            self.assertGreaterEqual(dst, 0)
            self.assertLess(dst, size)
        # Out-degree per rank == ndim.
        src_counts = Counter(src for src, _ in edges)
        self.assertEqual(src_counts, Counter({r: ndim for r in range(size)}))
        # No duplicate edges.
        self.assertEqual(len(edges), len(set(edges)))


def _make_job(uuid=1):
    return Job(
        uuid=uuid,
        topology=TopoType.T2D,
        shape=(1,),
        size=1,
        duration_sec=10.0,
        arrival_time_sec=0.0,
    )


class TestAddToAllocation(unittest.TestCase):

    def test_first_call_assigns_rank_zero(self):
        job = _make_job()
        job.addToAllocation("x0-y0")
        self.assertEqual(job.allocation, {0: {"node": "x0-y0", "num_xpu": 1}})

    def test_ranks_count_up_densely(self):
        job = _make_job()
        for name in ("x0-y0", "x1-y0", "x0-y1"):
            job.addToAllocation(name)
        self.assertEqual(list(job.allocation.keys()), [0, 1, 2])

    def test_default_num_xpu_is_one(self):
        job = _make_job()
        job.addToAllocation("x0-y0")
        self.assertEqual(job.allocation[0]["num_xpu"], 1)

    def test_explicit_num_xpu_is_recorded(self):
        job = _make_job()
        job.addToAllocation("x0-y0", num_xpu=4)
        self.assertEqual(job.allocation[0]["num_xpu"], 4)

    def test_reset_then_append_restarts_at_rank_zero(self):
        job = _make_job()
        job.addToAllocation("x0-y0")
        job.allocation = {}
        job.addToAllocation("x1-y0")
        self.assertEqual(job.allocation, {0: {"node": "x1-y0", "num_xpu": 1}})


if __name__ == "__main__":
    unittest.main()
