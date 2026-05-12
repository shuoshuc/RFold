import unittest

from common.job import enumerate_xmajor


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


from common.job import Job, TopoType


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
