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


if __name__ == "__main__":
    unittest.main()
