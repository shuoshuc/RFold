import unittest

from ClusterManager import astra_sim


class TestBuildBwMatrix(unittest.TestCase):

    def test_shape_2x2x1_bidirectional_neighbors(self):
        # Shape (2,2,1), x-fastest rank numbering:
        #   (0,0,0)->0  (1,0,0)->1  (0,1,0)->2  (1,1,0)->3
        # Axis 0 (size 2): bidirectional pairs (0,1) and (2,3).
        # Axis 1 (size 2): bidirectional pairs (0,2) and (1,3).
        # Axis 2 (size 1): degenerate, no edges.
        M = astra_sim.build_bw_matrix((2, 2, 1))
        self.assertEqual(len(M), 4)
        for row in M:
            self.assertEqual(len(row), 4)
        # Axis-0 neighbors
        self.assertEqual(M[0][1], 50.0)
        self.assertEqual(M[1][0], 50.0)
        self.assertEqual(M[2][3], 50.0)
        self.assertEqual(M[3][2], 50.0)
        # Axis-1 neighbors
        self.assertEqual(M[0][2], 50.0)
        self.assertEqual(M[2][0], 50.0)
        self.assertEqual(M[1][3], 50.0)
        self.assertEqual(M[3][1], 50.0)
        # Diagonal is zero
        for i in range(4):
            self.assertEqual(M[i][i], 0.0)
        # Non-neighbors are zero (diagonal-of-the-2x2-grid pairs)
        self.assertEqual(M[0][3], 0.0)
        self.assertEqual(M[3][0], 0.0)
        self.assertEqual(M[1][2], 0.0)
        self.assertEqual(M[2][1], 0.0)

    def test_shape_1x1x1_returns_1x1_zero(self):
        M = astra_sim.build_bw_matrix((1, 1, 1))
        self.assertEqual(M, [[0.0]])

    def test_shape_3_one_dim_ring(self):
        # Ring of 3: bidirectional pairs (0,1), (1,2), (0,2).
        M = astra_sim.build_bw_matrix((3,))
        self.assertEqual(M[0][1], 50.0)
        self.assertEqual(M[1][0], 50.0)
        self.assertEqual(M[1][2], 50.0)
        self.assertEqual(M[2][1], 50.0)
        # Wrap-around pair: rank 2 -> (2+1)%3 = 0
        self.assertEqual(M[2][0], 50.0)
        self.assertEqual(M[0][2], 50.0)
        # Diagonal zero
        self.assertEqual(M[0][0], 0.0)
        self.assertEqual(M[1][1], 0.0)
        self.assertEqual(M[2][2], 0.0)

    def test_custom_default(self):
        M = astra_sim.build_bw_matrix((2,), default=12.5)
        self.assertEqual(M[0][1], 12.5)
        self.assertEqual(M[1][0], 12.5)


if __name__ == "__main__":
    unittest.main()
