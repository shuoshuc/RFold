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


class TestBuildLtMatrix(unittest.TestCase):

    def test_default_500_on_neighbors(self):
        M = astra_sim.build_lt_matrix((2, 2, 1))
        self.assertEqual(M[0][1], 500.0)
        self.assertEqual(M[1][0], 500.0)
        self.assertEqual(M[0][2], 500.0)
        self.assertEqual(M[2][0], 500.0)
        # Non-neighbor
        self.assertEqual(M[0][3], 0.0)
        # Diagonal
        self.assertEqual(M[0][0], 0.0)

    def test_shape_2x2x2_bidirectional_axis2(self):
        # In (2,2,2), axis-2 pairs are bidirectional links between
        # rank k and rank k+4 for k in 0..3.
        M = astra_sim.build_lt_matrix((2, 2, 2))
        for k in range(4):
            self.assertEqual(M[k][k + 4], 500.0)
            self.assertEqual(M[k + 4][k], 500.0)

    def test_custom_default(self):
        M = astra_sim.build_lt_matrix((2,), default=7.0)
        self.assertEqual(M[0][1], 7.0)
        self.assertEqual(M[1][0], 7.0)


class TestWriteSchedule(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip_bw(self):
        from pathlib import Path
        M = [[0.0, 50.0], [50.0, 0.0]]
        path = Path(self.tmpdir) / "bw_schedule.txt"
        astra_sim.write_schedule(path, M, "BW")
        with path.open() as f:
            lines = f.read().splitlines()
        # 1 tag + 2 rows + 1 END = 4 lines.
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "BW")
        self.assertEqual(lines[-1], "END")
        # Body rows have N space-separated numeric fields.
        for row_idx in (1, 2):
            tokens = lines[row_idx].split()
            self.assertEqual(len(tokens), 2)
            for t in tokens:
                float(t)  # parses, otherwise raises ValueError

    def test_round_trip_lt(self):
        from pathlib import Path
        M = [[0.0, 500.0], [500.0, 0.0]]
        path = Path(self.tmpdir) / "latency_schedule.txt"
        astra_sim.write_schedule(path, M, "LT")
        with path.open() as f:
            head = f.readline().strip()
        self.assertEqual(head, "LT")

    def test_single_cell_matrix(self):
        from pathlib import Path
        path = Path(self.tmpdir) / "bw.txt"
        astra_sim.write_schedule(path, [[0.0]], "BW")
        with path.open() as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, ["BW", "0.0", "END"])


if __name__ == "__main__":
    unittest.main()
