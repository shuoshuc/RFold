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
        # Tag line includes topo_id 0: astra-sim's parser does
        # std::stoi(line.substr(3)) on the tag line and aborts if there
        # is no ID after the tag.
        self.assertEqual(lines[0], "BW 0")
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
        self.assertEqual(head, "LT 0")

    def test_single_cell_matrix(self):
        from pathlib import Path
        path = Path(self.tmpdir) / "bw.txt"
        astra_sim.write_schedule(path, [[0.0]], "BW")
        with path.open() as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, ["BW 0", "0.0", "END"])


class TestParseJctNsec(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name: str, body: str):
        from pathlib import Path
        p = Path(self.tmpdir) / name
        p.write_text(body)
        return p

    # Real jct.csv format from astra-sim: header line then comma-separated
    # rows of "<job_id>,<jct_ns>". One row per job.

    def test_single_job_csv(self):
        p = self._write(
            "jct.csv",
            "Job,JCT (nsec)\nJ0,1500000000\n",
        )
        self.assertEqual(astra_sim.parse_jct_nsec(p), 1500000000.0)

    def test_scientific_notation(self):
        p = self._write("jct.csv", "Job,JCT (nsec)\nJ0,1.5e9\n")
        self.assertEqual(astra_sim.parse_jct_nsec(p), 1.5e9)

    def test_takes_first_data_row_when_multiple_jobs(self):
        p = self._write(
            "jct.csv",
            "Job,JCT (nsec)\nJ0,2000000000\nJ1,3000000000\n",
        )
        self.assertEqual(astra_sim.parse_jct_nsec(p), 2000000000.0)

    def test_missing_file_raises(self):
        from pathlib import Path
        with self.assertRaises(RuntimeError):
            astra_sim.parse_jct_nsec(Path(self.tmpdir) / "nope.csv")

    def test_empty_file_raises(self):
        p = self._write("jct.csv", "")
        with self.assertRaises(RuntimeError):
            astra_sim.parse_jct_nsec(p)

    def test_header_only_raises(self):
        p = self._write("jct.csv", "Job,JCT (nsec)\n")
        with self.assertRaises(RuntimeError):
            astra_sim.parse_jct_nsec(p)

    def test_non_numeric_jct_raises(self):
        p = self._write("jct.csv", "Job,JCT (nsec)\nJ0,abc\n")
        with self.assertRaises(RuntimeError):
            astra_sim.parse_jct_nsec(p)


class TestRunAstra(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_astra_writes_inputs_invokes_subprocess_and_parses_jct(self):
        from pathlib import Path
        from unittest.mock import patch

        tmp_root = Path(self.tmpdir)

        # Simulate astra-sim by writing jct.csv into the outputs dir.
        # **kwargs absorbs stdout/stderr file handles passed by run_astra.
        def fake_run(cmd, **kwargs):
            self.assertEqual(cmd[0], "bash")
            self.assertEqual(cmd[1], "./run_astra.sh")
            self.assertEqual(cmd[2], "2x2x1")
            self.assertIn("--input-dir", cmd)
            self.assertIn("--output-dir", cmd)
            self.assertTrue(kwargs.get("check"))
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (out_dir / "jct.csv").write_text("Job,JCT (nsec)\nJ0,2500000000\n")
            class _R:
                returncode = 0
            return _R()

        with patch.object(astra_sim.subprocess, "run", side_effect=fake_run):
            result_nsec = astra_sim.run_astra(
                uuid=42,
                shape=(2, 2, 1),
                bw_matrix=astra_sim.build_bw_matrix((2, 2, 1)),
                lt_matrix=astra_sim.build_lt_matrix((2, 2, 1)),
                tmp_root=tmp_root,
            )

        # jct.csv emits ns; run_astra passes it through unchanged.
        self.assertEqual(result_nsec, 2500000000.0)
        # Inputs were written.
        uuid_dir = tmp_root / "42"
        self.assertTrue((uuid_dir / "inputs" / "bw_schedule.txt").exists())
        self.assertTrue((uuid_dir / "inputs" / "latency_schedule.txt").exists())
        # BW file is well-formed.
        bw_text = (uuid_dir / "inputs" / "bw_schedule.txt").read_text().splitlines()
        self.assertEqual(bw_text[0], "BW 0")
        self.assertEqual(bw_text[-1], "END")
        # N rows in the body (N = 2*2*1 = 4).
        self.assertEqual(len(bw_text), 6)  # tag + 4 rows + END

    def test_run_astra_propagates_subprocess_error(self):
        from pathlib import Path
        from unittest.mock import patch
        import subprocess as sp

        def fake_run(cmd, **kwargs):
            raise sp.CalledProcessError(returncode=1, cmd=cmd)

        with patch.object(astra_sim.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(sp.CalledProcessError):
                astra_sim.run_astra(
                    uuid=7,
                    shape=(2,),
                    bw_matrix=astra_sim.build_bw_matrix((2,)),
                    lt_matrix=astra_sim.build_lt_matrix((2,)),
                    tmp_root=Path(self.tmpdir),
                )


    def test_run_astra_uses_caller_provided_matrices(self):
        """run_astra writes the caller's BW/LT cells, not torus-neighbor defaults."""
        from pathlib import Path
        from unittest.mock import patch

        tmp_root = Path(self.tmpdir)
        # 2-NPU matrices distinguishable from any default the function might fabricate.
        bw = [[0.0, 7.0], [7.0, 0.0]]
        lt = [[0.0, 13.0], [13.0, 0.0]]

        def fake_run(cmd, **kwargs):
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (out_dir / "jct.csv").write_text("Job,JCT (nsec)\nJ0,1000\n")
            class _R:
                returncode = 0
            return _R()

        with patch.object(astra_sim.subprocess, "run", side_effect=fake_run):
            result_nsec = astra_sim.run_astra(
                uuid=11,
                shape=(2,),
                bw_matrix=bw,
                lt_matrix=lt,
                tmp_root=tmp_root,
            )

        self.assertEqual(result_nsec, 1000.0)
        bw_lines = (tmp_root / "11" / "inputs" / "bw_schedule.txt").read_text().splitlines()
        # tag + 2 rows + END
        self.assertEqual(bw_lines[0], "BW 0")
        self.assertEqual(bw_lines[-1], "END")
        # Row 0 cells reflect the caller-provided BW (7.0), not 50.0.
        self.assertIn("7.0", bw_lines[1])
        self.assertNotIn("50.0", bw_lines[1])
        lt_lines = (tmp_root / "11" / "inputs" / "latency_schedule.txt").read_text().splitlines()
        self.assertIn("13.0", lt_lines[1])
        self.assertNotIn("500.0", lt_lines[1])


if __name__ == "__main__":
    unittest.main()
