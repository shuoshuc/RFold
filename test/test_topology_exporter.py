import logging
import os
import tempfile
import unittest

import simpy

from common.job import Job, TopoType
from common.utils import spec_parser
from Cluster.cluster import Cluster
from ClusterManager.contention import ContentionModel
from ClusterManager.topology_exporter import TopologyExporter


C1_SPEC = "Cluster/models/c1.json"


def _make_t2d_job(uuid: int, shape, node_ranks: list[str], duration_sec: float = 10.0) -> Job:
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


class TestTopologyExporterSkip(unittest.TestCase):
    """Non-torus jobs must be skipped with a warning and no file output."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster)
        self.tmpdir = tempfile.mkdtemp(prefix="topo_export_test_")
        self.exporter = TopologyExporter(self.model, self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_clos_job_is_skipped_with_warning(self):
        clos_job = Job(
            uuid=42,
            topology=TopoType.CLOS,
            shape=(1,),
            size=1,
            duration_sec=10.0,
            arrival_time_sec=0,
        )
        clos_job.addToAllocation("x0-y0")
        with self.assertLogs(level="WARNING") as cm:
            self.exporter.export(clos_job)
        # No files created.
        self.assertEqual(os.listdir(self.tmpdir), [])
        # Warning message identifies the job.
        self.assertTrue(any("42" in line for line in cm.output))


class TestTopologyExporterFormat(unittest.TestCase):
    """Format round-trip: header, dims, trailer, exact cell values for a
    single uncontended (2,1) job on a 4x4 T2D cluster."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster)
        self.tmpdir = tempfile.mkdtemp(prefix="topo_export_test_")
        self.exporter = TopologyExporter(self.model, self.tmpdir)
        # Capture cluster's uniform per-link speed + latency (c1.json values).
        sample_link = next(iter(self.cluster.links.values()))
        self.link_speed_gbps = sample_link.speed_gbps
        self.link_lat_ns = sample_link.latency_ns

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_matrix(self, path):
        """Return (tag, topo_id, list-of-rows-of-floats) from a written file."""
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f]
        header = lines[0].split()
        tag, topo_id = header[0], int(header[1])
        self.assertEqual(lines[-1], "END")
        rows = [[float(v) for v in ln.split()] for ln in lines[1:-1]]
        return tag, topo_id, rows

    def test_bw_and_lt_files_have_expected_shape_and_values(self):
        """(2,1) T2D job alone: bw = link_speed/8 on both edges; lt = 1*link_lat
        on the forward edge and 3*link_lat on the wrap edge. Off-pattern cells
        and the diagonal are 0."""
        job = _make_t2d_job(uuid=7, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        self.exporter.export(job)
        bw_path = os.path.join(self.tmpdir, "job_7_bw.txt")
        lt_path = os.path.join(self.tmpdir, "job_7_lt.txt")
        self.assertTrue(os.path.isfile(bw_path))
        self.assertTrue(os.path.isfile(lt_path))

        bw_tag, bw_id, bw_rows = self._read_matrix(bw_path)
        self.assertEqual(bw_tag, "BW")
        self.assertEqual(bw_id, 0)
        # N = job.size = 2.
        self.assertEqual(len(bw_rows), 2)
        self.assertTrue(all(len(r) == 2 for r in bw_rows))
        # Forward edge (0,1): speed/8. Wrap edge (1,0): speed/8 (uncontended).
        self.assertAlmostEqual(bw_rows[0][1], self.link_speed_gbps / 8.0)
        self.assertAlmostEqual(bw_rows[1][0], self.link_speed_gbps / 8.0)
        # Diagonal stays zero.
        self.assertEqual(bw_rows[0][0], 0.0)
        self.assertEqual(bw_rows[1][1], 0.0)

        lt_tag, lt_id, lt_rows = self._read_matrix(lt_path)
        self.assertEqual(lt_tag, "LT")
        self.assertEqual(lt_id, 0)
        # Forward = 1 hop, wrap = 3 hops.
        self.assertAlmostEqual(lt_rows[0][1], self.link_lat_ns)
        self.assertAlmostEqual(lt_rows[1][0], 3 * self.link_lat_ns)
        self.assertEqual(lt_rows[0][0], 0.0)
        self.assertEqual(lt_rows[1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
