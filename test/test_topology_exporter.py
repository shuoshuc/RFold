import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import simpy

from common.job import Job, TopoType
from common.utils import spec_parser
from Cluster.cluster import Cluster
from ClusterManager.astra_runner import AstraSimRunner
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
        self.model = ContentionModel(self.env, self.cluster, MagicMock(spec=AstraSimRunner))
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
        self.model = ContentionModel(self.env, self.cluster, MagicMock(spec=AstraSimRunner))
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


class TestTopologyExporterContention(unittest.TestCase):
    """Two jobs sharing a +x link on row y=0: both jobs' wrap edges land on
    the link x2-y0 -> x3-y0 (via job_b's forward) so flow_count == 2 there.
    The exported bw cell for each job's wrap edge must show eff_bw_gbps / 2 / 8."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster, MagicMock(spec=AstraSimRunner))
        self.tmpdir = tempfile.mkdtemp(prefix="topo_export_test_")
        self.exporter = TopologyExporter(self.model, self.tmpdir)
        sample_link = next(iter(self.cluster.links.values()))
        self.link_speed_gbps = sample_link.speed_gbps

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_rows(self, path):
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f]
        return [[float(v) for v in ln.split()] for ln in lines[1:-1]]

    def test_shared_link_halves_bw_cell(self):
        """Mirrors test_all_links_shared_between_two_jobs_on_same_row in
        test_contention.py: two (2,1) jobs on y=0 share all +x links of that
        row, so every comm-pattern edge of each job has eff_bw = speed/2.
        Exported cells therefore equal (speed/2)/8."""
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_a)
        self.cluster.execute(job_b)
        self.exporter.exportRunning([job_a, job_b])

        expected = (self.link_speed_gbps / 2) / 8.0
        rows_a = self._read_rows(os.path.join(self.tmpdir, "job_1_bw.txt"))
        rows_b = self._read_rows(os.path.join(self.tmpdir, "job_2_bw.txt"))
        # Both ring edges (0,1) and (1,0) are bottlenecked at speed/2.
        self.assertAlmostEqual(rows_a[0][1], expected)
        self.assertAlmostEqual(rows_a[1][0], expected)
        self.assertAlmostEqual(rows_b[0][1], expected)
        self.assertAlmostEqual(rows_b[1][0], expected)


class TestTopologyExporterFileOps(unittest.TestCase):
    """Overwrite semantics and lazy directory creation."""

    def setUp(self):
        self.env = simpy.Environment()
        self.cluster = Cluster(self.env, spec=spec_parser(C1_SPEC))
        self.model = ContentionModel(self.env, self.cluster, MagicMock(spec=AstraSimRunner))
        self.tmpdir = tempfile.mkdtemp(prefix="topo_export_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _read_rows(self, path):
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f]
        return [[float(v) for v in ln.split()] for ln in lines[1:-1]]

    def test_export_creates_missing_directory(self):
        nested = os.path.join(self.tmpdir, "does", "not", "exist", "yet")
        self.assertFalse(os.path.isdir(nested))
        exporter = TopologyExporter(self.model, nested)
        job = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job)
        exporter.export(job)
        self.assertTrue(os.path.isfile(os.path.join(nested, "job_1_bw.txt")))

    def test_second_export_overwrites_first(self):
        """A second admit changes contention; the file should reflect only the
        latest state, not append a second matrix or keep the older one."""
        sample_link = next(iter(self.cluster.links.values()))
        speed = sample_link.speed_gbps
        exporter = TopologyExporter(self.model, self.tmpdir)
        job_a = _make_t2d_job(uuid=1, shape=(2, 1), node_ranks=["x0-y0", "x1-y0"])
        self.cluster.execute(job_a)
        exporter.export(job_a)
        bw_path = os.path.join(self.tmpdir, "job_1_bw.txt")
        rows = self._read_rows(bw_path)
        self.assertAlmostEqual(rows[0][1], speed / 8.0)  # full bandwidth

        # Admit a sharing peer; export job_a again.
        job_b = _make_t2d_job(uuid=2, shape=(2, 1), node_ranks=["x2-y0", "x3-y0"])
        self.cluster.execute(job_b)
        exporter.export(job_a)
        rows2 = self._read_rows(bw_path)
        # Now bottlenecked at speed/2.
        self.assertAlmostEqual(rows2[0][1], (speed / 2) / 8.0)
        # File still has exactly N rows + header + trailer (no append).
        with open(bw_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln]
        # 1 header + 2 data rows + 1 trailer == 4 non-empty lines.
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "BW 0")
        self.assertEqual(lines[-1], "END")


class TestTopologyExporterIntegration(unittest.TestCase):
    """End-to-end: feed two overlapping T2D jobs through ClusterManager with
    FLAGS.export_topology temporarily on. After both jobs complete, both
    jobs' files exist on disk reflecting their last in-flight snapshot."""

    def setUp(self):
        from common.flags import FLAGS
        # FLAGS is a singleton built from argparse; mutate args directly and
        # restore in tearDown. patch.object is unreliable against argparse
        # Namespace because of how attribute introspection works.
        self._old_export = FLAGS.args.export_topology
        self._old_dir = FLAGS.args.topology_export_dir
        self.tmpdir = tempfile.mkdtemp(prefix="topo_export_test_")
        FLAGS.args.export_topology = True
        FLAGS.args.topology_export_dir = self.tmpdir

    def tearDown(self):
        import shutil
        from common.flags import FLAGS
        FLAGS.args.export_topology = self._old_export
        FLAGS.args.topology_export_dir = self._old_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_files_refresh_on_admit_and_complete(self):
        from unittest.mock import patch
        from ClusterManager.manager import ClusterManager
        from ClusterManager.scheduling import SchedDecision

        env = simpy.Environment()
        cluster = Cluster(env, spec=spec_parser(C1_SPEC))
        sample_link = next(iter(cluster.links.values()))
        speed = sample_link.speed_gbps

        # Two T2D jobs on row y=0: arrive 1s apart, share +x links.
        j1 = _make_t2d_job(uuid=1, shape=(2, 1),
                           node_ranks=["x0-y0", "x1-y0"], duration_sec=5)
        j2 = _make_t2d_job(uuid=2, shape=(2, 1),
                           node_ranks=["x2-y0", "x3-y0"], duration_sec=2)
        j2.arrival_time_sec = 1

        mgr = ClusterManager(env, cluster=cluster, sim_njobs=2)

        def feeder():
            yield env.timeout(0)
            mgr.submitJob(j1)
            yield env.timeout(1)
            mgr.submitJob(j2)

        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=lambda j: (SchedDecision.ADMIT, j),
        ):
            env.process(mgr.schedule())
            env.process(feeder())
            env.run()

        # After both jobs complete, j1's last snapshot was taken right after
        # j2 completed (j1 alone again): bw cell = speed/8.
        bw_path = os.path.join(self.tmpdir, "job_1_bw.txt")
        self.assertTrue(os.path.isfile(bw_path))
        with open(bw_path) as f:
            lines = [ln.rstrip("\n") for ln in f]
        rows = [[float(v) for v in ln.split()] for ln in lines[1:-1]]
        self.assertAlmostEqual(rows[0][1], speed / 8.0)
        # j2's file also exists (last snapshot taken while j2 was running).
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "job_2_bw.txt")))


if __name__ == "__main__":
    unittest.main()
