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


if __name__ == "__main__":
    unittest.main()
