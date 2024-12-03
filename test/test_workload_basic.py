import unittest

from common.flags import *
from common.job import TopoType
from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay


class TestTraceReplay(unittest.TestCase):

    def setUp(self):
        self.trace = TraceReplay(None, tracefile=PHILLY_TRACE, cluster_mgr=None)

    def test_loading(self):
        self.assertNotEqual(len(self.trace.jobs), 0)
        self.assertEqual(len(self.trace.jobs), 111549)

    def test_generating(self):
        job0 = self.trace.jobs[0]
        self.assertEqual(job0.uuid, 0)
        self.assertEqual(job0.arrival_time_sec, 0)
        self.assertEqual(job0.topology, TopoType.CLOS)
        self.assertEqual(job0.size, 1)

        job1 = self.trace.jobs[1]
        self.assertEqual(job1.uuid, 1)
        self.assertEqual(job1.arrival_time_sec, 266563.0)
        self.assertEqual(job1.topology, TopoType.CLOS)
        self.assertEqual(job1.size, 1)

    def test_exporting(self):
        iat, size = self.trace.exportDist()
        self.assertNotEqual(iat, None)
        self.assertNotEqual(size, None)
        iat_lines = iat.readlines()
        self.assertEqual(len(iat_lines), 111549)
        self.assertTrue(iat_lines[0].startswith("#"))
        t, prob = iat_lines[1].strip().split(",")
        self.assertAlmostEqual(float(t), 266563.0)
        self.assertAlmostEqual(float(prob), 0.0008964750600638291)
        size_lines = size.readlines()
        self.assertEqual(len(size_lines), 31654)
        self.assertTrue(size_lines[0].startswith("#"))
        job_id, _, _, size, dur, prob = size_lines[1].strip().split(",")
        self.assertEqual(int(job_id), 0)
        self.assertAlmostEqual(float(size), 1.0)
        self.assertAlmostEqual(float(dur), 3613033.0)
        self.assertAlmostEqual(float(prob), 0.000896467023460542)


class TestWorkloadGen(unittest.TestCase):

    def setUp(self):
        self.wgen = WorkloadGenerator(
            None,
            arrival_time_file=TPU_ARRIVAL_TIME_DIST,
            job_size_file=TPU_JOB_SIZES_DIST,
            cluster_mgr=None,
        )

    def test_loading(self):
        self.assertEqual(len(self.wgen.dist_iat), 51)
        self.assertEqual(len(self.wgen.dist_size), 30)

    def test_sampling(self):
        # TPU job has bounded IAT.
        self.assertGreaterEqual(self.wgen.rv_iat.rvs(size=1)[0], 0)
        self.assertLess(self.wgen.rv_iat.rvs(size=1)[0], 1700)
        job_id = self.wgen.rv_size.rvs(size=1)[0]
        # TPU job category is also bounded.
        self.assertGreaterEqual(job_id, 0)
        self.assertLessEqual(job_id, 29)
        # NB: template job has uuid = 0 and arrival time = 0.
        self.assertEqual(self.wgen.jobs[job_id].uuid, 0)
        self.assertAlmostEqual(self.wgen.jobs[job_id].arrival_time_sec, 0)
        self.assertGreater(self.wgen.jobs[job_id].size, 0)


class TestWorkloadGenExported(unittest.TestCase):

    def setUp(self):
        self.trace = TraceReplay(None, tracefile=PHILLY_TRACE, cluster_mgr=None)
        iat, size = self.trace.exportDist()
        self.wgen = WorkloadGenerator(
            None, arrival_time_file=iat, job_size_file=size, cluster_mgr=None
        )

    def test_loading(self):
        self.assertEqual(len(self.wgen.dist_iat), 878)
        self.assertEqual(len(self.wgen.dist_size), 31653)

    def test_sampling(self):
        self.assertGreaterEqual(self.wgen.rv_iat.rvs(size=1)[0], 0)
        job_id = self.wgen.rv_size.rvs(size=1)[0]
        self.assertGreaterEqual(job_id, 0)
        # NB: template job has uuid = 0 and arrival time = 0.
        self.assertEqual(self.wgen.jobs[job_id].uuid, 0)
        self.assertAlmostEqual(self.wgen.jobs[job_id].arrival_time_sec, 0)
        self.assertGreater(self.wgen.jobs[job_id].size, 0)
