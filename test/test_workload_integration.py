import unittest
import simpy
from unittest.mock import MagicMock

from common.flags import *
from common.job import TopoType, Job
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay


class TestTraceReplayWithSimpy(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mock_mgr = MagicMock(spec=ClusterManager)
        self.trace = TraceReplay(
            self.env, tracefile=PHILLY_TRACE, cluster_mgr=self.mock_mgr
        )

    def test_run_one(self):
        self.assertEqual(len(self.trace.jobs), 111549)

        self.env.process(self.trace.run())
        # Run simulation for 1 second. There should only be one job generated.
        self.env.run(1)
        self.mock_mgr.submitJob.assert_called_once_with(
            Job(
                uuid=0,
                topology=TopoType.CLOS,
                shape=(1,),
                size=1,
                arrival_time_sec=0.0,
                duration_sec=3613033.0,
                sched_time_sec=None,
            )
        )
        # The simulation time should have advanced to t = 1 sec.
        self.assertAlmostEqual(self.env.now, 1.0)

    def test_run_all(self):
        self.assertEqual(len(self.trace.jobs), 111549)

        self.env.process(self.trace.run())
        # Run simulation to completion.
        self.env.run()
        # The submitJob() method should be called at least as many times as the job count.
        # Could be more if some jobs are rejected/deferred and retried.
        self.assertGreaterEqual(self.mock_mgr.submitJob.call_count, 111549)
        # The simulation time should have advanced to a very large time, e.g., t = 10000 sec.
        self.assertGreater(self.env.now, 10000.0)


class TestWorkloadGenWithSimpy(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mock_mgr = MagicMock(spec=ClusterManager)
        self.wgen = WorkloadGenerator(
            self.env,
            arrival_time_file=TPU_ARRIVAL_TIME_DIST,
            job_size_file=TPU_JOB_SIZES_DIST,
            cluster_mgr=self.mock_mgr,
        )

    def test_run_one(self):
        self.assertEqual(len(self.wgen.dist_iat), 51)
        self.assertEqual(len(self.wgen.dist_size), 30)

        self.env.process(self.wgen.run())
        # Run simulation for 1 second. There should only be one job generated.
        self.env.run(1)
        self.mock_mgr.submitJob.assert_called_once()
        self.assertGreater(self.wgen.abs_time_sec, 0)
        # The simulation time should have advanced to t = 1 sec.
        self.assertAlmostEqual(self.env.now, 1.0)

    def test_run_all(self):
        self.assertEqual(len(self.wgen.dist_iat), 51)
        self.assertEqual(len(self.wgen.dist_size), 30)

        self.env.process(self.wgen.run())
        # Run simulation for 10000 seconds. There should be more than one job generated.
        self.env.run(10000)
        self.assertGreater(self.mock_mgr.submitJob.call_count, 1)
        self.assertGreater(self.wgen.abs_time_sec, 0)
        # The simulation time should have advanced to t = 10000 sec.
        self.assertAlmostEqual(self.env.now, 10000.0)
