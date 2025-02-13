import unittest
import simpy
from unittest.mock import MagicMock

from common.flags import *
from common.job import TopoType, Job
from Cluster.cluster import Cluster
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay
from common.utils import spec_parser


class TestPhillyTraceReplayWithSimpy(unittest.TestCase):

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

    def test_trace_stop(self):
        self.assertEqual(len(self.trace.jobs), 111549)

        self.env.process(self.trace.run(stop_time=1))
        # Run simulation until completion. There should only be one job generated.
        self.env.run()
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


class TestC1TraceReplayWithSimpy(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        cluster = Cluster(self.env, spec=spec_parser(C1_MODEL))
        self.mgr = ClusterManager(self.env, cluster=cluster)
        self.trace = TraceReplay(self.env, tracefile=C1_TRACE, cluster_mgr=self.mgr)

    def test_run_all(self):
        """
        Test that C1 trace completes and the queueing delay breakdown is correct.
        """
        self.assertEqual(len(self.trace.jobs), 7)

        self.env.process(self.mgr.schedule())
        self.env.process(self.trace.run())
        # Run simulation to completion.
        self.env.run()
        self.mgr.job_stats = dict(sorted(self.mgr.job_stats.items()))
        # C1 trace contains 7 jobs, expect to find all of them completed.
        self.assertEqual(len(self.mgr.job_stats), 7)
        for job in self.mgr.job_stats.values():
            self.assertIsNotNone(job.queueing_delay_sec)
            self.assertIsNotNone(job.wait_on_shape_sec)
            self.assertIsNotNone(job.wait_on_resource_sec)
        # Job1 and job3 experience queueing, check their queueing delay breakdown.
        job1 = self.mgr.job_stats[1]
        self.assertEqual(job1.queueing_delay_sec, 6)
        self.assertEqual(job1.wait_on_shape_sec, 0)
        self.assertEqual(job1.wait_on_resource_sec, 6)
        job3 = self.mgr.job_stats[3]
        self.assertEqual(job3.queueing_delay_sec, 6)
        self.assertEqual(job3.wait_on_shape_sec, 4)
        self.assertEqual(job3.wait_on_resource_sec, 2)


class TestWorkloadGenWithSimpy(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mock_mgr = MagicMock(spec=ClusterManager)
        self.wgen = WorkloadGenerator(
            self.env,
            arrival_time_file=TPU_ARRIVAL_TIME_DIST,
            job_size_file=TPU_JOB_SIZES_DIST,
            cluster_mgr=self.mock_mgr,
            dur_trace=FLAGS.dur_trace_file,
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
