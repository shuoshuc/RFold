import copy
import simpy
import unittest
from unittest.mock import patch, call, MagicMock

from common.flags import *
from common.job import Job, TopoType
from Cluster.cluster import Cluster
from ClusterManager.manager import ClusterManager
from ClusterManager.scheduling import SchedDecision

JOB1 = Job(
    uuid=1,
    topology=TopoType.CLOS,
    shape=(1,),
    size=1,
    duration_sec=1000,
    arrival_time_sec=0,
)


class TestClusterManager(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        self.mock_cluster = MagicMock(spec=Cluster)
        self.mgr = ClusterManager(self.env, cluster=self.mock_cluster)

    def wgen_helper(self, jobs: list[Job]):
        for job in jobs:
            yield self.env.timeout(job.arrival_time_sec - self.env.now)
            self.mgr.submitJob(job)

    def test_zero_job(self):
        """
        Verify that there is zero job submitted and scheduled.
        """
        job1 = copy.deepcopy(JOB1)
        job1.arrival_time_sec = 2
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=(SchedDecision.ADMIT, job1),
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=1)
            # The job being scheduled never arrives before simulation terminates.
            self.mgr.scheduler.place.assert_not_called()
            # Simulation ends at t = 1, new job queue should be empty.
            self.assertTrue(self.mgr.new_job_queue.empty())
            # No job is executed.
            self.assertEqual(self.mock_cluster.execute.call_count, 0)

    def test_one_job_admit(self):
        """
        Verify that there is one job submitted and scheduled.
        """
        job1 = copy.deepcopy(JOB1)
        job1.arrival_time_sec = 2
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=(SchedDecision.ADMIT, job1),
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=3)
            # The job being scheduled arrives at t = 2.
            self.mgr.scheduler.place.assert_called_once_with(job1)
            # Simulation ends at t = 3, new job queue should be empty since the only
            # job is admitted.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.mock_cluster.execute.assert_called_once_with(job1)

    def test_multi_job_admit(self):
        """
        Verify that multiple jobs submitted at different times are all admitted.
        """
        job1 = copy.deepcopy(JOB1)
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 2
        job3 = copy.deepcopy(JOB1)
        job3.arrival_time_sec = 3

        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                (SchedDecision.ADMIT, job1),
                (SchedDecision.ADMIT, job2),
                (SchedDecision.ADMIT, job3),
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2, job3]))
            self.env.run(until=5)
            # Three jobs are submitted.
            self.assertEqual(self.mgr.scheduler.place.call_count, 3)
            # New job queue should be empty since all jobs are admitted.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.mock_cluster.execute.assert_has_calls(
                [call(job1), call(job2), call(job3)]
            )

    def test_one_job_reject(self):
        """
        Verify that there is one job submitted and rejected.
        """
        job1 = copy.deepcopy(JOB1)
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=(SchedDecision.REJECT, job1),
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=DEFERRED_SCHED_SEC + 1)
            # The job is rejected for the first time at t = 0. It is deferred until
            # t = DEFERRED_SCHED_SEC, when it gets rejected again. And the job is
            # deferred util t = 2 * DEFERRED_SCHED_SEC. So by the time simulation ends,
            # the job is still deferred.
            self.assertTrue(self.mgr.new_job_queue.empty())
            # There should be two scheduling attempts: one for the initial job and
            # one when the first deferral is over.
            self.mgr.scheduler.place.assert_has_calls([call(job1), call(job1)])
            # Deferred job queue should *NOT* be empty since the job is rejected a second
            # time and still deferred when simulation completes.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.assertEqual(self.mock_cluster.execute.call_count, 0)

    def test_multi_job_reject(self):
        """
        Verify that multiple jobs are submitted and rejected.
        """
        job1 = copy.deepcopy(JOB1)
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 2
        job3 = copy.deepcopy(JOB1)
        job3.arrival_time_sec = 3
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                (SchedDecision.REJECT, job1),
                (SchedDecision.REJECT, job2),
                (SchedDecision.REJECT, job3),
                (SchedDecision.REJECT, job1),
                (SchedDecision.REJECT, job2),
                (SchedDecision.REJECT, job3),
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2, job3]))
            self.env.run(until=DEFERRED_SCHED_SEC + 1)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # When job1 is released from deferral, job2 and job3 are also retried.
            # Must be in the exact order.
            self.mgr.scheduler.place.assert_has_calls(
                [call(job1), call(job2), call(job3), call(job1), call(job2), call(job3)]
            )
            # All three jobs are still deferred when simulation completes.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 3)
            self.assertEqual(self.mock_cluster.execute.call_count, 0)

    def test_mixed_jobs(self):
        """
        Verify that a mixture of jobs are handled as expected.
        """
        jobs = []
        cnt = 3
        for t in range(cnt):
            job = Job(
                uuid=t + 1,
                topology=TopoType.CLOS,
                shape=(1,),
                size=1,
                duration_sec=1000,
                arrival_time_sec=t,
            )
            jobs.append(job)

        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                (SchedDecision.ADMIT, jobs[0]),
                (SchedDecision.REJECT, jobs[1]),
                (SchedDecision.ADMIT, jobs[2]),
                (SchedDecision.ADMIT, jobs[1]),
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper(jobs))
            # Run simulation until t = 1. Only job1 is processed.
            self.env.run(until=1)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job1 is admitted, deferred queue should be empty.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 0)
            self.mgr.scheduler.place.assert_has_calls([call(jobs[0])])
            self.mock_cluster.execute.assert_called_once_with(jobs[0])
            # Continue simulation until t = 2. Job2 is processed.
            self.env.run(until=2)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job2 is rejected, it is put into the deferred queue.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.mgr.scheduler.place.assert_has_calls([call(jobs[i]) for i in range(2)])
            self.mgr.scheduler.place.assert_has_calls([call(jobs[0])])
            # Continue simulation until right before deferral is done. Job3 is processed.
            self.env.run(until=DEFERRED_SCHED_SEC - 1)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job3 is admitted, job2 is still in the deferred queue.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.mock_cluster.execute.assert_has_calls([call(jobs[0]), call(jobs[2])])
            self.mgr.scheduler.place.assert_has_calls([call(jobs[i]) for i in range(cnt)])
            # Continue simulation until right after deferral is done. Job2 is retried.
            self.env.run(until=DEFERRED_SCHED_SEC + 2)
            self.assertTrue(self.mgr.new_job_queue.empty())
            # Job2 is taken out of the deferred queue, retried and admitted.
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 0)
            self.mgr.scheduler.place.assert_has_calls(
                [call(jobs[i]) for i in range(cnt)] + [call(jobs[1])]
            )
            self.mock_cluster.execute.assert_has_calls(
                [call(jobs[0]), call(jobs[2]), call(jobs[1])]
            )

    def test_one_job_running(self):
        """
        Verify that there is one job running.
        """
        job1 = copy.deepcopy(JOB1)
        job1.duration_sec = 10
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            return_value=(SchedDecision.ADMIT, job1),
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1]))
            self.env.run(until=2)
            # The job is admitted and runs for 10 seconds. At t = 2, the job is still
            # in the running queue.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_completion, 10)
            self.mgr.scheduler.place.assert_has_calls([call(job1)])
            self.assertEqual(self.mock_cluster.execute.call_count, 1)
            self.env.run(until=12)
            # Job1 completes at t = 10, so running queue should be empty.
            self.assertEqual(self.mgr.running_job_queue.qsize(), 0)
            # Next completion time froze at 10.
            self.assertEqual(self.mgr.next_completion, 10)

    def test_multi_job_running(self):
        """
        Verify that there are multiple jobs running.
        """
        job1 = copy.deepcopy(JOB1)
        job1.arrival_time_sec = 5
        job1.duration_sec = 1
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 7
        job2.duration_sec = 5
        job3 = copy.deepcopy(JOB1)
        job3.arrival_time_sec = 9
        job3.duration_sec = 2
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                (SchedDecision.ADMIT, job1),
                (SchedDecision.ADMIT, job2),
                (SchedDecision.ADMIT, job3),
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2, job3]))
            self.env.run(until=6)
            # Job1 is the only job running at t = 5.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_completion, 6)
            self.mgr.scheduler.place.assert_has_calls([call(job1)])
            self.assertEqual(self.mock_cluster.execute.call_count, 1)
            self.env.run(until=7)
            # Job1 should have completed at t = 6.
            self.assertEqual(self.mgr.running_job_queue.qsize(), 0)
            self.assertEqual(self.mgr.next_completion, 6)
            self.env.run(until=11)
            # Job2 and job3 are running at t = 10.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 2)
            # Next completion time should be 9 + 2 = 11 (job3).
            self.assertEqual(self.mgr.next_completion, 11)
            self.env.run(until=12)
            # Job3 completes at t = 11, job2 still running.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            # Next completion time should be 7 + 5 = 12 (job2).
            self.assertEqual(self.mgr.next_completion, 12)
            self.env.run(until=15)
            # Job2 completes at t = 12.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 0)
            # Next completion time froze at 12
            self.assertEqual(self.mgr.next_completion, 12)

    def test_running_deferral_interruption(self):
        """
        Verify that completed jobs interrupt the deferred jobs.
        """
        # Job1 is rejected at t = 0 and deferred to t = 30.
        job1 = copy.deepcopy(JOB1)
        job1.duration_sec = 1
        # Job2 is admitted at t = 2 and completes at t = 7.
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 2
        job2.duration_sec = 5
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                (SchedDecision.REJECT, job1),
                (SchedDecision.ADMIT, job2),
                (SchedDecision.ADMIT, job1),
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2]))
            self.env.run(until=2)
            # Job1 is the only job running at t = 1.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_retry, 0 + DEFERRED_SCHED_SEC)
            self.assertEqual(self.mgr.running_job_queue.qsize(), 0)
            self.assertEqual(self.mgr.next_completion, 0)
            self.mgr.scheduler.place.assert_has_calls([call(job1)])
            self.assertEqual(self.mock_cluster.execute.call_count, 0)
            self.env.run(until=3)
            # Job2 is running at t = 2.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertEqual(self.mgr.deferred_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_retry, 0 + DEFERRED_SCHED_SEC)
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_completion, 7)
            self.mgr.scheduler.place.assert_has_calls([call(job1), call(job2)])
            self.assertEqual(self.mock_cluster.execute.call_count, 1)
            self.env.run(until=8)
            # Job2 completes at t = 7. Deferred job1 is interrupted and retried, which is
            # then accepted.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            # Next completion time should be 7 + 1 = 8 (job1).
            self.assertEqual(self.mgr.next_completion, 8)
            self.mgr.scheduler.place.assert_has_calls(
                [call(job1), call(job2), call(job1)]
            )
            self.assertEqual(self.mock_cluster.execute.call_count, 2)

    def test_new_running_job_interruption(self):
        """
        Verify that new running jobs with earlier completion interrupt the existing one.
        """
        # Job1 is admitted at t = 0 and completes at t = 10.
        job1 = copy.deepcopy(JOB1)
        job1.duration_sec = 10
        # Job2 is admitted at t = 2 and completes at t = 3.
        job2 = copy.deepcopy(JOB1)
        job2.arrival_time_sec = 2
        job2.duration_sec = 1
        with patch(
            "ClusterManager.scheduling.SchedulingPolicy.place",
            side_effect=[
                (SchedDecision.ADMIT, job1),
                (SchedDecision.ADMIT, job2),
            ],
        ):
            self.env.process(self.mgr.schedule())
            self.env.process(self.wgen_helper([job1, job2]))
            self.env.run(until=2)
            # Job1 is running at t = 1.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_completion, 10)
            self.mgr.scheduler.place.assert_has_calls([call(job1)])
            self.assertEqual(self.mock_cluster.execute.call_count, 1)
            self.env.run(until=3)
            # Job2 is running at t = 2. It interrupts the running queue guard timer waiting
            # for job1.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 2)
            self.assertEqual(self.mgr.next_completion, 3)
            self.mgr.scheduler.place.assert_has_calls([call(job1), call(job2)])
            self.assertEqual(self.mock_cluster.execute.call_count, 2)
            self.env.run(until=4)
            # Job2 completes at t = 3. Job1 is the next to complete.
            self.assertTrue(self.mgr.new_job_queue.empty())
            self.assertTrue(self.mgr.deferred_job_queue.empty())
            self.assertEqual(self.mgr.running_job_queue.qsize(), 1)
            self.assertEqual(self.mgr.next_completion, 10)
            self.mgr.scheduler.place.assert_has_calls([call(job1), call(job2)])
            self.assertEqual(self.mock_cluster.execute.call_count, 2)
