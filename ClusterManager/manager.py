import logging
import simpy
from Cluster.cluster import Cluster
from queue import PriorityQueue, Empty
from typing import Generator, Optional

from common.job import Job
from common.flags import *
from ClusterManager.scheduling import SchedulingPolicy, SchedDecision


class ClusterManager:
    def __init__(self, env: simpy.core.Environment, cluster: Cluster):
        self.env = env
        # The cluster instance under management.
        self.cluster = cluster
        # The scheduling policy module makes scheduling decisions given the
        # cluster state and a job.
        self.scheduler = SchedulingPolicy(env)
        # A priority queue with newly arrived jobs.
        # [Producer]: WorkloadGen module, deferredQueueGuard()
        # [Consumer]: schedule()
        self.new_job_queue: PriorityQueue[Job] = PriorityQueue()
        # A priority queue holding rejected jobs that are deferred for scheduling.
        # [Producer]: deferJob()
        # [Consumer]: deferredQueueGuard()
        self.deferred_job_queue: PriorityQueue[Job] = PriorityQueue()
        # A priority queue with running jobs.
        self.running_job_queue: PriorityQueue[Job] = PriorityQueue()
        # An event to signal the arrival of a new job.
        # Once it fires, keep in mind to reset it to a new event.
        self.event_arrival = self.env.event()
        # An event to signal that a deferred job is enqueued.
        # Once it fires, keep in mind to reset it to a new event.
        self.event_deferral = self.env.event()
        # Tracks the earliest (future) time to retry scheduling a deferred job.
        self.next_retry = self.env.now
        # Guard process for the deferred job queue.
        self.guard_proc = env.process(self.deferredQueueGuard())

    def submitJob(self, job: Job):
        """
        Enqueue a job into the new job queue.
        """
        self.new_job_queue.put(job)
        logging.info(f"t = {self.env.now}, enqueued: {job.short_print()}")
        self.event_arrival.succeed()
        # Reset the arrival event to a new one for next arrival.
        self.event_arrival = self.env.event()

    def deferJob(self, job: Job):
        """
        Defer a job for some time until the next attempt for scheduling.
        This method does not block, the actual waiting is done by a separate deferred
        queue guard process.
        """
        empty_before = self.deferred_job_queue.empty()
        # Make sure enqueue happens before notifying. Otherwise, the consumer wakes up
        # to an empty queue.
        # NB: if the queue is not empty before, new deferred jobs do not refresh the next
        # retry time. This means when the timer expires, all deferred jobs are batch
        # released at once. However, this can be changed to a more fine-grained per-job
        # timeout if needed.
        self.deferred_job_queue.put(job)
        if empty_before:
            self.next_retry = self.env.now + DEFERRED_SCHED_SEC
            self.event_deferral.succeed()
            self.event_deferral = self.env.event()
        logging.info(
            f"t = {self.env.now}, deferred: {job.short_print()}, next retry: t = {self.next_retry}"
        )

    def fetchOneNewJob(self) -> Generator[simpy.events.Event, None, Job]:
        """
        Fetch a job from the new job queue. Wait until success if the queue is empty.
        """
        try:
            job = self.new_job_queue.get(block=False)
        except Empty:
            yield self.event_arrival
            job = self.new_job_queue.get(block=False)
        return job

    def fetchOneDeferredJobNoWait(self) -> Optional[Job]:
        """
        Fetch a job from the deferred job queue. If no job is available, return None.
        """
        try:
            job = self.deferred_job_queue.get(block=False)
        except Empty:
            return None
        return job

    def fetchOneCompletedJob(self) -> Generator[simpy.events.Event, None, Job]:
        pass

    def deferredQueueGuard(self) -> Generator[simpy.events.Event, None, None]:
        """
        Guard process monitoring the deferred job queue. It waits for a job to be put
        into the deferred queue and then starts a timer to release the deferred jobs,
        i.e., moving them back to the arrival queue.
        """
        while True:
            # Wait until a deferred job is enqueued.
            if self.deferred_job_queue.empty():
                yield self.event_deferral
            # Only start the timer when there is actually a deferred job.
            yield self.env.timeout(self.next_retry - self.env.now)
            logging.debug(
                f"t = {self.env.now}, deferred queue guard wakes up, "
                f"queue size: {self.deferred_job_queue.qsize()}"
            )
            # Deferral is over, move deferred jobs back to the arrival queue.
            moved_jobs = []
            while not self.deferred_job_queue.empty():
                deferred_job = self.fetchOneDeferredJobNoWait()
                if deferred_job is not None:
                    self.submitJob(deferred_job)
                    moved_jobs.append(deferred_job.uuid)
            if moved_jobs:
                logging.info(f"t = {self.env.now}, released deferred jobs: {moved_jobs}")

    def schedule(self) -> Generator[simpy.events.Event, None, None]:
        """
        Schedule received jobs. The main scheduling loop should not be blocking, except for
        (1) waiting for jobs to process, (2) simulating necessary processing delay.
        """
        logging.info(f"t = {self.env.now}, cluster scheduling starts")
        while True:
            # Wait until a new job arrives or fetch one that is already available.
            # Note that a new job could also be a deferred job that is put back for retry.
            job = yield self.env.process(self.fetchOneNewJob())

            # Make a scheduling decision on the current job.
            decision = self.scheduler.place(job)
            logging.info(
                f"t = {self.env.now}, schedule {job.short_print()}, "
                f"decision: {SchedDecision(decision).name}"
            )
            if decision == SchedDecision.ADMIT:
                pass
            elif decision == SchedDecision.REJECT:
                # Rejected jobs go into a deferred queue. To avoid busy looping, jobs
                # in the deferred queue are only checked once every `DEFERRED_SCHED_SEC`,
                # since the cluster state likely has not changed over a short period of
                # time and scheduling the same job possibly leads to the same outcome.
                # Note that jobs in the deferred queue maintain the same relative order,
                # but the entire set of jobs are essentially reordered across queues.
                self.deferJob(job)
            elif decision == SchedDecision.PREEMPT:
                # TODO: replace with actual preemption
                # Sleep for a short period to simulate job migration delay.
                sleep = 4
                logging.info(f"t = {self.env.now}, migration delay: {sleep} sec")
                yield self.env.timeout(sleep)
            elif decision == SchedDecision.RECONFIGURE:
                pass
