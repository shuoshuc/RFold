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
        # [Producer]: schedule()
        # [Consumer]: deferredQueueGuard()
        self.deferred_job_queue: PriorityQueue[Job] = PriorityQueue()
        # A priority queue with running jobs, sorted by job completion time.
        # [Producer]: schedule(), runningQueueGuard()
        # [Consumer]: runningQueueGuard()
        self.running_job_queue: PriorityQueue[Job] = PriorityQueue()
        # An event to signal the arrival of a new job.
        # Once it fires, keep in mind to reset it to a new event.
        self.event_arrival = self.env.event()
        # An event to signal that a deferred job is enqueued.
        # Once it fires, keep in mind to reset it to a new event.
        self.event_deferral = self.env.event()
        # An event to signal that a running job is enqueued.
        # Once it fires, keep in mind to reset it to a new event.
        self.event_running = self.env.event()
        # Tracks the earliest (future) time to retry scheduling a deferred job.
        self.next_retry = self.env.now
        # Tracks the earliest (future) time that a running job will complete.
        self.next_completion = self.env.now
        # Guard process for the deferred job queue.
        self.dq_guard_proc = env.process(self.deferredQueueGuard())
        # Guard process for the running job queue.
        self.rq_guard_proc = env.process(self.runningQueueGuard())

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

    def fetchNextCompletingJobNoWait(self) -> Optional[Job]:
        """
        Fetch the job completing first from the running job queue. If no job is available,
        return None.
        """
        try:
            job = self.running_job_queue.get(block=False)
        except Empty:
            return None
        return job

    def deferredQueueGuard(self) -> Generator[simpy.events.Event, None, None]:
        """
        Guard process monitoring the deferred job queue. It waits for a job to be put
        into the deferred queue and then starts a timer to release the deferred jobs,
        i.e., moving them back to the arrival queue.
        """
        while True:
            # Wait until a deferred job is enqueued. If the process is interrupted here,
            # which means nothing is deferred, it should just go back to waiting.
            try:
                if self.deferred_job_queue.empty():
                    yield self.event_deferral
            except simpy.Interrupt:
                logging.debug(
                    f"t = {self.env.now}, deferred queue guard interrupted while waiting."
                )
                continue

            # Start the timer now that there is actually a deferred job. If the process
            # is interrupted here, what needs to be done is still the same, just that the
            # waiting time is cut short.
            try:
                yield self.env.timeout(self.next_retry - self.env.now)
                logging.debug(
                    f"t = {self.env.now}, deferred queue guard wakes up, "
                    f"queue size: {self.deferred_job_queue.qsize()}"
                )
            except simpy.Interrupt:
                logging.debug(
                    f"t = {self.env.now}, deferral interrupted, "
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

    def runningQueueGuard(self) -> Generator[simpy.events.Event, None, None]:
        """
        Guard process monitoring the running job queue. It waits for a job to be put
        into the running queue and then waits until the time has advanced to the job's
        completion time. Once the job is completed, the guard moves on to the next job.
        """
        while True:
            # Wait until a running job is enqueued. If the process is interrupted here,
            # which is unexpected, it should just go back to waiting.
            try:
                if self.running_job_queue.empty():
                    yield self.event_running
            except simpy.Interrupt:
                logging.warning(
                    f"t = {self.env.now}, running queue guard unexpectedly interrupted."
                )
                continue

            # This may seem inefficient dequeueing and enqueuing the job back and forth.
            # The main reason is to lazy update the next completion time when it is the
            # the time to wait for completion. Given that a considerable amount of
            # reordering could occur in the running queue, lazy update is less error-prone.
            # We also do not want to hold on to the job while waiting for completion, because
            # that leads to a misrepresentation of the running queue size.
            job = self.fetchNextCompletingJobNoWait()
            if job is None:
                logging.error(f"[ERROR] t = {self.env.now}, running job queue is empty.")
                continue
            self.next_completion = job.priority
            self.running_job_queue.put(job)
            # Start the timer now that there is actually a running job. The process can
            # be interrupted here if a new job with earlier completion time was enqueued.
            try:
                yield self.env.timeout(self.next_completion - self.env.now)
                # Job completion means resource frees up, notify the deferred queue guard
                # so that deferred jobs can be retried.
                job = self.fetchNextCompletingJobNoWait()
                if job is not None:
                    logging.info(f"t = {self.env.now}, {job.short_print()} completed")
                    self.dq_guard_proc.interrupt()
            except simpy.Interrupt:
                # The timer was interrupted, meaning a new running job with earlier
                # completion time was enqueued.
                logging.debug(f"t = {self.env.now}, runningQueueGuard interrupted")

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
            decision, job_to_sched = self.scheduler.place(job)
            logging.info(
                f"t = {self.env.now}, schedule {job_to_sched.short_print()}, "
                f"decision: {SchedDecision(decision).name}"
            )
            if decision == SchedDecision.ADMIT:
                # Admitted jobs are directly executed on the cluster.
                self.executeOnCluster(job_to_sched)
            elif decision == SchedDecision.REJECT:
                # Rejected jobs go into a deferred queue. To avoid busy looping, jobs
                # in the deferred queue are only checked once every `DEFERRED_SCHED_SEC`,
                # since the cluster state likely has not changed over a short period of
                # time and scheduling the same job possibly leads to the same outcome.
                # Note that jobs in the deferred queue maintain the same relative order,
                # but the entire set of jobs are essentially reordered across queues.
                self.deferJob(job_to_sched)
            elif decision == SchedDecision.PREEMPT:
                # TODO: replace with actual preemption
                # Sleep for a short period to simulate job migration delay.
                sleep = 4
                logging.info(f"t = {self.env.now}, migration delay: {sleep} sec")
                yield self.env.timeout(sleep)
                # Block until migration completes, and then execute on the cluster.
                self.executeOnCluster(job_to_sched)
            elif decision == SchedDecision.RECONFIGURE:
                # TODO: replace with actual reconfiguration
                # Sleep for a short period to simulate reconfiguration delay.
                sleep = 1
                logging.info(f"t = {self.env.now}, reconfiguration delay: {sleep} sec")
                yield self.env.timeout(sleep)
                # Block until reconfiguration completes, and then execute on the cluster.
                self.executeOnCluster(job_to_sched)

    def executeOnCluster(self, job: Job):
        """
        Send the job to the cluster for execution. Timestamp the job with the scheduled
        time and move it to the running queue for continuous tracking.
        """
        # Scheduled time = time when the job is executed.
        job.sched_time_sec = self.env.now
        # Once a job is scheduled, its priority changes from arrival time to desired
        # completion time.
        job.priority = self.env.now + job.duration_sec
        self.cluster.execute(job)
        # Move the running job into the running queue.
        self.running_job_queue.put(job)
        if self.next_completion > self.env.now and job.priority < self.next_completion:
            # This job has an earlier completion time than the existing (future)
            # completion time. Interrupt the running queue guard to restart the timer.
            self.rq_guard_proc.interrupt()
        else:
            # If the existing completion time is in the past, or the new job completes
            # after the existing completion time, it is a normal enqueue, no interrupt.
            self.event_running.succeed()
            self.event_running = self.env.event()
