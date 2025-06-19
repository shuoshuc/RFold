import bisect
import logging
import simpy
from Cluster.cluster import Cluster
from typing import Generator, Optional

from common.job import Job
from common.flags import FLAGS
from common.utils import Signal
from ClusterManager.scheduling import SchedulingPolicy, SchedDecision


class SortedList:
    """
    A sorted list implementation that can be used as a job queue.
    """

    def __init__(self):
        self.slist: list[Job] = []

    def enqueue(self, value: Job):
        """
        Inserts while keeping the list sorted
        """
        bisect.insort(self.slist, value)

    def peek(self) -> Optional[Job]:
        """
        Gets the head of the list, not removing it.
        """
        return next(iter(self.slist), None)

    def remove(self, value: Job):
        """
        Removes the first occurrence matching the uuid.
        """
        for i, job in enumerate(self.slist):
            if job.uuid == value.uuid:
                del self.slist[i]
                break

    def dequeue(self) -> Job:
        """
        Pops the head of the list.
        """
        return self.slist.pop(0)

    def __len__(self) -> int:
        """
        Returns the length of the list.
        """
        return len(self.slist)

    def __repr__(self) -> str:
        return f"SortedList({self.slist})"


class ClusterManager:
    def __init__(
        self,
        env: simpy.core.Environment,
        cluster: Cluster,
        closed_loop_threshold: int = 0,
    ):
        self.env = env
        # The cluster instance under management.
        self.cluster = cluster
        # The scheduling policy module makes scheduling decisions given the
        # cluster state and a job.
        self.scheduler = SchedulingPolicy(env, cluster)
        # A queue of newly arrived jobs, sorted by arrival times.
        # [Producer]: WorkloadGen module
        # [Consumer]: schedule()
        self.new_job_queue: SortedList[Job] = SortedList()
        # A queue of running jobs, sorted by job completion times.
        # [Producer]: schedule()
        # [Consumer]: runningQueueGuard()
        self.running_job_queue: SortedList[Job] = SortedList()
        # An event to signal the arrival of a new job.
        self.event_arrival = Signal(env)
        # An event to signal that a running job is enqueued.
        self.event_running = Signal(env)
        # An event to signal that a running job is completed.
        self.event_completion = Signal(env)
        # Tracks the earliest (future) time that a running job will complete.
        self.next_completion = self.env.now
        # Guard process for the running job queue.
        self.running_guard_proc = env.process(self.runningQueueGuard())
        # A list of job UUIDs to watch for. Simulation only terminates when they complete.
        self.jobs_to_watch: list[int] = []
        # A map from job UUID to job, tracking statistics.
        self.job_stats: dict[int, Job] = {}
        # A list of cluster statistics, each tuple is
        # (time, utilization, # jobs running, # jobs queued).
        self.cluster_stats: list[tuple[int, float, int, int]] = []
        # If set, new job queue will drop jobs exceeding this threshold.
        self.closed_loop_threshold: int = closed_loop_threshold

    def submitJob(self, job: Job, wait_to_complete: bool):
        """
        Enqueue a job into the new job queue. If `wait_to_complete` is True, simulation
        only terminates after the job is completed.
        """
        if len(self.new_job_queue) > self.closed_loop_threshold > 0:
            logging.debug(
                f"t = {self.env.now}, new job queue len {len(self.new_job_queue)}, "
                f"dropping job: {job.short_print()}"
            )
            return
        self.new_job_queue.enqueue(job)
        logging.debug(f"t = {self.env.now}, enqueued: {job.short_print()}")
        if wait_to_complete:
            self.jobs_to_watch.append(job.uuid)
        self.event_arrival.trigger()

    def fetchOneNewJob(self) -> Generator[simpy.events.Event, None, Job]:
        """
        Fetch a job from the new job queue. Wait until success if none is available.
        """
        job = self.new_job_queue.peek()
        if job is None:
            # There is no job available, just wait.
            yield self.event_arrival.signal()
            return self.new_job_queue.peek()
        elif job.sched_time_sec is not None:
            # There is a job, but it has failed scheduling at least once.
            # It should not be retried immediately, wait until another job arrives or
            # a running one completes.
            yield self.event_arrival.signal() | self.event_completion.signal()
            return job
        else:
            # This job is fresh new, try to schedule immediately.
            return job

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
                if self.running_job_queue.peek() is None:
                    yield self.event_running.signal()
            except simpy.Interrupt:
                logging.warning(
                    f"t = {self.env.now}, running queue guard unexpectedly interrupted."
                )
                continue

            job = self.running_job_queue.peek()
            if job is None:
                logging.error(f"[ERROR] t = {self.env.now}, running job queue is empty.")
                continue
            self.next_completion = job.priority
            # Start the timer now that there is actually a running job. The process can
            # be interrupted here if a new job with earlier completion time was enqueued.
            try:
                yield self.env.timeout(self.next_completion - self.env.now)
                self.running_job_queue.remove(job)
                # This job is being watched, should be removed from the watch list.
                if job.uuid in self.jobs_to_watch:
                    self.jobs_to_watch.remove(job.uuid)
                logging.debug(f"t = {self.env.now}, {job.short_print()} completed")
                self.completeOnCluster(job)
            except simpy.Interrupt:
                # The timer was interrupted, meaning a new running job with earlier
                # completion time was enqueued.
                logging.debug(f"t = {self.env.now}, runningQueueGuard interrupted")

    def schedule(self) -> Generator[simpy.events.Event, None, None]:
        """
        Schedule received jobs. The main scheduling loop should not be blocking, except for
        (1) waiting for scheduling decision, (2) simulating necessary processing delay.
        """
        logging.debug(f"t = {self.env.now}, cluster scheduling starts")
        while True:
            # Wait until a new job arrives or fetch one that is already available.
            # Specific scheduling policies can decide which job to pick.
            job = yield self.env.process(self.fetchOneNewJob())

            # Make a scheduling decision on the current job.
            decision, job_to_sched = self.scheduler.place(job)
            # Once scheduling is attempted, change the sched time to something other
            # than None. This is to distinguish fresh new jobs.
            job.sched_time_sec = float("-inf")
            logging.info(
                f"t = {self.env.now}, schedule {job_to_sched.short_print()}, "
                f"decision: {SchedDecision(decision).name}"
            )
            if decision == SchedDecision.ADMIT:
                # Admitted jobs are directly executed on the cluster.
                self.executeOnCluster(job_to_sched)
                # The admitted job is sent for execution, resource usage should be updated now.
                # Scan the queue to see if the reject reason has changed for any queued job
                # as a result of this admission.
                # Need to be extra careful not to modify any of the fields used for sorting.
                for queued_job in self.new_job_queue.slist:
                    # Rejected jobs with reason "shape" are the ones interesting.
                    if queued_job.reject_reason != "shape":
                        continue
                    # Reason has changed to "resource" because of the admitted job.
                    if not self.scheduler.check_total_xpu(
                        queued_job
                    ) or not self.scheduler.check_total_node(queued_job):
                        queued_job.logRejectReason(self.env.now, "resource")
            elif decision == SchedDecision.REJECT:
                # We need to filter the infeasible shapes for static torus in order to
                # get firstfit to run.
                if FLAGS.place_policy in ["firstfit", "folding"] and any(
                    sz > min(FLAGS.dim) for sz in job.shape
                ):
                    self.new_job_queue.remove(job)
                    if job.uuid in self.jobs_to_watch:
                        self.jobs_to_watch.remove(job.uuid)
                    job.queueing_delay_sec = float("inf")
                    job.jct_sec = float("inf")
                    job.slowdown = float("inf")
                    self.job_stats[job.uuid] = job
                    logging.info(
                        f"t = {self.env.now}, job shape exceeds torus dimension, skipping job {job.short_print()}"
                    )
                    continue
                if len(self.running_job_queue) <= 0:
                    raise RuntimeError(
                        f"t = {self.env.now}, cluster is empty, but job is rejected: "
                        f"{job_to_sched.short_print()}"
                    )
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

            # Watch list closes, it will only be drained going forward.
            if self.env.now > FLAGS.sim_mark_sec:
                logging.info(
                    f"Wait list has closed, there are {len(self.jobs_to_watch)} jobs."
                )
            # All jobs to watch for have completed, time to exit.
            if len(self.jobs_to_watch) <= 0:
                return

    def executeOnCluster(self, job: Job):
        """
        Send the job to the cluster for execution. Timestamp the job with the scheduled
        time and move it to the running queue for continuous tracking.
        """
        # Scheduled time = time when the job is executed.
        job.updateQueueingTime(self.env.now)
        # Once a job is scheduled, its priority changes from arrival time to desired
        # completion time.
        job.priority = self.env.now + job.duration_sec
        self.cluster.execute(job)
        # Move the running job into the running queue.
        self.new_job_queue.remove(job)
        self.running_job_queue.enqueue(job)
        if self.next_completion > self.env.now and job.priority < self.next_completion:
            # This job has an earlier completion time than the existing (future)
            # completion time. Interrupt the running queue guard to restart the timer.
            self.running_guard_proc.interrupt()
        else:
            # If the existing completion time is in the past, or the new job completes
            # after the existing completion time, it is a normal enqueue, no interrupt.
            self.event_running.trigger()
        self.logClusterStats()

    def completeOnCluster(self, job: Job):
        """
        Send the completing job to the cluster to free up resources. Job statistics
        are also updated.
        """
        self.cluster.complete(job)
        # Update job statistics.
        job.completion_time_sec = self.env.now
        job.jct_sec = self.env.now - job.arrival_time_sec
        job.slowdown = job.jct_sec / job.duration_sec
        self.job_stats[job.uuid] = job
        # Notify the main schedule loop.
        self.event_completion.trigger()
        self.logClusterStats()

    def flushAllQueues(self):
        """
        Flush all jobs (new, running) in the queues and add them to `self.job_stats`
        for complete statistics. This method is destructive so should only be called
        when the main schedule loop exits.
        """
        for queue in [
            self.new_job_queue,
            self.running_job_queue,
        ]:
            try:
                while job := queue.dequeue():
                    self.job_stats[job.uuid] = job
            except IndexError:
                continue

    def logClusterStats(self):
        """
        Log one sample of cluster statistics.
        """
        self.cluster_stats.append(
            (
                self.env.now,
                (self.cluster.numNodes() - self.cluster.totalIdleNodes())
                / self.cluster.numNodes(),
                len(self.new_job_queue.slist),
                len(self.running_job_queue.slist),
            )
        )
