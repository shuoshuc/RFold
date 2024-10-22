import logging
import simpy
from Cluster.cluster import Cluster
from queue import PriorityQueue, Empty
from typing import Generator

from common.job import Job
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
        self.new_job_queue: PriorityQueue[Job] = PriorityQueue()
        # An event to signal the arrival of a new job.
        # Once it fires, keep in mind to reset it to a new event.
        self.event_arrival = self.env.event()

    def submitJob(self, job: Job):
        """
        Enqueue a job into the new job queue.
        """
        self.new_job_queue.put(job)
        logging.info(f"t = {self.env.now}, enqueued: {job.short_print()}")
        self.event_arrival.succeed()
        # Reset the arrival event to a new one for next arrival.
        self.event_arrival = self.env.event()

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

    def schedule(self) -> Generator[simpy.events.Event, None, None]:
        """
        Schedule jobs from the new job queue.
        """
        logging.info(f"t = {self.env.now}, cluster scheduling starts")
        while True:
            # Wait for a new job to arrive (might return immediately if job queue is not empty).
            job = yield self.env.process(self.fetchOneNewJob())
            logging.info(f"t = {self.env.now}, new job: {job.short_print()}")

            # Try to schedule the current job.
            decision = self.scheduler.place(job)
            logging.info(
                f"t = {self.env.now}, job {job.uuid}, sched: {SchedDecision(decision).name}"
            )
            if decision == SchedDecision.ADMIT:
                pass
            elif decision == SchedDecision.REJECT:
                # If the job cannot be scheduled right away, put it back and wait for another chance.
                self.new_job_queue.put(job)
                # TODO: wait on some event before retrying.
                # If there is no new job other than this one, retrying leads to the same rejection.
                # So either (1) a new job arrives and out-of-order is allowed, we try scheduling the new job,
                # or (2) we wait for a running job to complete and free up resources,
                # or (3) some time has passed and the scheduling decision would change (decision based on remaining job time).
            elif decision == SchedDecision.PREEMPT:
                # TODO: replace with actual preemption
                # Sleep for a short period to simulate job migration delay.
                sleep = 4
                logging.info(f"t = {self.env.now}, migration delay: {sleep} sec")
                yield self.env.timeout(sleep)
            elif decision == SchedDecision.RECONFIGURE:
                pass
