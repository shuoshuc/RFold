import simpy
import logging
from Cluster.cluster import Cluster
from multiprocessing import Queue

from common.job import Job


class ClusterManager:
    def __init__(self, env: simpy.core.Environment, cluster: Cluster):
        self.env = env
        # The cluster instance under management.
        self.cluster = cluster
        # A queue with newly arrived jobs.
        self.new_job_queue: Queue[Job] = Queue()

    def submitJob(self, job: Job):
        '''
        Enqueue a job into the new job queue.
        '''
        self.new_job_queue.put(job)
        logging.info(
            f't = {self.env.now}, job: {job.uuid}, arrival: {job.arrival_time_sec}')
