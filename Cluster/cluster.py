import logging
import simpy
from queue import PriorityQueue, Empty

from common.job import Job
from common.flags import *


class Cluster:
    """
    A cluster is the high-level abstraction of the resources.
    It contains a collection of nodes, and also models the network topology, link
    bandwidth, etc. The cluster manager and other external components interact with
    this class to access the resources. This class also exposes certain states for
    monitoring purposes.
    """

    def __init__(self, env: simpy.core.Environment):
        self.env = env

    def execute(self, job: Job):
        """
        Executes a job on the cluster. The job is broken down into subjobs and sent to
        nodes for execution. The shape and duration of the job remain as-is and should
        always succeed execution, because the scheduler admits the job only when it can
        successfully start.
        """
        logging.info(
            f"t = {self.env.now}, executing job {job.uuid}, shape {job.shape}, "
            f"duration {job.duration_sec}."
        )
        # TODO: break down the job into subjobs, send to nodes for execution.
        # Update states.
