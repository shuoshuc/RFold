import logging
import random
import simpy
from enum import Enum
from typing import Tuple

from common.flags import *
from common.job import Job
from Cluster.cluster import Cluster


class SchedDecision(Enum):
    # Admit a job immediately.
    ADMIT = 1
    # Reject a job immediately.
    REJECT = 2
    # Preempt running jobs to admit a job.
    # May take some time to migrate.
    PREEMPT = 3
    # Reconfigure the cluster topology to admit a job.
    # May take some time.
    RECONFIGURE = 4


class SchedulingPolicy:
    def __init__(self, env: simpy.core.Environment, cluster: Cluster):
        self.env = env
        self.cluster = cluster
        random.seed(42)

    def randDecision(self, job: Job) -> Tuple[SchedDecision, Job]:
        """
        Makes a random scheduling decision and does not modify the given job.
        To be used for testing purposes.
        """
        return random.choice([d for d in SchedDecision]), job

    def _simplefit(self, job: Job) -> Tuple[SchedDecision, Job]:
        node_id = "n1"
        avail_xpus = self.cluster.getIdleXPU(node_id)
        for x in job.shape:
            if x <= avail_xpus:
                job.allocation[node_id] = x
                avail_xpus -= x
            else:
                logging.debug(f"Job {job.uuid} rejected, insufficient number of XPUs.")
                return SchedDecision.REJECT, job
        return SchedDecision.ADMIT, job

    def _firstfit(self, job: Job) -> Tuple[SchedDecision, Job]:
        num_req_nodes = len(job.shape)
        if num_req_nodes > self.cluster.numNodes():
            logging.debug(f"Job {job.uuid} rejected, insufficient number of nodes.")
            return SchedDecision.REJECT, job
        # for x in job.shape:
        #     for node in self.cluster.nodes.values():
        #         if x <= node.numIdleXPU():
        #             job.allocation[node.name] = x
        #             break
        return SchedDecision.ADMIT, job

    def place(self, job: Job, policy: str = SCHED_POLICY) -> Tuple[SchedDecision, Job]:
        """
        Make a scheduling decision for a job. Note that the job (e.g., shape, duration)
        could be modifed to achieve a more desirable scheduling decision. But if the job
        is rejected, it is not modified.
        The (modified) job is returned along with the decision.
        """
        if policy == "firstfit":
            return self._firstfit(job)
        elif policy == "simplefit":
            return self._simplefit(job)
        # The default fallback is to reject all jobs.
        return SchedDecision.REJECT, job
