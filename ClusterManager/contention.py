import logging
import simpy
from typing import Iterable

from common.job import Job
from Cluster.cluster import Cluster


class ContentionModel:
    """
    Models network/resource contention between concurrently running jobs and
    converts it into per-job slowdown factors. The factors are pushed onto each
    Job, which uses them to track its own progress.

    The slowdown function is the pluggable boundary. The default identity
    implementation returns 1.0 for every job, making contention modeling a
    no-op. Replace with a real model when ready.
    """

    def __init__(self, env: simpy.core.Environment, cluster: Cluster):
        self.env = env
        self.cluster = cluster

    def slowdown(self, running_jobs: Iterable[Job]) -> dict[int, float]:
        """
        Pluggable slowdown function. Inputs:
          - running_jobs: every currently placed job. Each Job's .allocation
            field carries the exact job-to-node mapping; cluster topology is
            available via self.cluster.
        Returns: {job.uuid: slowdown_factor}. Factor >= 1.0 means slower; 1.0
        means no contention. Identity stub by default.
        """
        return {job.uuid: 1.0 for job in running_jobs}

    def recompute(self, running_jobs: Iterable[Job]) -> None:
        """
        Recompute slowdowns for all currently running jobs and push the new
        factor onto each. Jobs update their own work-done accumulator and ETA.
        Logs a debug line whenever a factor changes.
        """
        jobs = list(running_jobs)
        factors = self.slowdown(jobs)
        for job in jobs:
            old_s = job.current_slowdown
            new_s = factors[job.uuid]
            job.applySlowdown(new_s, self.env.now)
            if old_s != new_s:
                logging.debug(
                    f"t = {self.env.now}, Job {job.uuid} slowdown "
                    f"{old_s} -> {new_s}, new ETA {job.priority}"
                )
