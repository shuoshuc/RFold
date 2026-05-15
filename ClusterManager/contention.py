import logging
import math
import simpy
from dataclasses import dataclass
from typing import Iterable

from common.job import Job
from Cluster.cluster import Cluster


@dataclass(frozen=True)
class MinTopoEdge:
    """
    One edge of a job's minimum topology: the bottleneck-bandwidth and
    cumulative-latency view of a single (src_rank, dst_rank) comm-pattern pair
    after the underlying multi-hop path has been collapsed to a single
    synthetic edge.
    """
    src_rank: int
    dst_rank: int
    eff_bw_gbps: float
    eff_lat_ns: float


MinTopology = list[MinTopoEdge]


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

    def computeMinTopology(self, job: Job) -> MinTopology:
        """
        Build the minimum-topology view of `job` against the current cluster
        link-flow state. One MinTopoEdge per comm-pattern edge. Empty list
        for jobs whose topology has no comm pattern (mesh, Clos).

        For each comm-pattern pair:
          eff_bw_gbps = min over path links of (link.speed_gbps / link.flow_count)
          eff_lat_ns  = sum over path links of link.latency_ns

        Precondition: `job` must have been passed to Cluster.execute() so that
        flow_count >= 1 on every link of every routed path; calling on an
        un-executed torus job raises ZeroDivisionError on the per-link division.
        """
        edges: MinTopology = []
        for src_rank, dst_rank, links in self.cluster.routeJobPaths(job):
            if not links:
                # src_rank == dst_rank (degenerate); not produced by the
                # current ring comm pattern but cheap to handle defensively.
                edges.append(
                    MinTopoEdge(src_rank, dst_rank, float("inf"), 0.0)
                )
                continue
            eff_bw_gbps = min(l.speed_gbps / l.flow_count for l in links)
            eff_lat_ns = sum(l.latency_ns for l in links)
            edges.append(MinTopoEdge(src_rank, dst_rank, eff_bw_gbps, eff_lat_ns))
        return edges

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
            # math.isclose avoids spurious log spam when a real (non-stub)
            # slowdown function returns FP-derived factors that differ only
            # in the last bit (e.g., 1.9999999999998 vs 2.0).
            if not math.isclose(old_s, new_s):
                logging.debug(
                    f"t = {self.env.now}, Job {job.uuid} slowdown "
                    f"{old_s} -> {new_s}, new ETA {job.priority}"
                )
