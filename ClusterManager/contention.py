import logging
import math
import simpy
from dataclasses import dataclass
from typing import Iterable

from pathlib import Path

from common.job import Job
from Cluster.cluster import Cluster
from ClusterManager import astra_sim


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

    def findImpactedJobs(
        self,
        triggering_job: Job,
        running_jobs: Iterable[Job],
        is_admit: bool,
    ) -> set[Job]:
        """
        Return the set of running jobs whose comm-pattern paths share at
        least one physical link with `triggering_job`'s paths.

        For admit events, `triggering_job` is included in the set; for
        complete events it is excluded (no longer running).

        A non-torus triggering job (no comm pattern) has no paths to share:
          - admit:    returns {triggering_job}
          - complete: returns set()
        """
        changed_links: set[str] = set()
        for _src, _dst, links in self.cluster.routeJobPaths(triggering_job):
            changed_links.update(l.name for l in links)
        if not changed_links:
            return {triggering_job} if is_admit else set()

        impacted: set[Job] = set()
        for job in running_jobs:
            if job.uuid == triggering_job.uuid:
                continue
            for _src, _dst, links in self.cluster.routeJobPaths(job):
                if any(l.name in changed_links for l in links):
                    impacted.add(job)
                    break
        if is_admit:
            impacted.add(triggering_job)
        return impacted

    def slowdown(
        self,
        impacted_jobs: Iterable[Job],
        min_topologies: dict[int, MinTopology],
    ) -> dict[int, float]:
        """
        Pluggable slowdown function. Inputs:
          - impacted_jobs: the subset of running jobs whose contention state
            just changed (share a link with the triggering job; includes the
            admitted job for admit events).
          - min_topologies: {job.uuid: MinTopology} for every impacted job.
            Each MinTopoEdge carries effective bandwidth and cumulative
            latency for one comm-pattern pair.
        Returns: {job.uuid: factor}. Factor >= 1.0 means slower; 1.0 means
        no contention. Identity stub by default; does NOT consume
        min_topologies in this revision.
        """
        return {job.uuid: 1.0 for job in impacted_jobs}

    def onAdmit(self, admitted_job: Job, running_jobs: Iterable[Job]) -> None:
        """
        Called by ClusterManager.executeOnCluster after a job has been
        admitted and link flows have been bumped. Updates slowdown factors
        on the admitted job and on every other running job whose paths
        share a link with it.
        """
        impacted = self.findImpactedJobs(admitted_job, running_jobs, is_admit=True)
        self._applyToImpacted(impacted)

    def onComplete(self, completed_job: Job, running_jobs: Iterable[Job]) -> None:
        """
        Called by ClusterManager.completeOnCluster after a job has been
        completed and link flows have been decremented. Updates slowdown
        factors on every remaining running job whose paths shared a link
        with the completed job.
        """
        impacted = self.findImpactedJobs(completed_job, running_jobs, is_admit=False)
        self._applyToImpacted(impacted)

    def _applyToImpacted(self, impacted: set[Job]) -> None:
        """
        Compute the min topology for each impacted job, call the pluggable
        slowdown function, then push the new factor onto each impacted job
        via Job.applySlowdown. Non-impacted jobs are intentionally left
        untouched (their slowdown didn't change, so the next event that
        touches them will accrue work correctly over the unchanged-factor
        interval).
        """
        if not impacted:
            return
        min_topos = {j.uuid: self.computeMinTopology(j) for j in impacted}
        factors = self.slowdown(impacted, min_topos)
        for job in impacted:
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

    def runAstraSim(self, job: Job) -> None:
        """
        Drive a fluid-model astra-sim run for `job` and write the
        resulting JCT (in seconds) onto `job.astra_dur_sec`. Assumes
        the caller has already restricted invocation to torus jobs;
        coerces `job.shape` to a tuple of ints.
        """
        shape = tuple(int(s) for s in job.shape)
        job.astra_dur_sec = astra_sim.run_astra(
            uuid=job.uuid,
            shape=shape,
            tmp_root=Path("./tmp"),
        )
