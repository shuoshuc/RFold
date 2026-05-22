import logging
import math
import simpy
from dataclasses import dataclass
from typing import Iterable

from common.job import Job, TopoType
from Cluster.cluster import Cluster
from ClusterManager import astra_sim
from ClusterManager.astra_runner import AstraSimRunner, AstraJobSpec


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

    def __init__(
        self,
        env: simpy.core.Environment,
        cluster: Cluster,
        astra_runner,
    ):
        self.env = env
        self.cluster = cluster
        self.astra_runner = astra_runner

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
        Compute per-job slowdown factors. For an impacted torus job whose
        astra_ideal_dur_nsec and astra_real_dur_nsec are both set,
        factor = real / ideal. For jobs where either field is missing
        or ideal is zero (non-torus, never-computed, or degenerate),
        factor = 1.0.
        """
        factors: dict[int, float] = {}
        for job in impacted_jobs:
            ideal = job.astra_ideal_dur_nsec
            real = job.astra_real_dur_nsec
            if ideal is None or real is None or ideal <= 0:
                factors[job.uuid] = 1.0
            else:
                factors[job.uuid] = real / ideal
        return factors

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
        Compute the min topology for each impacted job, dispatch the
        real-JCT astra-sim batch in parallel, then call the slowdown
        function and push the new factor onto each impacted job via
        Job.applySlowdown.

        Single-NPU torus jobs (prod(shape) <= 1) short-circuit to
        astra_real_dur_nsec = astra_ideal_dur_nsec without invoking
        astra-sim. Non-torus jobs (Clos, Mesh) are skipped from the
        dispatch but still go through slowdown (which returns 1.0 for
        them).
        """
        if not impacted:
            return
        min_topos = {j.uuid: self.computeMinTopology(j) for j in impacted}

        # Build the dispatch list for impacted torus jobs.
        specs: list[AstraJobSpec] = []
        for job in impacted:
            if job.topology not in (TopoType.T2D, TopoType.T3D_NT, TopoType.T3D_T):
                continue
            shape = tuple(int(s) for s in job.shape)
            if math.prod(shape) <= 1:
                # No collective communication on a single rank; slowdown is 1.0.
                job.astra_real_dur_nsec = job.astra_ideal_dur_nsec
                continue
            bw, lt = _matrices_from_min_topology(min_topos[job.uuid], int(job.size))
            specs.append(AstraJobSpec(uuid=job.uuid, shape=shape, bw=bw, lt=lt))

        if specs:
            results = self.astra_runner.runMany(specs)
            uuid_to_job = {j.uuid: j for j in impacted}
            for uuid, real_nsec in results.items():
                job = uuid_to_job[uuid]
                ideal = job.astra_ideal_dur_nsec
                if ideal is not None and real_nsec < ideal:
                    logging.warning(
                        f"astra real ({real_nsec}) < ideal ({ideal}) "
                        f"for job {uuid}"
                    )
                job.astra_real_dur_nsec = real_nsec
                logging.debug(
                    f"t = {self.env.now}, Job {uuid} "
                    f"real_ns={real_nsec} ideal_ns={ideal}"
                )

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
        resulting JCT (in nanoseconds) onto `job.astra_ideal_dur_nsec`.

        Idempotent: if `job.astra_ideal_dur_nsec` is already set, returns
        immediately. Single-NPU torus shapes (prod(shape) <= 1) are
        short-circuited to 1.0 ns without invoking astra-sim (the
        analytical backend rejects npus_count <= 1).
        """
        if job.astra_ideal_dur_nsec is not None:
            return
        shape = tuple(int(s) for s in job.shape)
        if math.prod(shape) <= 1:
            job.astra_ideal_dur_nsec = 1.0
            return
        bw = astra_sim.build_bw_matrix(shape)
        lt = astra_sim.build_lt_matrix(shape)
        job.astra_ideal_dur_nsec = self.astra_runner.runOne(
            uuid=job.uuid, shape=shape, bw=bw, lt=lt,
        )


def _matrices_from_min_topology(
    topo: MinTopology, n: int
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Translate a MinTopology (one entry per comm-pattern edge) into the
    N x N BW (GB/s) and LT (ns) matrices that astra-sim's fluid model
    consumes. Mirrors TopologyExporter's conventions: eff_bw_gbps / 8
    for BW; eff_lat_ns passed through unchanged.

    Edges are directed; the matrices reflect each MinTopoEdge as a
    single (src_rank, dst_rank) cell. Callers that need a symmetric
    matrix should already have symmetric edges in `topo`.
    """
    bw = [[0.0] * n for _ in range(n)]
    lt = [[0.0] * n for _ in range(n)]
    for e in topo:
        bw[e.src_rank][e.dst_rank] = e.eff_bw_gbps / 8.0
        lt[e.src_rank][e.dst_rank] = e.eff_lat_ns
    return bw, lt
