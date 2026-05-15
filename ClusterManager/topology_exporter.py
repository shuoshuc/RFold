import logging
import os
from typing import Iterable

from common.job import Job, TopoType
from ClusterManager.contention import ContentionModel


_TORUS_TOPOLOGIES = (TopoType.T2D, TopoType.T3D_NT, TopoType.T3D_T)


class TopologyExporter:
    """
    Writes per-job effective-topology bandwidth + latency matrices in the
    astra-sim gen_schedule.py format. Reuses ContentionModel.computeMinTopology
    so the contention model and the exporter never disagree about what
    "effective topology" means.

    Files are overwritten on every export call: only the latest snapshot of
    each job is kept on disk.
    """

    def __init__(self, contention_model: ContentionModel, output_dir: str):
        self.contention_model = contention_model
        self.output_dir = output_dir

    def export(self, job: Job) -> None:
        """
        Write bw + lt files for one job. Non-torus jobs are skipped with a
        warning (they have no comm pattern and therefore no effective
        topology). The output directory is created on first write.
        """
        if job.topology not in _TORUS_TOPOLOGIES:
            logging.warning(
                f"TopologyExporter: skipping non-torus job {job.uuid} "
                f"(topology={job.topology})"
            )
            return
        # Remaining implementation added in Task 3.
        raise NotImplementedError

    def exportRunning(self, jobs: Iterable[Job]) -> None:
        """Call export(job) for every job in the iterable."""
        for j in jobs:
            self.export(j)
