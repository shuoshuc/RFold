import logging
import os
from typing import Iterable

import numpy as np

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
        warning. The output directory is created lazily on first write.
        Bandwidth is emitted in GB/s (eff_bw_gbps / 8); latency in ns.
        """
        if job.topology not in _TORUS_TOPOLOGIES:
            logging.warning(
                f"TopologyExporter: skipping non-torus job {job.uuid} "
                f"(topology={job.topology})"
            )
            return
        n = int(job.size)
        bw = np.zeros((n, n), dtype=np.float64)
        lt = np.zeros((n, n), dtype=np.float64)
        for edge in self.contention_model.computeMinTopology(job):
            bw[edge.src_rank][edge.dst_rank] = edge.eff_bw_gbps / 8.0
            lt[edge.src_rank][edge.dst_rank] = edge.eff_lat_ns
        os.makedirs(self.output_dir, exist_ok=True)
        _write_matrix_file(
            os.path.join(self.output_dir, f"job_{job.uuid}_bw.txt"), "BW", bw
        )
        _write_matrix_file(
            os.path.join(self.output_dir, f"job_{job.uuid}_lt.txt"), "LT", lt
        )

    def exportRunning(self, jobs: Iterable[Job]) -> None:
        """Call export(job) for every job in the iterable."""
        for j in jobs:
            self.export(j)


def _write_matrix_file(path: str, tag: str, mat: np.ndarray) -> None:
    """Write a matrix in the astra-sim gen_schedule.py format:
    header "<tag> 0", N rows of space-separated %g floats, trailer "END"."""
    n = mat.shape[0]
    with open(path, "w") as f:
        f.write(f"{tag} 0\n")
        for r in range(n):
            f.write(" ".join(f"{v:g}" for v in mat[r]) + "\n")
        f.write("END\n")
