import csv
import logging
import numpy as np
import simpy
from io import StringIO
from collections.abc import Iterator
from typing import Tuple

from common.job import TopoType, Job, SplitShape
from ClusterManager.manager import ClusterManager


class TraceReplay:
    """
    Replays a given trace file.
    """

    def __init__(
        self, env: simpy.core.Environment, tracefile: str, cluster_mgr: ClusterManager
    ):
        self.env = env
        self.cluster_mgr = cluster_mgr
        self.jobs = []
        self._parse_trace(tracefile)
        self.job_iter = iter(self.jobs)

    def _parse_trace(self, tracefile: str):
        """
        Parses a trace in csv format.
        """
        with open(tracefile, "r", encoding="utf-8") as f:
            for line in f.readlines():
                # Skips the comment line.
                if line.startswith("#"):
                    continue
                # Ignore the job size in the trace because the size could become different depending
                # on whether fractional XPUs are allowed.
                jid, arrival_time_sec, topo_type, shape, _, duration = line.strip().split(
                    ","
                )
                shape_tup = SplitShape(shape, TopoType[topo_type])
                self.jobs.append(
                    Job(
                        uuid=int(jid),
                        arrival_time_sec=float(arrival_time_sec),
                        topology=TopoType[topo_type],
                        shape=shape_tup,
                        size=sum(shape_tup),
                        duration_sec=float(duration),
                    )
                )

    def run(self) -> Iterator[simpy.events.Timeout]:
        """
        Fetch a job from the trace and submit it to the ClusterManager.
        Repeat until the jobs run out.
        """
        job = next(self.job_iter, None)
        while job:
            # For a future job, wait until it becomes current.
            yield self.env.timeout(job.arrival_time_sec - self.env.now)
            # Submit the job to the cluster manager.
            self.cluster_mgr.submitJob(job)
            job = next(self.job_iter, None)

    def exportDist(self) -> Tuple[StringIO, StringIO]:
        """
        Converts the trace into two distributions - IAT and size distribution.
        Exports a tuple of two StringIO objects, first is IAT and second is size.
        Each StringIO is equivalent to a csv file.
        """
        buf_iat, buf_size = StringIO(), StringIO()

        size_dict = {}
        arrival_times = []
        for job in self.jobs:
            delim = "+" if job.topology == TopoType.CLOS else "x"
            job_key = (
                job.topology.name,
                delim.join(map(str, job.shape)),
                job.size,
                job.duration_sec,
            )
            arrival_times.append(job.arrival_time_sec)
            size_dict[job_key] = size_dict.setdefault(job_key, 0) + 1

        # Prepare the IAT distribution.
        IATs = np.diff(arrival_times).tolist()
        probs = [1 / len(IATs) * 100] * len(IATs)
        iat_dist = list(zip(IATs, probs))
        f_iat = csv.writer(buf_iat, delimiter=",")
        f_iat.writerow(["# IAT (sec)", "Probability (%)"])
        f_iat.writerows(iat_dist)
        buf_iat.seek(0)

        # Prepare the size distribution.
        f_size = csv.writer(buf_size, delimiter=",")
        f_size.writerow(
            ["# id", "topology", "shape", "size", "duration (sec)", "Probability (%)"]
        )
        for i, (job_key, freq) in enumerate(size_dict.items()):
            row = list(job_key) + [freq / len(self.jobs) * 100]
            row.insert(0, i)
            f_size.writerow(row)
        buf_size.seek(0)

        return (buf_iat, buf_size)
