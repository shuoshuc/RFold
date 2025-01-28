import logging
import random
import simpy
import numpy as np
from enum import Enum
from numpy.typing import NDArray
from typing import Optional, Tuple

from common.flags import *
from common.job import Job, TopoType
from Cluster.cluster import Cluster
from itertools import permutations


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

    def _find_submesh(
        self, array: NDArray[np.float64], a: int, b: int
    ) -> Optional[tuple[int, int]]:
        """
        Finds a feasible AxB shaped submesh in the 2D mesh/torus.

        Parameters:
        array: 2D numpy array, where 0 means no XPU available.
        a: job shape along dimension x
        b: job shape along dimension y

        Returns:
        A tuple of (x, y) indicating the starting coordinates if a submesh is found.
        Otherwise None.
        """
        if a > self.cluster.dimx or b > self.cluster.dimy:
            logging.debug(f"Job shape ({a}, {b}) exceeds cluster dimension.")
            return None

        # The search range for torus should consider wrap-around.
        if self.cluster.topo in (TopoType.T2D,):
            xrange = self.cluster.dimx
            yrange = self.cluster.dimy
        else:
            xrange = self.cluster.dimx - a + 1
            yrange = self.cluster.dimy - b + 1
        for x in range(xrange):
            for y in range(yrange):
                # Check if all nodes in the submesh have available XPUs.
                if all(
                    array[(x + i) % self.cluster.dimx][(y + j) % self.cluster.dimy] > 0
                    for i in range(a)
                    for j in range(b)
                ):
                    return x, y
        return None

    def _find_slice(
        self, array: NDArray[np.float64], a: int, b: int, c: int
    ) -> Optional[tuple[int, int, int]]:
        """
        Finds a feasible AxBxC shaped submesh in the 3D mesh/torus.

        Parameters:
        array: 3D numpy array, where 0 means no XPU available.
        a: job shape along dimension x
        b: job shape along dimension y
        c: job shape along dimension z

        Returns:
        A tuple of (x, y, z) indicating the starting coordinates if a slice is found.
        Otherwise None.
        """
        if a > self.cluster.dimx or b > self.cluster.dimy or c > self.cluster.dimz:
            logging.debug(f"Job shape ({a}, {b}, {c}) exceeds cluster dimension.")
            return None

        # The search range for torus should consider wrap-around.
        if self.cluster.topo in (TopoType.T3D_NT, TopoType.T3D_T):
            xrange = self.cluster.dimx
            yrange = self.cluster.dimy
            zrange = self.cluster.dimz
        else:
            xrange = self.cluster.dimx - a + 1
            yrange = self.cluster.dimy - b + 1
            zrange = self.cluster.dimz - c + 1
        for x in range(xrange):
            for y in range(yrange):
                for z in range(zrange):
                    # Check if all nodes in the slice have available XPUs.
                    if all(
                        array[(x + i) % self.cluster.dimx][(y + j) % self.cluster.dimy][
                            (z + k) % self.cluster.dimz
                        ]
                        > 0
                        for i in range(a)
                        for j in range(b)
                        for k in range(c)
                    ):
                        return x, y, z
        return None

    def _fit_2d(
        self, avail_array: NDArray[np.float64], job: Job
    ) -> Tuple[SchedDecision, Job]:
        """
        Fits a 2D job into the cluster.
        Performs transposition if the original placement fails.
        """
        a, b = job.shape
        loc = self._find_submesh(avail_array, a, b)
        loc_t = self._find_submesh(avail_array, b, a)
        if not loc and not loc_t:
            logging.debug(f"Job {job.uuid} rejected, no feasible placement found.")
            return SchedDecision.REJECT, job
        # If the placement is after transposition, update the job shape.
        base_x, base_y = loc if loc else loc_t
        a, b = (a, b) if loc else (b, a)
        job.shape = (a, b)
        for i in range(base_x, base_x + a):
            for j in range(base_y, base_y + b):
                # Make sure to handle wrap-around indices for torus.
                job.allocation[f"x{i % self.cluster.dimx}-y{j % self.cluster.dimy}"] = 1
        return SchedDecision.ADMIT, job

    def _fit_3d(
        self, avail_array: NDArray[np.float64], job: Job
    ) -> Tuple[SchedDecision, Job]:
        """
        Fits a 3D job into the cluster.
        Performs transposition if the original placement fails.
        """
        shapes = list(permutations(job.shape))
        # A shape should have 6 permutations in 3D space. For now, they are treated as equivalent.
        # But ideally, the best shape should be the one that generates least fragmentation.
        # Need some form of evaluation to determine the best shape.
        for a, b, c in shapes:
            loc = self._find_slice(avail_array, a, b, c)
            if not loc:
                continue
            base_x, base_y, base_z = loc
            # Update the job shape since the placement could be after transposition.
            job.shape = (a, b, c)
            for i in range(base_x, base_x + a):
                for j in range(base_y, base_y + b):
                    for k in range(base_z, base_z + c):
                        # Make sure to handle wrap-around indices for torus.
                        job.allocation[
                            f"x{i % self.cluster.dimx}-"
                            f"y{j % self.cluster.dimy}-"
                            f"z{k % self.cluster.dimz}"
                        ] = 1
            return SchedDecision.ADMIT, job

        logging.debug(f"Job {job.uuid} rejected, no feasible placement found.")
        return SchedDecision.REJECT, job

    def _check_total_xpu(self, job: Job) -> bool:
        """
        Check if the total number of XPUs required by the job is available.
        If not, return False. Otherwise, return True.
        """
        return job.size <= self.cluster.totalIdleXPU()

    def _check_total_node(self, job: Job) -> bool:
        """
        Check if the total number of nodes required by the job is available.
        If not, return False. Otherwise, return True.
        """
        req_node_cnt = len(job.shape) if job.topology == TopoType.CLOS else job.size
        return req_node_cnt <= self.cluster.totalIdleNodes()

    def _firstfit(self, job: Job) -> Tuple[SchedDecision, Job]:
        # Jobs that fail the total XPU check get rejected.
        if not self._check_total_xpu(job):
            logging.debug(f"Job {job.uuid} rejected, insufficient total number of XPUs.")
            return SchedDecision.REJECT, job
        if not self._check_total_node(job):
            logging.debug(f"Job {job.uuid} rejected, insufficient number of idle nodes.")
            return SchedDecision.REJECT, job

        # TODO: generalize to other topologies.
        if job.topology in (TopoType.CLOS,):
            logging.debug(f"Job {job.uuid} rejected, unsupported topology.")
            return SchedDecision.REJECT, job

        if job.topology in (TopoType.MESH2D, TopoType.T2D):
            return self._fit_2d(self.cluster.toArray(), job)
        elif job.topology in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
            return self._fit_3d(self.cluster.toArray(), job)

    def place(self, job: Job, policy: str = SCHED_POLICY) -> Tuple[SchedDecision, Job]:
        """
        Make a scheduling decision for a job. Note that the job (e.g., shape, duration)
        could be modifed to achieve a more desirable scheduling decision. But if the job
        is rejected, it is not modified.
        The (modified) job is returned along with the decision.
        """
        if job.topology != self.cluster.topo:
            raise ValueError("Job topology mismatches cluster topology.")

        if policy == "firstfit":
            return self._firstfit(job)
        elif policy == "simplefit":
            pass
        # The default fallback is to reject all jobs.
        return SchedDecision.REJECT, job
