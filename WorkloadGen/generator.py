import copy
import logging
import math
import numpy as np
import random
import scipy.stats as stats
import simpy
from collections.abc import Iterator
from io import StringIO
from scipy.stats import rv_discrete
from typing import Union

from common.flags import FLAGS
from common.job import TopoType, Job, SplitShape
from common.simpleUUID import SimpleUUID
from common.utils import extract_duration, factorize2, factorize3
from ClusterManager.manager import ClusterManager


class WorkloadGenerator:
    """
    A workload generator class that fits the given probability distribution.
    It can generate workloads following the given distribution.
    """

    def __init__(
        self,
        env: simpy.core.Environment,
        arrival_time_file: Union[str, StringIO],
        job_size_file: Union[str, StringIO],
        cluster_mgr: ClusterManager,
        dur_trace: str,
    ):
        self.env = env
        self.cluster_mgr = cluster_mgr
        if not arrival_time_file or not job_size_file:
            raise RuntimeError("Distribution files are not provided")
        # Job inter-arrival time is modeled by `self.rv_iat`, in seconds.
        # Job size is modeled by `self.rv_size`.
        self.abs_time_sec = 0
        self._loadIATDist(arrival_time_file)
        self._loadSizeDist(job_size_file)
        self.uuidgen = SimpleUUID()
        self.cached_duration = extract_duration(dur_trace)
        random.seed(42)

    def _loadSizeDist(self, filename: Union[str, StringIO]):
        """
        Loads the given job size csv file (or a StringIO equivalent) and
        parses it into a distribution (PDF).

        Assumption: The job size distribution can contain duplicates, i.e.,
        multiple entries with the same (topology, shape, size, duration) tuple.
        During parsing, the duplicates are de-duplicated and the probabilities are summed up.
        """
        self.dist_size = []
        accumulated_prob = {}
        self.jobs = {}

        is_file = not isinstance(filename, StringIO)
        f = filename
        if is_file:
            f = open(filename, "r", encoding="utf-8")
        for line in f.readlines():
            # Skips the comment line.
            if line.startswith("#"):
                continue
            _, topo_type, slice_shape, job_size, duration, p = line.strip().split(",")
            job_key = (TopoType[topo_type], slice_shape, job_size, float(duration))
            new_prob = accumulated_prob.setdefault(job_key, 0) + float(p)
            accumulated_prob[job_key] = new_prob
        if is_file:
            f.close()
        for i, (job_key, p) in enumerate(accumulated_prob.items()):
            topology, slice_shape, job_size, duration = job_key
            # Random variable only needs a list of values and their probabilities.
            # Hence, the specific job info is tracked by the `self.jobs` dict.
            self.dist_size.append([i, float(p) / 100])
            shape_tup = SplitShape(slice_shape, topology)
            self.jobs[i] = Job(
                uuid=0,
                arrival_time_sec=0,
                topology=topology,
                shape=shape_tup,
                # TODO: handle the case when job size is a fraction.
                size=int(job_size),
                duration_sec=duration,
            )
        slice_ids, probs = zip(*self.dist_size)
        # If the distribution does not add up to 1, normalize it.
        # E.g., this happens in Table 2 of Google's TPUv4 paper (ISCA'23).
        if sum(probs) != 1:
            probs = np.array(probs) / sum(probs)
        self.rv_size = rv_discrete(values=(slice_ids, probs))

    def _loadIATDist(self, filename: Union[str, StringIO]):
        """
        Loads the given job inter-arrival time (IAT) csv file (or a StringIO equivalent)
        and parses it into a distribution (PDF).
        """
        self.dist_iat = {}

        is_file = not isinstance(filename, StringIO)
        f = filename
        if is_file:
            f = open(filename, "r", encoding="utf-8")
        for line in f.readlines():
            # Skips the comment line.
            if line.startswith("#"):
                continue
            iat_sec, p = map(float, line.strip().split(","))
            new_prob = self.dist_iat.setdefault(iat_sec, 0) + p
            self.dist_iat[iat_sec] = new_prob
        if is_file:
            f.close()
        iats, probs = zip(*self.dist_iat.items())
        # If the distribution does not add up to 1, normalize it.
        if sum(probs) != 1:
            probs = np.array(probs) / sum(probs)
        self.rv_iat = rv_discrete(values=(iats, probs))

    def run(self, time_mark: float = None) -> Iterator[simpy.events.Timeout]:
        """
        Generates jobs indefinitely and enqueues them.

        Parameters:
            time_mark: jobs generated before this mark must complete.
        """
        while True:
            iat = float(round(self.rv_iat.rvs(size=1)[0]))
            j = self.rv_size.rvs(size=1)[0]
            new_job = copy.deepcopy(self.jobs[j])
            if max(new_job.shape) > max(FLAGS.dim):
                continue
            new_job.uuid = self.uuidgen.fetch()
            new_job.arrival_time_sec = self.abs_time_sec
            # Some job distributions have no info about duration.
            # To handle this, use duration from another trace.
            # Some traces (e.g., Philly) have zero-duration jobs, likely due to
            # logging granularity. Zero-duration jobs should be ignored.
            while not new_job.duration_sec:
                new_job.duration_sec = random.choice(self.cached_duration)
            # Conditionally ignore twisted torus.
            if not FLAGS.no_ignore_twist and new_job.topology == TopoType.T3D_T:
                new_job.topology = TopoType.T3D_NT

            yield self.env.timeout(new_job.arrival_time_sec - self.env.now)
            # If a stop time is provided, jobs generated prior to the stop time must all
            # complete. Otherwise, all jobs must complete.
            wait_to_complete = True
            if time_mark is not None and self.abs_time_sec > time_mark:
                wait_to_complete = False
            self.cluster_mgr.submitJob(new_job, wait_to_complete)

            self.abs_time_sec += iat


class MixedWorkload:
    """
    A mixed workload generator that combines 1D/2D/3D workloads.
    """

    def __init__(
        self,
        env: simpy.core.Environment,
        cluster_mgr: ClusterManager,
        ndim: int,
        rsize: int,
        arrival_time_file: Union[str, StringIO],
        job_size_file: Union[str, StringIO],
        dur_trace: str,
    ):
        self.env = env
        self.cluster_mgr = cluster_mgr
        self.ndim = ndim
        self.rsize = rsize
        if not arrival_time_file or not job_size_file:
            raise RuntimeError("Distribution files are not provided")
        # Job inter-arrival time is modeled by `self.rv_iat`, in seconds.
        self.abs_time_sec = 0
        self._loadIATDist(arrival_time_file)
        self.uuidgen = SimpleUUID()
        self.cached_duration = extract_duration(dur_trace)
        random.seed(42)

        # Cache all the valid job shapes.
        self.shapes = {}
        # Jobs of size 1, 2 are special.
        self.shapes[1] = [[(1, 1, 1)], [], []]
        self.shapes[2] = [[(2, 1, 1)], [], []]
        for s in range(4, 2049, 4):
            self.shapes[s] = [[], [], []]
            if s <= 256:
                # Valid to have 1D shape.
                self.shapes[s][0] = [(s, 1, 1)]
            for dim in [2, 3]:
                self.shapes[s][dim - 1] = self.all_shapes_for_size(s, dim)
            # If size `s` has no valid shapes, remove it from the dict.
            if all(len(shape_list) <= 0 for shape_list in self.shapes[s]):
                del self.shapes[s]

    def all_shapes_for_size(self, job_size: int, dim: int) -> list[tuple[int, ...]]:
        """
        For a given job size and dimension, generates all valid shapes.
        """
        if dim not in [2, 3]:
            raise ValueError(f"_generate_shapes(): invalid dimension {dim}")

        shapes = []
        limit = self.cluster_mgr.cluster.numNodes() // self.rsize**self.ndim * self.rsize
        factorize = factorize2 if dim == 2 else factorize3
        for tup in factorize(job_size):
            if not all(i % 2 == 0 for i in tup):
                continue
            if not all(j <= limit for j in tup):
                continue
            final_tup = tup if len(tup) == 3 else (*tup, 1)
            # TODO: handle 2D torus.
            if self.check_shape(final_tup):
                shapes.append(final_tup)
        return shapes

    def sample_job_size(self) -> int:
        """
        Generates a random job size.
        """
        # # scale is the inverse of the rate parameter.
        # # Divide by 4 to ensure at least 2 dimensions are even.
        # scale = target_size
        # lower = 1
        # upper = target_size

        # sample = int(
        #     stats.truncexpon.rvs(
        #         b=(upper - lower) / scale, loc=lower, scale=scale, size=1
        #     )[0]
        # )
        # job_size = sample if sample in (1, 2) else int(math.ceil(sample / 4) * 4)
        # return job_size
        return random.choice(list(self.shapes.keys()))

    def check_shape(self, shape: tuple[int, ...]) -> bool:
        blocks_needed, _, _ = self.cluster_mgr.scheduler._reconfig_param_map(
            shape, self.rsize
        )
        return sum(blocks_needed.values()) <= (
            self.cluster_mgr.cluster.numNodes() // self.rsize**self.ndim
        )

    # def generate_dimension(self, job_size: int) -> int:
    #     """
    #     Generates a dimension (1D/2D/3D) given a job size.
    #     Hint: small jobs are more likely to be 1D, medium jobs are likely to be 2D,
    #     large jobs are likely to be 3D.
    #     """
    #     if job_size <= 256:
    #         return int(np.random.choice([1, 2]))
    #     elif job_size <= 1024:
    #         return int(np.random.choice([2, 3]))
    #     else:
    #         return int(np.random.choice([2, 3]))

    def generate_shape(self, job_size: int) -> tuple[int, ...]:
        # dim = self.generate_dimension(job_size)
        idx_choices = []
        for i, shape_list in enumerate(self.shapes[job_size]):
            if len(shape_list) > 0:
                idx_choices.append(i)
        choice = int(np.random.choice(idx_choices))
        shape_idx = random.randint(0, len(self.shapes[job_size][choice]) - 1)
        return self.shapes[job_size][choice][shape_idx]

    def _loadIATDist(self, filename: Union[str, StringIO]):
        """
        Loads the given job inter-arrival time (IAT) csv file (or a StringIO equivalent)
        and parses it into a distribution (PDF).
        """
        self.dist_iat = {}

        is_file = not isinstance(filename, StringIO)
        f = filename
        if is_file:
            f = open(filename, "r", encoding="utf-8")
        for line in f.readlines():
            # Skips the comment line.
            if line.startswith("#"):
                continue
            iat_sec, p = map(float, line.strip().split(","))
            new_prob = self.dist_iat.setdefault(iat_sec, 0) + p
            self.dist_iat[iat_sec] = new_prob
        if is_file:
            f.close()
        iats, probs = zip(*self.dist_iat.items())
        # If the distribution does not add up to 1, normalize it.
        if sum(probs) != 1:
            probs = np.array(probs) / sum(probs)
        self.rv_iat = rv_discrete(values=(iats, probs))

    def run(self, time_mark: float = None) -> Iterator[simpy.events.Timeout]:
        """
        Generates jobs indefinitely and enqueues them.

        Parameters:
            time_mark: jobs generated before this mark must complete.
        """
        while True:
            iat = float(round(self.rv_iat.rvs(size=1)[0]))
            job_size = self.sample_job_size()
            job_shape = self.generate_shape(job_size)
            new_job = Job(
                uuid=self.uuidgen.fetch(),
                arrival_time_sec=self.abs_time_sec,
                topology=TopoType.T3D_NT if self.ndim == 3 else TopoType.T2D,
                shape=job_shape,
                size=job_size,
                duration_sec=0,
            )
            while not new_job.duration_sec:
                new_job.duration_sec = random.choice(self.cached_duration)

            yield self.env.timeout(new_job.arrival_time_sec - self.env.now)
            # If a stop time is provided, jobs generated prior to the stop time must all
            # complete. Otherwise, all jobs must complete.
            wait_to_complete = True
            if time_mark is not None and self.abs_time_sec > time_mark:
                wait_to_complete = False
            self.cluster_mgr.submitJob(new_job, wait_to_complete)
            self.abs_time_sec += iat
