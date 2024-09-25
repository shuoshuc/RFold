#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
from scipy.stats import rv_discrete

from job import TopoType, Job
from simpleUUID import SimpleUUID

TPU_JOB_SIZES_DIST = 'data/tpu_job_size.txt'
TPU_ARRIVAL_TIME_DIST = 'data/tpu_arrival_time.txt'

class WorkloadGenerator:
    '''
    A workload generator class that fits the given probability distribution.
    It can generate workloads following the given distribution.
    '''
    def __init__(self, job_size_file, arrival_time_file):
        if not job_size_file or not arrival_time_file:
            raise RuntimeError('Distribution files are not provided')
        # Job size is modeled by `self.rv_size`.
        # Job inter-arrival time is modeled by `self.rv_iat`, in seconds.
        self.abs_time_sec = 0
        self._loadSizeDist(job_size_file)
        self._loadIATDist(arrival_time_file)
        self.uuidgen = SimpleUUID()

    def _loadSizeDist(self, filename):
        '''
        Loads the given job size csv file and parses it into a distribution (PDF).
        '''
        self.dist_size = []
        self.jobs = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # Skips the comment line.
                if line.startswith('#'):
                    continue
                jid, slice_shape, topo_type, job_size, p = line.strip().split(',')
                # Random variable only needs a list of values and their probabilities.
                # Hence, the specific job info is tracked by the `self.jobs` dict.
                self.dist_size.append([int(jid), float(p) / 100])
                self.jobs[int(jid)] = Job(uuid=0, topology=TopoType[topo_type],
                                          shape=tuple(map(int, slice_shape.split('x'))),
                                          size=int(job_size))
        slice_ids, probs = zip(*self.dist_size)
        # If the distribution does not add up to 1, normalize it.
        # E.g., this happens in Table 2 of Google's TPUv4 paper (ISCA'23).
        if sum(probs) != 1:
            probs = np.array(probs) / sum(probs)
        self.rv_size = rv_discrete(values=(slice_ids, probs))

    def _loadIATDist(self, filename):
        '''
        Loads the given job inter-arrival time (IAT) csv file and
        parses it into a distribution (PDF).
        '''
        self.dist_iat = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # Skips the comment line.
                if line.startswith('#'):
                    continue
                iat_sec, p = map(float, line.strip().split(','))
                new_prob = self.dist_iat.setdefault(iat_sec, 0) + p
                self.dist_iat[iat_sec] = new_prob
        iats, probs = zip(*self.dist_iat.items())
        # If the distribution does not add up to 1, normalize it.
        if sum(probs) != 1:
            probs = np.array(probs) / sum(probs)
        self.rv_iat = rv_discrete(values=(iats, probs))

    def run(self, num_samples=1):
        '''
        Generates one or more jobs. Returns a list of (IAT, Job) tuples.
        '''
        jobs = []
        iat_samples = self.rv_iat.rvs(size=num_samples).tolist()
        for iat in iat_samples:
            j = self.rv_size.rvs(size=1)[0]
            new_job = copy.deepcopy(self.jobs[j])
            new_job.uuid = self.uuidgen.fetch()
            new_job.arrival_time_sec = self.abs_time_sec
            self.abs_time_sec += iat
            jobs.append(new_job)
        return jobs

def main():
    size_dist, arrival_time = TPU_JOB_SIZES_DIST, TPU_ARRIVAL_TIME_DIST
    wgen = WorkloadGenerator(size_dist, arrival_time)
    samples = wgen.run(10)
    for sample in samples:
        print(sample)
    print(wgen.run())

if __name__ == "__main__":
    main()