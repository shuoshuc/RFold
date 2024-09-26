#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay

TPU_JOB_SIZES_DIST = 'WorkloadGen/data/tpu_job_size.txt'
TPU_ARRIVAL_TIME_DIST = 'WorkloadGen/data/tpu_arrival_time.txt'
PHILLY_TRACE = 'WorkloadGen/data/philly_trace.csv'


def main():
    size_dist, arrival_time = TPU_JOB_SIZES_DIST, TPU_ARRIVAL_TIME_DIST
    tpu_wgen = WorkloadGenerator(arrival_time, size_dist)
    samples = tpu_wgen.run(5)
    for i, sample in enumerate(samples):
        print(f'{i + 1} / {len(samples)} TPU job from dist.: {sample}')

    trace = TraceReplay(PHILLY_TRACE)
    jobs = trace.run(5)
    for i, job in enumerate(jobs):
        print(f'{i + 1} / {len(jobs)} Philly GPU job from trace: {job}')
    csv_iat, csv_size = trace.exportDist()
    philly_wgen = WorkloadGenerator(csv_iat, csv_size)
    samples = philly_wgen.run(5)
    for i, sample in enumerate(samples):
        print(f'{i + 1} / {len(samples)} Philly GPU job from dist.: {sample}')


if __name__ == "__main__":
    main()
