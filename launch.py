#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay

TPU_JOB_SIZES_DIST = 'WorkloadGen/data/tpu_job_size.csv'
TPU_ARRIVAL_TIME_DIST = 'WorkloadGen/data/tpu_arrival_time.csv'
PHILLY_TRACE = 'WorkloadGen/data/philly_trace.csv'
ALIBABA_TRACE = 'WorkloadGen/data/alibaba_v2020.csv'


def main():
    # TPU trace
    size_dist, arrival_time = TPU_JOB_SIZES_DIST, TPU_ARRIVAL_TIME_DIST
    tpu_wgen = WorkloadGenerator(arrival_time, size_dist)
    samples = tpu_wgen.run(5)
    for i, sample in enumerate(samples):
        print(f'{i + 1} / {len(samples)} TPU job from dist.: {sample}')

    print('----------------------------------------')

    # Philly trace
    trace = TraceReplay(PHILLY_TRACE)
    jobs = trace.run(5)
    for i, job in enumerate(jobs):
        print(f'{i + 1} / {len(jobs)} Philly GPU job from trace: {job}')
    csv_iat, csv_size = trace.exportDist()
    philly_wgen = WorkloadGenerator(csv_iat, csv_size)
    samples = philly_wgen.run(5)
    for i, sample in enumerate(samples):
        print(f'{i + 1} / {len(samples)} Philly GPU job from dist.: {sample}')

    print('----------------------------------------')

    # Alibaba trace
    alibaba_trace = TraceReplay(ALIBABA_TRACE)
    jobs = alibaba_trace.run(5)
    for i, job in enumerate(jobs):
        print(f'{i + 1} / {len(jobs)} Alibaba GPU job from trace: {job}')
    csv_iat, csv_size = alibaba_trace.exportDist()
    alibaba_wgen = WorkloadGenerator(csv_iat, csv_size)
    samples = alibaba_wgen.run(5)
    for i, sample in enumerate(samples):
        print(f'{i + 1} / {len(samples)} Alibaba GPU job from dist.: {sample}')


if __name__ == "__main__":
    main()
