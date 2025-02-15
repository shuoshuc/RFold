#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import os
import subprocess
from itertools import product

from common.flags import *
import time

# Short-hand mapping to the files.
STRMAP = {
    "philly": PHILLY_TRACE,
    "iat": IAT_DIST,
}


def run_process(i, tot, args):
    sim_dur, dim, dur_file, iat_file, policy = args
    run_dir = f"run{i}"

    start = time.time()
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    cmd = (
        f"python3 launch.py -t {sim_dur} --dim {dim} --dur_trace_file {STRMAP[dur_file]} "
        f"--iat_file={STRMAP[iat_file]} --sched_policy {policy} --stats_outdir {run_dir} "
        f"--log_level WARNING"
    )
    result = subprocess.run(cmd, shell=True)
    end = time.time()

    if result.returncode != 0:
        print(f"[ERROR] run{i} failed with return code {result.returncode}")
    print(
        f"run {i}/{tot} took {round((end - start) / 60, 0)} min: sim_dur={sim_dur}, dim={dim}, "
        f"dur_file={dur_file}, iat={iat_file}, sched_policy={policy}"
    )


def main():
    processes = []
    # All the parameters to sweep over.
    sim_duration = [1800, 3600]
    dimensions = ["16,16,16", "32,32,32"]
    duration_trace = ["philly"]
    iat_distribution = ["iat"]
    sched_policy = ["firstfit", "slurm_hilbert"]

    start_time = time.time()
    configs = list(
        product(sim_duration, dimensions, duration_trace, iat_distribution, sched_policy)
    )
    for i, args in enumerate(configs):
        p = multiprocessing.Process(
            target=run_process,
            args=(i + 1, len(configs), args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    end_time = time.time()
    print(f"Total execution time: {round((end_time - start_time) / 60, 0)} min")


if __name__ == "__main__":
    main()
