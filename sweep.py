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


def run_process(i, args):
    sim_dur, dim, dur_file, iat_file = args
    run_dir = f"run{i}"

    start = time.time()
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    cmd = (
        f"python3 launch.py -t {sim_dur} --dim {dim} --dur_trace_file {STRMAP[dur_file]} "
        f"--iat_file={STRMAP[iat_file]} --stats_output {run_dir}/stats.csv "
        f"--trace_output {run_dir}/trace.csv --log_level WARNING"
    )
    result = subprocess.run(cmd, shell=True)
    end = time.time()

    if result.returncode != 0:
        print(f"[ERROR] run{i} failed with return code {result.returncode}")
    print(
        f"run{i} took {round((end - start) / 60, 0)} min: sim_dur={sim_dur}, dim={dim}, "
        f"dur_file={dur_file}, iat={iat_file}"
    )


def main():
    processes = []
    # All the parameters to sweep over.
    sim_duration = [1800, 3600]
    dimensions = ["16,16,16", "32,32,32"]
    duration_trace = ["philly"]
    iat_distribution = ["iat"]

    start_time = time.time()
    for i, args in enumerate(
        list(product(sim_duration, dimensions, duration_trace, iat_distribution))
    ):
        p = multiprocessing.Process(
            target=run_process,
            args=(i, args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    end_time = time.time()
    print(f"Total execution time: {round((end_time - start_time) / 60, 0)} min")


if __name__ == "__main__":
    main()
