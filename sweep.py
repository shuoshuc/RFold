#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import subprocess
import sys
import time
from itertools import product


def run_process(i, tot, cmd):
    run_dir = f"run{i}"

    start = time.time()
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    result = subprocess.run(cmd + f"--stats_outdir {run_dir}", shell=True)
    end = time.time()

    if result.returncode != 0:
        print(f"[ERROR] run{i} failed with return code {result.returncode}", flush=True)
    print(
        f"run {i}/{tot} took {round((end - start) / 60, 0)} min: {cmd}",
        flush=True,
    )


def main(trace_folder: str):
    # Only import flags in the function scope. common.flags imports argparse, which
    # messes with sys.argv when loaded. Hence, delay importing until args are consumed.
    from common.flags import (
        PHILLY_TRACE,
        ALIBABA_TRACE,
        HELIOS_TRACE,
        ACME_TRACE,
        IAT_DIST,
    )

    # All the parameters to sweep over.
    sim_duration = [50 * 3600, 100 * 3600]
    dimensions = ["16,16,16", "24,24,24", "32,32,32"]
    if trace_folder:
        # Include all csv files in the trace folder.
        trace = [
            os.path.join(root, file)
            for root, _, files in os.walk(trace_folder)
            for file in files
            if file.endswith(".csv")
        ]
    else:
        trace = [PHILLY_TRACE, ALIBABA_TRACE, HELIOS_TRACE, ACME_TRACE]
    iat = [IAT_DIST]
    sched_policy = ["firstfit", "slurm_hilbert"]

    start_time = time.time()
    configs = list(product(sim_duration, dimensions, trace, iat, sched_policy))
    cmds = []
    for args in configs:
        sim_dur, dim, trace_file, iat_file, policy = args
        cmd = (
            f"python3 launch.py -t {sim_dur} --dim {dim} --sched_policy {policy} "
            f"--log_level WARNING "
        )
        if trace_folder:
            cmd += f"-r {trace_file} "
        else:
            cmd += f"--dur_trace_file {trace_file} --iat_file {iat_file} "
        cmds.append(cmd)

    # Reserve 2 cores for the system to remain responsive.
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        pool.starmap(
            run_process, [(i + 1, len(configs), cmd) for i, cmd in enumerate(cmds)]
        )
    end_time = time.time()
    print(f"Total execution time: {round((end_time - start_time) / 60, 0)} min")


if __name__ == "__main__":
    trace_folder = None
    if len(sys.argv) < 2:
        print(
            "No trace folder provided, disable trace replay. "
            "To run replay: python sweep.py <trace_folder>"
        )
    else:
        trace_folder = sys.argv[1]
        # Drop the command line arguments to avoid argparse error.
        sys.argv = sys.argv[:1]
    main(trace_folder)
