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


def replay_helper(i, cmd):
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    end = time.time()

    if result.returncode != 0:
        print(f"[ERROR] run{i} failed with return code {result.returncode}", flush=True)
    print(
        f"run {i} took {round((end - start) / 60, 0)} min: {cmd}",
        flush=True,
    )


def gen_trace(runs):
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
    sim_dur = 10000 * 3600
    dim = "16,16,16"
    trace = PHILLY_TRACE
    clt = 1e6
    iat = IAT_DIST
    rsize = 4
    shape_multiple = 2
    policy = "reconfig"

    start_time = time.time()
    cmds = []
    for _ in range(runs):
        cmd = (
            f"python3 launch.py -t {sim_dur} --dim {dim} --place_policy {policy} "
            f"--rsize {rsize} -clt {clt} --dur_trace_file {trace} --iat_file {iat} "
            f"--shape_multiple {shape_multiple} "
            # "--log_level WARNING "
        )
        cmds.append(cmd)

    # Reserve 2 cores for the system to remain responsive.
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        pool.starmap(run_process, [(i + 1, runs, cmd) for i, cmd in enumerate(cmds)])
    end_time = time.time()
    print(f"Total execution time: {round((end_time - start_time) / 60, 0)} min")


def replay(trace_folder: str):
    sim_duration = [5000 * 3600]
    dimensions = ["16,16,16"]
    place_policy = ["rfold", "reconfig", "firstfit", "folding"]
    rsize = [4]
    trace_paths = [
        os.path.join(trace_folder, subfolder) for subfolder in os.listdir(trace_folder)
    ]

    start_time = time.time()
    configs = list(product(sim_duration, dimensions, trace_paths, place_policy, rsize))
    cmds = []
    for args in configs:
        sim_dur, dim, trace_path, policy, rsize = args
        trace = os.path.join(trace_path, "trace.csv")
        output = os.path.join(trace_path, policy)
        if not os.path.exists(output):
            os.makedirs(output)
        cmd = (
            f"python3 launch.py -t {sim_dur} --dim {dim} --place_policy {policy} "
            f"--rsize {rsize} -r {trace} --stats_outdir {output}"
        )
        failure_config = os.path.join(trace_path, "failed_nodes.csv")
        if os.path.isfile(failure_config):
            cmd += f" --failure_config {failure_config}"
        cmds.append(cmd)

    # Reserve 2 cores for the system to remain responsive.
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        pool.starmap(replay_helper, [(i + 1, cmd) for i, cmd in enumerate(cmds)])
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
    if not trace_folder:
        gen_trace(runs=100)
    else:
        replay(trace_folder)
