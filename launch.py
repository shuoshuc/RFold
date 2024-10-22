#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2024 Carnegie Mellon University
Author: Shawn Chen <shuoshuc@cs.cmu.edu>

This script is the entry point of the simulation framework.
"""


import simpy
import logging

from common.flags import *
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay


def main():
    env = simpy.Environment()

    # Initialize the cluster manager.
    mgr = ClusterManager(env, cluster=None)
    # Spin up the workload generator.
    trace = TraceReplay(env, tracefile=ALIBABA_TRACE, cluster_mgr=mgr)
    if USE_TRACE:
        workload = trace
    else:
        csv_iat, csv_size = trace.exportDist()
        workload = WorkloadGenerator(
            env, arrival_time_file=csv_iat, job_size_file=csv_size, cluster_mgr=mgr
        )

    # Start simulation.
    logging.info("Simulation starts")
    env.process(mgr.schedule())
    env.process(workload.run())
    env.run(until=SIM_DURATION_SEC)
    logging.info("Simulation completes")


if __name__ == "__main__":
    lvl = getattr(logging, LOG_LEVEL.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid log level: {lvl}")
    # Configure the root logger.
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="{module}::{funcName}():{lineno}\t\t{message}", style="{"
    )
    logger.setLevel(lvl)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    main()
