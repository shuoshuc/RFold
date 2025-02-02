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
from common.job import TopoType
from common.utils import PrettyForm, spec_parser
from Cluster.cluster import Cluster
from Cluster.model_builder import build
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay

# C3 model is pretty large, so avoid loading from a model file.
# Just directly generate it when needed.
C3_MODEL = build(
    topo=TopoType.T3D_NT,
    name="c3",
    dimension=(16, 16, 16),
    xpu_per_node=1,
    port_per_node=6,
    port_speed_gbps=800,
)


def main():
    env = simpy.Environment()

    # Either load model from a file or directly generate it.
    # model = spec_parser(MODEL_FILE)
    model = C3_MODEL

    # Initialize the cluster.
    cluster = Cluster(env, spec=model)
    # Initialize the cluster manager.
    mgr = ClusterManager(env, cluster=cluster)
    # Spin up the workload generator.
    trace = TraceReplay(env, tracefile=TRACE_NAME, cluster_mgr=mgr)
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
    env.process(workload.run(stop_time=SIM_DURATION_SEC))
    env.run()
    logging.info("Simulation completes")
    mgr.sweepAllQueues()

    logging.info("----[Summary]-----")
    mgr.job_stats = dict(sorted(mgr.job_stats.items()))
    for job in mgr.job_stats.values():
        logging.info(f"{job.stats()}")


if __name__ == "__main__":
    lvl = getattr(logging, LOG_LEVEL.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid log level: {lvl}")
    # Configure the root logger.
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = PrettyForm(
        fmt="{module: <40} {message}",
        style="{",
    )
    logger.setLevel(lvl)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    main()
