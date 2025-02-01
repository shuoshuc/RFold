#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import simpy
import logging

from common.flags import *
from common.job import TopoType
from common.utils import PrettyForm, viz3D, job_stats_to_trace
from Cluster.cluster import Cluster
from Cluster.model_builder import build
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator
from WorkloadGen.trace import TraceReplay


def main():
    env = simpy.Environment()

    # Initialize the cluster.
    model = build(
        topo=TopoType.T3D_NT,
        name="c3",
        dimension=(16, 16, 16),
        xpu_per_node=1,
        port_per_node=6,
        port_speed_gbps=800,
    )
    cluster = Cluster(env, spec=model)
    # Initialize the cluster manager.
    mgr = ClusterManager(env, cluster=cluster)
    workload = WorkloadGenerator(
        env,
        arrival_time_file=TPU_ARRIVAL_TIME_DIST,
        job_size_file=TPU_JOB_SIZES_DIST,
        cluster_mgr=mgr,
        dur_trace=PHILLY_TRACE,
    )

    # Start simulation.
    logging.info("Simulation starts")
    env.process(mgr.schedule())
    # Simulate a limited duration workload.
    env.process(workload.run(stop_time=SIM_DURATION_SEC))
    env.run(until=SIM_DURATION_SEC)
    logging.info("Simulation completes")
    mgr.sweepAllQueues()

    logging.info("----[Summary]-----")
    mgr.job_stats = dict(sorted(mgr.job_stats.items()))
    for job in mgr.job_stats.values():
        logging.info(f"{job.stats()}")

    job_stats_to_trace(mgr.job_stats, "test_trace.csv")


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
