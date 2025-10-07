#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2024 Carnegie Mellon University
Author: Shawn Chen <shuoshuc@cs.cmu.edu>

This script is the entry point of the simulation framework.
"""


import os
import simpy
import logging

from common.flags import FLAGS
from common.job import TopoType
from common.utils import (
    PrettyForm,
    spec_parser,
    job_stats_to_trace,
    dump_job_stats,
    dump_cluster_stats,
)
from Cluster.cluster import Cluster
from Cluster.model_builder import build, build_torus
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator, MixedWorkload
from WorkloadGen.trace import TraceReplay


def main():
    env = simpy.Environment()

    # Load model from a file if specified or fall back to directly generating one.
    if FLAGS.model_file:
        model = spec_parser(FLAGS.model_file)
    else:
        model = build_torus(
            name="c3",
            dimension=FLAGS.dim,
        )

    # Initialize the cluster.
    cluster = Cluster(env, spec=model)
    # If a failure config is provided, read it and fail the nodes accordingly.
    if FLAGS.failure_config:
        with open(FLAGS.failure_config, "r") as f:
            failed_nodes = [line.strip() for line in f]
        logging.debug(f"Failing nodes: {failed_nodes}")
        cluster.failNodes(failed_nodes)
    # Initialize the cluster manager.
    mgr = ClusterManager(
        env,
        cluster=cluster,
        sim_njobs=FLAGS.sim_njobs,
        closed_loop_threshold=FLAGS.closed_loop_threshold,
    )
    # Spin up the workload generator. If a trace is provided, replay the trace.
    # Otherwise, generate a new workload.
    if FLAGS.replay_trace:
        workload = TraceReplay(env, tracefile=FLAGS.replay_trace, cluster_mgr=mgr)
    else:
        workload = MixedWorkload(
            env,
            cluster_mgr=mgr,
            ndim=len(FLAGS.dim),
            rsize=FLAGS.rsize,
            arrival_time_file=FLAGS.iat_file,
            job_size_file=FLAGS.job_size_file,
            dur_trace=FLAGS.dur_trace_file,
            desired_dim=-1,
            shape_multiple=FLAGS.shape_multiple,
        )

    # Start simulation.
    logging.info("Simulation starts")
    mgr_proc = env.process(mgr.schedule())
    env.process(workload.run())
    # Run the simulation until the manager process exits.
    # Note: this might leave some jobs incomplete.
    env.run(until=mgr_proc)
    logging.info("Simulation completes")
    # Flush jobs in all queues if this is not a replay, as we need incomplete jobs to
    # form a sustained workload.
    if not FLAGS.replay_trace:
        mgr.flushAllQueues()

    logging.info("----[Summary]-----")
    mgr.job_stats = dict(sorted(mgr.job_stats.items()))
    for job in mgr.job_stats.values():
        logging.info(f"{job.stats()}")
    # Dump the stats to a file.
    if FLAGS.stats_outdir:
        dump_job_stats(mgr.job_stats, os.path.join(FLAGS.stats_outdir, "job_stats.csv"))
        dump_cluster_stats(
            mgr.cluster_stats, os.path.join(FLAGS.stats_outdir, "cluster_stats.csv")
        )
        # Dump the trace (if generated in runtime) to a file.
        if not FLAGS.replay_trace:
            job_stats_to_trace(
                mgr.job_stats, os.path.join(FLAGS.stats_outdir, "trace.csv")
            )


if __name__ == "__main__":
    # Set up logging.
    lvl = getattr(logging, FLAGS.log_level.upper(), None)
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

    # Simulation logic.
    main()
