#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2024 Carnegie Mellon University
Author: Shawn Chen <shuoshuc@cs.cmu.edu>

This script is the entry point of the simulation framework.
"""


import simpy
import logging

from common.flags import FLAGS
from common.job import TopoType
from common.utils import PrettyForm, spec_parser, job_stats_to_trace, dump_stats
from Cluster.cluster import Cluster
from Cluster.model_builder import build, build_torus
from ClusterManager.manager import ClusterManager
from WorkloadGen.generator import WorkloadGenerator
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
    # Initialize the cluster manager.
    mgr = ClusterManager(env, cluster=cluster)
    # Spin up the workload generator. If a trace is provided, replay the trace.
    # Otherwise, generate a new workload.
    if FLAGS.replay_trace:
        workload = TraceReplay(env, tracefile=FLAGS.replay_trace, cluster_mgr=mgr)
    else:
        workload = WorkloadGenerator(
            env,
            arrival_time_file=FLAGS.iat_file,
            job_size_file=FLAGS.job_size_file,
            cluster_mgr=mgr,
            dur_trace=FLAGS.dur_trace_file,
        )

    # Start simulation.
    logging.info("Simulation starts")
    env.process(mgr.schedule())
    env.process(workload.run(stop_time=FLAGS.sim_sec))
    # Run the simulation until the specified time.
    # Note: this will leave some jobs incomplete. To complete all jobs, run until the end
    # of the trace by setting until=None. However, the draining period would have
    # different properties since there is no new job arriving.
    env.run(until=FLAGS.sim_sec)
    logging.info("Simulation completes")
    mgr.sweepAllQueues()

    logging.info("----[Summary]-----")
    mgr.job_stats = dict(sorted(mgr.job_stats.items()))
    for job in mgr.job_stats.values():
        logging.info(f"{job.stats()}")
    # Dump the trace generated in runtime to a file.
    if not FLAGS.replay_trace and FLAGS.trace_output:
        job_stats_to_trace(mgr.job_stats, FLAGS.trace_output)
    # Dump the stats to a file.
    if FLAGS.stats_output:
        dump_stats(mgr.job_stats, FLAGS.stats_output)


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
