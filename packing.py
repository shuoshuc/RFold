#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import random
import simpy
import time
import logging
import numpy as np
import scipy.stats as stats

from common.flags import FLAGS
from common.job import TopoType, Job, FormShape, SplitShape
from common.simpleUUID import SimpleUUID
from common.utils import PrettyForm, factorize
from Cluster.cluster import Cluster
from Cluster.model_builder import build, build_torus
from ClusterManager.scheduling import SchedulingPolicy, SchedDecision

uuidgen = SimpleUUID()


def sample_job_size(target_size: int) -> int:
    """
    Generates a random job size.
    """
    # scale is the inverse of the rate parameter.
    # Divide by 4 to ensure at least 2 dimensions are even.
    scale = target_size
    lower = 1
    upper = target_size

    sample = int(
        stats.truncexpon.rvs(b=(upper - lower) / scale, loc=lower, scale=scale, size=1)[0]
    )
    # job_size = sample if sample in (1, 2) else sample // 4 * 4
    job_size = sample if sample in (1, 2) else int(math.ceil(sample / 4) * 4)
    return job_size


def job_size_to_shape2d(job_size: int) -> tuple[int, int]:
    """
    Generates a job with a shape given the size.
    """
    # By default, a job is 1D.
    shape_tup = (job_size, 1)
    # Fast track return.
    if job_size == 1:
        return shape_tup

    # Job size > 128, must be 2D.
    if job_size > 128:
        job_dim = 2
    # The rest can be any dimension.
    else:
        job_dim = int(np.random.choice([1, 2]))
    x, y, _ = factorize(job_size, job_dim)

    return (x, y)


def job_size_to_shape3d(job_size: int) -> tuple[int, int, int]:
    """
    Generates a job with a shape given the size.
    """
    # By default, a job is 1D. We can change this to 2D or 3D if needed.
    shape_tup = (job_size, 1, 1)
    # Fast track return.
    if job_size == 1:
        return shape_tup

    # Job size > 1024 must be 3D.
    if job_size > 1024:
        job_dim = 3
    # Job size in (128, 1024) must be at least 2D.
    elif job_size > 128:
        job_dim = int(np.random.choice([2, 3]))
    # The rest can be any dimension.
    else:
        job_dim = int(np.random.choice([1, 2, 3]))
    x, y, z = factorize(job_size, job_dim)

    return (x, y, z)


def dump_trace(jobs: list[Job], output: str):
    if not jobs or not output:
        raise ValueError("Invalid input to dump.")

    out = []
    for job in jobs:
        out.append(
            [
                job.uuid,
                job.arrival_time_sec,
                job.topology.name,
                FormShape(job.shape, job.topology),
                job.size,
                job.duration_sec,
                job.reject_reason,
            ]
        )
    with open(output, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "#job id",
                "arrival time (sec)",
                "topology",
                "shape",
                "size",
                "duration (sec)",
                "reject reason",
            ]
        )
        writer.writerows(out)


def pack():
    env = simpy.Environment()

    # Initialize the cluster.
    logging.info(f"Product of FLAGS.dim: {math.prod(FLAGS.dim)}")
    model = build_torus(name="c3", dimension=FLAGS.dim)
    cluster = Cluster(env, spec=model)
    # Initialize the scheduler.
    sched = SchedulingPolicy(env, cluster)
    topo_type = TopoType.T3D_NT if len(FLAGS.dim) == 3 else TopoType.T2D
    shape_tup = (1, 1, 1) if len(FLAGS.dim) == 3 else (1, 1)
    job_size_to_shape = (
        job_size_to_shape3d if len(FLAGS.dim) == 3 else job_size_to_shape2d
    )

    idle = cluster.totalIdleXPU()
    jobs = []
    logging.info("Simulation starts")
    while idle > 0:
        # Simple job generator.
        job = Job(
            uuid=uuidgen.fetch(),
            arrival_time_sec=0,
            topology=topo_type,
            shape=shape_tup,
            size=1,
            duration_sec=10000,
        )
        job.size = sample_job_size(target_size=math.prod(FLAGS.dim) // 2)
        job.shape = job_size_to_shape(job.size)

        # Place the job.
        decision, job_to_sched = sched.place(job)
        if decision == SchedDecision.ADMIT:
            # Update the resource usage.
            cluster.execute(job_to_sched)
        elif decision == SchedDecision.REJECT:
            if job.size > cluster.totalIdleXPU():
                job.reject_reason = "resource"
            else:
                job.reject_reason = "shape"

        idle -= job.size
        jobs.append(job)
    logging.info("Simulation completes")
    logging.info("----[Summary]-----")
    logging.info(f"total jobs: {len(jobs)}, size: {sum([job.size for job in jobs])}")
    logging.info(
        f"cluster util: {math.prod(FLAGS.dim) - cluster.totalIdleXPU()} / {math.prod(FLAGS.dim)}"
    )

    return jobs


def main():
    RUNS = 500
    for i in range(1, RUNS + 1):
        jobs = pack()
        # Dump the trace to a file.
        if FLAGS.stats_outdir:
            dump_trace(jobs, os.path.join(FLAGS.stats_outdir, f"trace{i}.csv"))


def read_trace(input_path: str) -> dict[str, list[Job]]:
    if not os.path.isdir(input_path):
        raise ValueError(f"{input_path} is not a valid directory")

    jobs = {}
    for file_name in os.listdir(input_path):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(input_path, file_name)
        with open(file_path, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if "#job id" in row[0]:
                    continue
                jobs.setdefault(file_name, []).append(
                    Job(
                        uuid=int(row[0]),
                        arrival_time_sec=float(row[1]),
                        topology=TopoType[row[2]],
                        shape=SplitShape(row[3], TopoType[row[2]]),
                        size=int(row[4]),
                        duration_sec=float(row[5]),
                    )
                )
    return jobs


def replay(input_path: str):
    job_map = read_trace(input_path)

    cluster_util = []
    for trace_name, jobs in job_map.items():
        logging.info(f"Processing trace: {trace_name}")
        env = simpy.Environment()
        # Initialize the cluster.
        logging.info(f"Product of FLAGS.dim: {math.prod(FLAGS.dim)}")
        model = build_torus(name="c3", dimension=FLAGS.dim)
        cluster = Cluster(env, spec=model)
        # Initialize the scheduler.
        sched = SchedulingPolicy(env, cluster)

        logging.info("Simulation starts")
        # Place the job.
        for job in jobs:
            decision, job_to_sched = sched.place(job)
            if decision == SchedDecision.ADMIT:
                # Update the resource usage.
                cluster.execute(job_to_sched)
            elif decision == SchedDecision.REJECT:
                if job.size > cluster.totalIdleXPU():
                    job.reject_reason = "resource"
                else:
                    job.reject_reason = "shape"
        logging.info("Simulation completes")
        logging.info("[Summary]")
        logging.info(f"total jobs: {len(jobs)}, size: {sum([job.size for job in jobs])}")
        logging.info(
            f"cluster util: {math.prod(FLAGS.dim) - cluster.totalIdleXPU()} / {math.prod(FLAGS.dim)}"
        )
        cluster_util.append(
            (math.prod(FLAGS.dim) - cluster.totalIdleXPU()) / math.prod(FLAGS.dim) * 100
        )
        logging.info("----[end]-----")
    if FLAGS.stats_outdir:
        with open(os.path.join(FLAGS.stats_outdir, "util.csv"), mode="w") as file:
            writer = csv.writer(file)
            for u in cluster_util:
                writer.writerow([u])


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
    random.seed(time.time())
    # main()
    # Overriding the flag definition, in this script it is a directory.
    replay(FLAGS.replay_trace)
