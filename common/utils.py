import csv
import json
import logging
import simpy
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from common.job import FormShape


class Signal:
    """
    A wrapper class for simpy events.
    """

    def __init__(self, env: simpy.core.Environment):
        self.env = env
        self.event = env.event()

    def trigger(self):
        """
        Trigger once and reloads the event.
        """
        self.event.succeed()
        self.event = self.env.event()

    def signal(self):
        """
        Expose the underlying event.
        """
        return self.event


class PrettyForm(logging.Formatter):
    """
    Custom log formatter to make padding and alignment easier.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        record.module = f"{record.module}::{record.funcName}():{record.lineno}"
        return super().format(record)


def spec_parser(specfile: str) -> dict:
    """
    Parse the cluster spec file.
    """
    with open(specfile, "r") as f:
        return json.load(f)


def dump_spec(spec: dict, specfile: str):
    """
    Dump the cluster spec to a file if `specfile` is specified.
    """
    if specfile:
        with open(specfile, "w") as f:
            json.dump(spec, f, indent=4)


def viz3D(dimx: int, dimy: int, dimz: int, array: NDArray[np.float64]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Generate coordinates for the corners of the cube
    x, y, z = np.meshgrid(np.arange(dimx), np.arange(dimy), np.arange(dimz))

    # Plot circles at each corner
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                alpha = 1 if array[i, j, k] > 0 else 0.2
                ax.scatter(i, j, k, s=100, alpha=alpha)

    # Connect circles with lines
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if i < dimx - 1:
                    ax.plot(
                        [x[i, j, k], x[i + 1, j, k]],
                        [y[i, j, k], y[i + 1, j, k]],
                        [z[i, j, k], z[i + 1, j, k]],
                        "k-",
                        color="gray",
                    )
                if j < dimy - 1:
                    ax.plot(
                        [x[i, j, k], x[i, j + 1, k]],
                        [y[i, j, k], y[i, j + 1, k]],
                        [z[i, j, k], z[i, j + 1, k]],
                        "k-",
                        color="gray",
                    )
                if k < dimz - 1:
                    ax.plot(
                        [x[i, j, k], x[i, j, k + 1]],
                        [y[i, j, k], y[i, j, k + 1]],
                        [z[i, j, k], z[i, j, k + 1]],
                        "k-",
                        color="gray",
                    )

    # Set the aspect of the plot to be equal
    ax.set_box_aspect([1, 1, 0.9])
    plt.show()


def factorize2(x):
    """
    Generates all (m, n) such that m * n = x.
    """
    pairs = set()
    for m in range(2, int(np.sqrt(x)) + 1):
        if x % m == 0:
            n = x // m
            pairs.add(tuple(sorted((m, n))))
    return pairs


def factorize3(N):
    """
    Generates all (x, y, z) such that x * y * z = N.
    """
    triplets = set()

    x = 2
    while x**3 <= N:
        if N % x == 0:
            residual = N // x
            # Start y from x to ensure x <= y
            y = x
            while y**2 <= residual:
                if residual % y == 0:
                    triplets.add((x, y, residual // y))
                y += 1
        x += 1
    return triplets


def extract_duration(csv_path):
    durations = []
    with open(csv_path, mode="r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Skips the comment line.
            if row[0].startswith("#"):
                continue
            # Last column must be job duration in seconds.
            duration = float(row[-1])
            durations.append(duration)
    return durations


def job_stats_to_trace(stats: dict, trace_output: str):
    """
    Convert job stats exported from ClusterManager to a trace file.
    """
    if not stats or not trace_output:
        raise ValueError("Invalid input to output to trace.")

    trace = []
    for job in stats.values():
        trace.append(
            [
                job.uuid,
                job.arrival_time_sec,
                job.topology.name,
                FormShape(job.shape, job.topology),
                job.size,
                job.duration_sec,
            ]
        )

    with open(trace_output, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "#job id",
                "arrival time (sec)",
                "topology",
                "shape",
                "size",
                "duration (sec)",
            ]
        )
        writer.writerows(trace)


def dump_job_stats(stats: dict, stats_output: str):
    """
    Convert job stats exported from ClusterManager to a csv file.
    """
    if not stats or not stats_output:
        raise ValueError("Invalid input to output to trace.")

    out = []
    for job in stats.values():
        if job.queueing_delay_sec is None:
            continue
        out.append(
            [
                job.uuid,
                job.arrival_time_sec,
                job.sched_time_sec,
                job.completion_time_sec,
                job.size,
                job.queueing_delay_sec,
                job.jct_sec,
                job.wait_on_resource_sec,
                job.wait_on_shape_sec,
                job.slowdown,
            ]
        )
    with open(stats_output, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "#job id",
                "arrival time (sec)",
                "sched time (sec)",
                "complete time (sec)",
                "size",
                "queueing (sec)",
                "jct (sec)",
                "wait on resource (sec)",
                "wait on shape (sec)",
                "slowdown",
            ]
        )
        writer.writerows(out)


def dump_cluster_stats(stats: list[tuple], stats_output: str):
    """
    Convert cluster stats exported from ClusterManager to a csv file.
    """
    if not stats or not stats_output:
        raise ValueError("Invalid input to output to trace.")

    out = []
    for t, util, running, queued in stats:
        out.append([t, util, running, queued])
    with open(stats_output, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["#time (sec)", "util", "jobs queued", "jobs running"])
        writer.writerows(out)
