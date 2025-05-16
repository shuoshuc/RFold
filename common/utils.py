import csv
import json
import logging
import random
import simpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from multiprocessing import Queue, Process
from numpy.typing import NDArray
from itertools import product
from typing import Optional

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


def factorize(n: int, dim: int) -> tuple[int, int, int]:
    import sympy

    def _factorize(n):
        factors = []
        for p, e in sympy.factorint(n).items():
            factors.extend([p] * e)
        return factors

    x = 1
    y = 1
    z = 1
    options = ["x", "y", "z"]
    for p in _factorize(n):
        selection = str(np.random.choice(options[:dim]))
        if selection == "x":
            x *= p
        elif selection == "y":
            y *= p
        else:
            z *= p

    return sorted([x, y, z], reverse=True)


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
                job.completion_time_sec,
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


def array_to_graph(avail: NDArray) -> nx.Graph:
    """
    Constructs a NetworkX graph from a 2D or 3D numpy array.
    """
    dimensions = avail.shape
    # 1. Create the full periodic grid graph. NX expects the dimensions
    # in reverse order, e.g., (z, y, x).
    graph = nx.grid_graph(dim=dimensions[::-1], periodic=True)

    # 2. Remove unavailable nodes
    nodes_to_remove = []
    for coords in product(*(range(d) for d in dimensions)):
        if avail[coords] == 0:
            nodes_to_remove.append(coords)

    # 3. Remove the unavailable nodes and their incident edges
    graph.remove_nodes_from(nodes_to_remove)

    return graph


def normalize_cycle(cycle: list[tuple]) -> tuple[tuple]:
    """
    Normalizes a cycle to a canonical representation. The canonical form is the
    lexicographically smallest tuple representation among all rotations of the cycle
    and its reverse.
    """

    # Helper generator to yield all rotations of a given list as tuples
    # Each rotation preserves the relative order for that starting point.
    def _all_rotations(nodes: list[tuple]) -> iter:
        for i in range(len(nodes)):
            yield tuple(nodes[i:] + nodes[:i])

    cycle_variants = []
    # All rotations of the original sequence
    cycle_variants.extend(_all_rotations(cycle))
    # All rotations of the reversed sequence
    cycle_variants.extend(_all_rotations(cycle[::-1]))

    # min() uses built-in lexicographical sorting to find the "smallest" tuple.
    # This tuple represents the canonical form of the cycle.
    return min(cycle_variants)


def find_1D_cycle(avail: NDArray, n: int) -> Optional[list[tuple]]:
    """
    Finds one simple cycle of a specific length N in a graph derived from
    an availability array (2D or 3D torus).

    Args:
        avail: 2D or 3D availability array (0s and 1s).
        n: The desired length of the cycles to find (must be >= 3).

    Returns:
        One feasible cycle of length N in the graph.
        If no cycle is found, returns None.
    """
    if n < 3:
        raise ValueError(f"Cycle length N must be >= 3. Requested N={n}.")

    graph = array_to_graph(avail)
    if graph.number_of_nodes() < n:
        return []

    all_cycles_iter = nx.simple_cycles(graph, length_bound=n)
    for cycle in all_cycles_iter:
        # print(f"cycle len: {len(cycle)}")
        if len(cycle) == n:
            # Since we just need one cycle, no need to normalize it.
            # canonical_repr = normalize_cycle(cycle)
            return list(cycle)

    return None


def is_on_face(node_coord: tuple[int], axis: int, rsize: int) -> bool:
    """
    Checks if a node coordinate is on the face of the grid along the specified axis.
    Args:
        node_coord: The coordinates of the node.
        axis: The axis to check (0, 1, or 2).
        rsize: The size of the grid along the specified axis.
    Returns:
        True if the node is on the face, False otherwise.
    """
    if axis > len(node_coord) - 1:
        raise ValueError(
            f"Axis={axis} out of bounds for the node coordinates {node_coord}."
        )
    return (node_coord[axis] == 0) or (node_coord[axis] == rsize - 1)


def _find_simple_path(
    block_coord: tuple[int, ...], avail: NDArray, N: int, axis: int, rsize: int
) -> Optional[list[tuple]]:
    """
    Finds a path of length N satisfying face constraints.

    Args:
        avail: A 2D or 3D numpy array where 1 represents an
               available node and 0 represents an occupied node.
        N: The desired length of the path (number of nodes).
        axis: The axis defining the faces (0, 1, or 2 for 3D).

    Returns:
        A list of coordinate tuples representing the path,
        or None if no such path is found.
    """
    G = array_to_graph(avail)
    face_nodes = [node for node in G.nodes if is_on_face(node, axis, rsize)]

    if not face_nodes:
        return None

    for src in face_nodes:
        for dst in face_nodes:
            if src == dst:
                continue
            for path in nx.all_simple_paths(G, source=src, target=dst, cutoff=N - 1):
                if len(path) == N:
                    path_global_coords = []
                    for local_node_coord in path:
                        # Translate coordinate in the block to global coordinate.
                        global_node_coord = tuple(
                            b * rsize + l for b, l in zip(block_coord, local_node_coord)
                        )
                        path_global_coords.append(global_node_coord)
                    return path_global_coords

    return None


def find_simple_path(
    out_q: Queue,
    block_coord: tuple[int, ...],
    avail: NDArray,
    N: int,
    axis: int,
    rsize: int,
) -> Optional[list[tuple]]:
    path = _find_simple_path(
        block_coord=block_coord,
        avail=avail,
        N=N,
        axis=axis,
        rsize=rsize,
    )
    out_q.put(path)


def find_simple_path_helper(
    block_coord: tuple[int, ...],
    avail: NDArray,
    N: int,
    axis: int,
    rsize: int,
    timeout_sec: int,
) -> Optional[list[tuple]]:
    out_q = Queue()
    child_process = Process(
        target=find_simple_path,
        args=(out_q, block_coord, avail, N, axis, rsize),
        daemon=True,
    )
    child_process.start()
    child_process.join(timeout=timeout_sec)

    # After the timeout, check if the child process is still running
    if child_process.is_alive():
        child_process.kill()
        child_process.join(timeout=1)

    if not out_q.empty():
        return out_q.get()
    else:
        return None
