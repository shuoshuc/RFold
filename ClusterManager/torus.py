import networkx as nx
import numpy as np
from math import prod
from multiprocessing import Queue, Process
from numpy.typing import NDArray
from itertools import product
from sympy import divisors
from typing import Optional


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


def real_shape_dimension(shape: tuple[int, ...]) -> int:
    """
    Check if the job is 1D, 2D or 3D.
    Note: single node job is technically 1D but we return a 0.
    """
    return sum(dim != 1 for dim in shape)


def folded_shape_helper(shape: tuple[int, ...]) -> set[tuple[int, ...]]:
    """
    Given a shape tuple, return all possible folded shapes.
    """

    def exempted(n: int, num_factors: int) -> bool:
        # Smallest N for a*b=N with a,b > 1 is 2*2=4
        if num_factors == 2 and n < 4:
            return True
        # Smallest N for a*b*c=N with a,b,c > 1 is 2*2*2=8
        if num_factors == 3 and n < 8:
            return True
        return False

    def factor_bound(i: int, num_factors: int) -> int:
        return i**2 if num_factors == 2 else i**3

    num_factors = len(shape)
    if num_factors not in [2, 3]:
        raise ValueError("Invalid shape dimension.")
    # The original shape is always a feasible option.
    results = {tuple(sorted(shape))}
    n = prod(shape)
    # No need to proceed for a very small n.
    if exempted(n, num_factors):
        return set()

    for a in divisors(n):
        # Check up to sqrt(n) / cbrt(n) to deduplicate pairs.
        if factor_bound(a, num_factors) > n:
            break

        if num_factors == 2:
            results.add(tuple(sorted((a, n // a))))
        elif num_factors == 3:
            # Fix a, now find b and c such that b*c = n/a.
            bc = n // a
            for b in divisors(bc):
                if b * b > bc:
                    break
                results.add(tuple(sorted((a, b, bc // b))))
    return results


def are_homomorphic(orig: tuple[int, ...], target: tuple[int, ...], rsize: int) -> bool:
    """
    Constructs two torus graphs from the given shapes and checks if they are homomorphic.
    """
    # Note: only dimensions that are multiples of `rsize` can have wrap-around links.
    orig_graph = nx.grid_graph(
        orig,
        periodic=[True if dim % rsize == 0 else False for dim in orig],
    )
    target_graph = nx.grid_graph(
        target,
        periodic=[True if dim % rsize == 0 else False for dim in target],
    )
    isomatcher = nx.isomorphism.GraphMatcher(target_graph, orig_graph)
    # return isomatcher.subgraph_is_isomorphic()
    return isomatcher.subgraph_is_monomorphic()
