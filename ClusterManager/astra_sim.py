"""
Per-job inputs and orchestration for the rfold-astra fluid-model
container. Pure helpers (matrix builders, file writers, jct.csv parser)
live here so they can be unit-tested without spawning Docker. The only
non-pure function is `run_astra`, which is the subprocess seam.
"""

from itertools import product
from typing import Tuple


def _torus_neighbor_matrix(
    shape: Tuple[int, ...], value: float
) -> list[list[float]]:
    """
    N×N matrix (N = prod(shape)) where every bidirectional torus-neighbor
    pair (i, j) gets `value` and every other cell (including the
    diagonal) gets 0.0. Rank numbering is x-fastest, matching
    `common/job.py::compute_ring_comm_pattern`.
    """
    N = 1
    for s in shape:
        N *= s
    M = [[0.0] * N for _ in range(N)]
    strides = [1] * len(shape)
    for d in range(1, len(shape)):
        strides[d] = strides[d - 1] * shape[d - 1]
    for d in range(len(shape)):
        s_d = shape[d]
        if s_d <= 1:
            continue
        remaining_axes = [a for a in range(len(shape)) if a != d]
        remaining_extents = [shape[a] for a in remaining_axes]
        for rev in product(*[range(e) for e in reversed(remaining_extents)]):
            fiber_coords = rev[::-1]
            base_rank = sum(
                c * strides[a] for a, c in zip(remaining_axes, fiber_coords)
            )
            for k in range(s_d):
                src = base_rank + k * strides[d]
                dst = base_rank + ((k + 1) % s_d) * strides[d]
                M[src][dst] = value
                M[dst][src] = value
    return M


def build_bw_matrix(
    shape: Tuple[int, ...], default: float = 50.0
) -> list[list[float]]:
    """N×N bandwidth matrix; bidirectional torus neighbors = default."""
    return _torus_neighbor_matrix(shape, default)


def build_lt_matrix(
    shape: Tuple[int, ...], default: float = 500.0
) -> list[list[float]]:
    """N×N latency matrix (ns); bidirectional torus neighbors = default."""
    return _torus_neighbor_matrix(shape, default)


def write_schedule(path, matrix: list[list[float]], tag: str) -> None:
    """
    Write a fluid-model schedule file at `path`. Format:
        <tag>
        <row 0: N space-separated floats>
        ...
        <row N-1>
        END
    `path` accepts anything pathlib.Path or os.PathLike-like.
    """
    with open(path, "w") as f:
        f.write(f"{tag}\n")
        for row in matrix:
            f.write(" ".join(repr(x) for x in row) + "\n")
        f.write("END\n")
