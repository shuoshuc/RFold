"""
Per-job inputs and orchestration for the rfold-astra fluid-model
container. Pure helpers (matrix builders, file writers, jct.csv parser)
live here so they can be unit-tested without spawning Docker. The only
non-pure function is `run_astra`, which is the subprocess seam.

JCT values are passed through in nanoseconds end-to-end — astra-sim
emits ns in jct.csv, and consumers (e.g. Job.astra_ideal_dur_nsec)
store ns directly.
"""

import logging
import subprocess
from itertools import product
from pathlib import Path
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
        <tag> 0
        <row 0: N space-separated floats>
        ...
        <row N-1>
        END

    The "0" on the tag line is the topology ID. astra-sim's parser
    (astra-sim/network_frontend/analytical/reconfigurable/main.cc) does
    `std::stoi(line.substr(3))` on the tag line and aborts if no ID is
    present; topo_id 0 is the required initial-topology entry. We only
    emit one block per file because our integration is single-topology
    per job (no in-run reconfiguration).
    `path` accepts anything pathlib.Path or os.PathLike-like.
    """
    with open(path, "w") as f:
        f.write(f"{tag} 0\n")
        for row in matrix:
            f.write(" ".join(repr(x) for x in row) + "\n")
        f.write("END\n")


def parse_jct_nsec(csv_path) -> float:
    """
    Read jct.csv (astra-sim format: `Job,JCT (nsec)` header followed by
    `<job_id>,<jct_ns>` rows) and return the first job's JCT in
    nanoseconds (the same unit astra-sim emits).

    Raises RuntimeError if the file is missing, empty, has no data row,
    or the JCT cell is not numeric.
    """
    p = Path(csv_path)
    if not p.exists():
        raise RuntimeError(f"jct.csv not found at {csv_path}")
    lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"jct.csv at {csv_path} is empty")
    if lines[0].startswith("Job"):
        lines = lines[1:]
    if not lines:
        raise RuntimeError(f"jct.csv at {csv_path}: no data rows")
    parts = lines[0].split(",")
    if len(parts) < 2:
        raise RuntimeError(
            f"jct.csv at {csv_path}: data row '{lines[0]}' has < 2 columns"
        )
    try:
        return float(parts[1].strip())
    except ValueError:
        raise RuntimeError(
            f"jct.csv at {csv_path}: JCT cell '{parts[1]}' is not numeric"
        )


def run_astra(
    uuid: int,
    shape: Tuple[int, ...],
    bw_matrix: list[list[float]],
    lt_matrix: list[list[float]],
    tmp_root,
) -> float:
    """
    Materialize per-job inputs under <tmp_root>/<uuid>/inputs/, invoke
    run_astra.sh, and return the parsed JCT in nanoseconds.

    The caller supplies BW (GB/s) and LT (ns) matrices; this function no
    longer fabricates them from torus-neighbor defaults. Use
    build_bw_matrix / build_lt_matrix if the ideal (no-contention) view
    is what you want.

    Raises subprocess.CalledProcessError if the container exits non-zero
    or RuntimeError if jct.csv is malformed/missing.
    """
    tmp_root = Path(tmp_root)
    uuid_dir = tmp_root / str(uuid)
    inputs_dir = uuid_dir / "inputs"
    outputs_dir = uuid_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    write_schedule(inputs_dir / "bw_schedule.txt", bw_matrix, "BW")
    write_schedule(inputs_dir / "latency_schedule.txt", lt_matrix, "LT")

    shape_str = "x".join(str(s) for s in shape)
    cmd = [
        "bash", "./run_astra.sh", shape_str,
        "--input-dir", str(inputs_dir),
        "--output-dir", str(outputs_dir),
    ]
    log_stdout = outputs_dir / "astra_sim.log"
    log_stderr = outputs_dir / "astra_sim.err"
    with open(log_stdout, "w") as fout, open(log_stderr, "w") as ferr:
        try:
            subprocess.run(cmd, check=True, stdout=fout, stderr=ferr)
        except subprocess.CalledProcessError:
            logging.error(
                "astra-sim failed for uuid=%s shape=%s; "
                "see %s and %s for diagnostics",
                uuid, shape, log_stdout, log_stderr,
            )
            raise

    return parse_jct_nsec(outputs_dir / "jct.csv")
