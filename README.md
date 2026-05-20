This project leverages OCS to improve the ML cluster utilization and job performance.

# Requirements

This simulator uses uv. However, you can also manually manage the environment by installing Python 3.9+ and some Python packages.
To install all dependencies, run:
```bash
pip install 'scipy>=1.13.1' 'numpy>=2.0.2' 'simpy>=4.1.1' 'matplotlib>=3.9.2' 'numpy-hilbert-curve>=1.0.1' 'networkx>=3.2.1' 'sympy>=1.13.3'
```

# Folder structure
* **common/**: common things used by all modules, e.g., flags, data structure definitions.
* **Cluster/**: cluster/node and topology implementation.
* **ClusterManager/**: cluster manager module.
* **WorkloadGen/**: the workload generator module.
* **test/**: unit tests.
* **launch.py**: starting point of the simulator.

# How to run
First make sure your local uv environment is up tp date:
```bash
uv sync
```
To start a simulation, run:
```bash
uv run launch.py
```
To run all unit tests, execute:
```bash
uv run python -m unittest
```

# Astra-sim Docker image

To build the image, run:
```bash
docker buildx build --load -t rfold-astra .
```

## In-image layout

- `/app/configs/` — static configs baked into the image (symlinks into
  `astra-sim-artifacts/examples/fluid-model/inputs/`):
  `RemoteMemory.json`, `network.yml`, `sys.json`.
- `/app/tools/` — symlink to `astra-sim-artifacts/tools/`.
- `/app/inputs/` — **not pre-created**. Bind-mount the host directory
  containing your dynamic inputs (`bw_schedule.txt`,
  `latency_schedule.txt`) here at run time.
- `/app/output/` — **not pre-created**. Bind-mount a host directory here
  to collect results (`jct.csv`).

## Running the fluid-model experiment

```bash
docker run --rm \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  -v /host/path/to/schedules:/app/inputs:ro \
  -v /host/path/to/output:/app/output \
  rfold-astra \
  bash /app/astra-sim-artifacts/examples/fluid-model/run.sh <JOB_SHAPE>
```

`<JOB_SHAPE>` is the torus shape in `XxYxZ` form (e.g., `2x2x1`). The
host's `schedules` directory must contain `bw_schedule.txt` and
`latency_schedule.txt`.
