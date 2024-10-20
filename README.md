The MLOCS project leverages OCS to improve the ML cluster performance, reliability, cost and power consumption.

# Requirements

This simulator requires Python packages `scipy`, `numpy`, `simpy`.
To install all dependencies, run:
```bash
pip install 'scipy>=1.13.1' 'numpy>=2.0.2' 'simpy>=4.1.1'
```

# Folder structure
* common/: common things used by all modules, e.g., flags, data structure definitions.
* Cluster/: cluster/node and topology implementation.
* ClusterManager/: cluster manager module.
* WorkloadGen/: the workload generator module.
* test/: unit tests.
* launch.py: starting point of the simulator.

# How to run
To start a simulation, run:
```bash
python launch.py
```
To run all unit tests, execute:
```bash
python -m unittest
```