This project leverages OCS to improve the ML cluster utilization and job performance.

# Requirements

This simulator requires Python 3.9+ and some Python packages.
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
To start a simulation, run:
```bash
python launch.py
```
To run all unit tests, execute:
```bash
python -m unittest
```
