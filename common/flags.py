# Enable fractional XPU support if true.
FRAC_XPU = False

# True to replay a trace, False to generate new workload.
USE_TRACE = True

# Set the log level to one of the following (decreasing verbosity):
# DEBUG, INFO, WARNING, ERROR, CRITICAL.
LOG_LEVEL = "INFO"

# Time duration (seconds) a job is deferred for scheduling.
# This typically happens when the initial scheduling decision of
# a job is reject. The deferral avoids busy looping.
DEFERRED_SCHED_SEC = 30

# Length of the simulation in seconds.
SIM_DURATION_SEC = 400000

# Some file path string constants.
TPU_JOB_SIZES_DIST = "WorkloadGen/data/tpu_job_size.csv"
TPU_ARRIVAL_TIME_DIST = "WorkloadGen/data/tpu_arrival_time.csv"
PHILLY_TRACE = "WorkloadGen/data/philly_trace.csv"
ALIBABA_TRACE = "WorkloadGen/data/alibaba_v2020.csv"
HELIOS_TRACE = "WorkloadGen/data/helios.csv"
ACME_TRACE = "WorkloadGen/data/acme.csv"
TOY_TRACE = "WorkloadGen/data/toy_trace.csv"

# Scheduling policy to use.
SCHED_POLICY = "simplefit"

# Number of nodes in the cluster.
NUM_NODES = 1

# Number of XPUs on each node.
NUM_XPU = 8
