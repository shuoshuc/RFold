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
C1_TRACE = "WorkloadGen/data/c1_trace.csv"
C2_TRACE = "WorkloadGen/data/c2_trace.csv"

C1_MODEL = "Cluster/models/c1.json"
C2_MODEL = "Cluster/models/c2.json"

# Scheduling policy to use.
SCHED_POLICY = "simplefit"

# Path to the cluster spec file.
MODEL_FILE = C1_MODEL

# The trace to use.
TRACE_NAME = C1_TRACE
