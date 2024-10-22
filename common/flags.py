# Enable fractional XPU support if true.
FRAC_XPU = True

# True to replay a trace, False to generate new workload.
USE_TRACE = True

# Set the log level to one of the following (decreasing verbosity):
# DEBUG, INFO, WARNING, ERROR, CRITICAL.
LOG_LEVEL = "INFO"

# Length of the simulation in seconds.
SIM_DURATION_SEC = 400000

# Some file path string constants.
TPU_JOB_SIZES_DIST = "WorkloadGen/data/tpu_job_size.csv"
TPU_ARRIVAL_TIME_DIST = "WorkloadGen/data/tpu_arrival_time.csv"
PHILLY_TRACE = "WorkloadGen/data/philly_trace.csv"
ALIBABA_TRACE = "WorkloadGen/data/alibaba_v2020.csv"
