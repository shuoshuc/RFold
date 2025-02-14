import argparse

# Some file path string constants.
# C1: 2D Torus
# C2: 2-tier Clos
TPU_JOB_SIZES_DIST = "WorkloadGen/data/tpu_job_size.csv"
TPU_ARRIVAL_TIME_DIST = "WorkloadGen/data/tpu_arrival_time.csv"
IAT_DIST = "WorkloadGen/data/iat.csv"
PHILLY_TRACE = "WorkloadGen/data/philly_trace.csv"
ALIBABA_TRACE = "WorkloadGen/data/alibaba_v2020.csv"
HELIOS_TRACE = "WorkloadGen/data/helios.csv"
ACME_TRACE = "WorkloadGen/data/acme.csv"
C1_TRACE = "WorkloadGen/data/c1_trace.csv"
C2_TRACE = "WorkloadGen/data/c2_trace.csv"

C1_MODEL = "Cluster/models/c1.json"
C2_MODEL = "Cluster/models/c2.json"


class Flags:
    def __init__(self):
        # Parse command line arguments.
        self.parser = argparse.ArgumentParser(description="Simulation entry point.")
        self.parser.add_argument(
            "-t",
            "--sim_sec",
            type=int,
            default=360000,
            help=("Simulation duration in seconds."),
        )
        self.parser.add_argument(
            "--defer_sched_sec",
            type=int,
            default=int(1e9),
            help=(
                "Time duration (seconds) a job is deferred for scheduling."
                "This typically happens when the initial scheduling decision of "
                "a job is reject. The deferral avoids busy looping."
            ),
        )
        self.parser.add_argument(
            "--sched_policy",
            type=str,
            default="firstfit",
            help=(
                "Scheduling policy to use. Available options: firstfit, slurm_hilbert."
            ),
        )
        self.parser.add_argument(
            "--no_ignore_twist",
            action="store_true",
            help=("Whether to ignore twisted torus and treat them as normal torus."),
        )
        self.parser.add_argument(
            "--t1_reserved_ports",
            type=int,
            default=0,
            help=(
                "Number of reserved ports on each T1 switch."
                "This is to model further scaling (T1 connecting to T2), even though T2 switches "
                "are not really listed in the cluster spec."
            ),
        )
        self.parser.add_argument(
            "--dim",
            type=lambda s: tuple(int(item) for item in s.split(",")),
            default=(16, 16, 16),
            help=(
                "Comma separated tuple of dimensions. "
                "E.g., 16,16 for a 2D torus with 16 nodes in each dimension, "
                "16,16,16 for a 3D torus with 16 nodes in each dimension."
            ),
        )
        self.parser.add_argument(
            "--model_file",
            type=str,
            default="",
            help=("Path to the cluster spec file."),
        )
        self.parser.add_argument(
            "-r",
            "--replay_trace",
            type=str,
            default="",
            help=("The trace file to replay."),
        )
        self.parser.add_argument(
            "--iat_file",
            type=str,
            default=IAT_DIST,
            help=("Inter-arrival time distribution file."),
        )
        self.parser.add_argument(
            "--job_size_file",
            type=str,
            default=TPU_JOB_SIZES_DIST,
            help=("Job size/shape distribution file."),
        )
        self.parser.add_argument(
            "--dur_trace_file",
            type=str,
            default=PHILLY_TRACE,
            help=("Use job duration from the given trace."),
        )
        self.parser.add_argument(
            "--frac_xpu",
            action="store_true",
            help=("Enable fractional XPU support if true."),
        )
        self.parser.add_argument(
            "--log_level",
            type=str,
            default="INFO",
            help=(
                "Set the log level to one of the following (decreasing verbosity): "
                "DEBUG, INFO, WARNING, ERROR, CRITICAL."
            ),
        )
        self.parser.add_argument(
            "--trace_output",
            type=str,
            default="",
            help=("File path to write the trace to."),
        )
        self.parser.add_argument(
            "--stats_output",
            type=str,
            default="",
            help=("File path to write the stats to."),
        )
        self.args = self.parser.parse_args()

    @property
    def sim_sec(self):
        return self.args.sim_sec

    @property
    def defer_sched_sec(self):
        return self.args.defer_sched_sec

    @property
    def sched_policy(self):
        return self.args.sched_policy

    @property
    def no_ignore_twist(self):
        return self.args.no_ignore_twist

    @property
    def t1_reserved_ports(self):
        return self.args.t1_reserved_ports

    @property
    def dim(self):
        return self.args.dim

    @property
    def model_file(self):
        return self.args.model_file

    @property
    def replay_trace(self):
        return self.args.replay_trace

    @property
    def iat_file(self):
        return self.args.iat_file

    @property
    def job_size_file(self):
        return self.args.job_size_file

    @property
    def dur_trace_file(self):
        return self.args.dur_trace_file

    @property
    def frac_xpu(self):
        return self.args.frac_xpu

    @property
    def log_level(self):
        return self.args.log_level

    @property
    def trace_output(self):
        return self.args.trace_output

    @property
    def stats_output(self):
        return self.args.stats_output


# Instantiate a global flags object.
FLAGS = Flags()
