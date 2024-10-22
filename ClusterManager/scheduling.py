import random
import simpy
from enum import Enum

from common.job import Job


class SchedDecision(Enum):
    # Admit a job immediately.
    ADMIT = 1
    # Reject a job immediately.
    REJECT = 2
    # Preempt running jobs to admit a job.
    # May take some time to migrate.
    PREEMPT = 3
    # Reconfigure the cluster topology to admit a job.
    # May take some time.
    RECONFIGURE = 4


class SchedulingPolicy:
    def __init__(self, env: simpy.core.Environment):
        self.env = env
        random.seed(42)

    def place(self, job: Job) -> SchedDecision:
        return random.choice([d for d in SchedDecision])
