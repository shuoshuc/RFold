import random
import simpy
from enum import Enum
from typing import Tuple

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

    def place(self, job: Job) -> Tuple[SchedDecision, Job]:
        """
        Make a scheduling decision for a job. Note that the job (e.g., shape, duration)
        could be modifed to achieve a more desirable scheduling decision. But if the job
        is rejected, it is not modified.
        The (modified) job is returned along with the decision.
        """
        return random.choice([d for d in SchedDecision]), job
