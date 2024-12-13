from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Tuple, Optional, Union

from common.flags import *


class TopoType(Enum):
    T2D = "2D Torus"
    T3D_NT = "3D Torus non-twisted"
    T3D_T = "3D Torus twisted"
    CLOS = "Folded Clos"


@dataclass(order=True)
class Job:
    # Priority + monotonic UUID should be unique to distinguish jobs.
    priority: float = field(init=False, repr=False)
    uuid: int
    topology: TopoType
    # For 2D Torus, shape should appear as (x, y), where x and y are the
    # number of XPUs in each dimension.
    # For 3D Torus, shape is (x, y, z).
    # For folded Clos, shape should appear as a tuple of (x, y, z, ...),
    # where each x, y, z represents the number of XPUs on a single machine.
    shape: Tuple[Union[float, int], ...]
    size: Union[float, int]
    # Absolute job arrival time, not IAT.
    arrival_time_sec: float = 0
    duration_sec: Optional[float] = None
    # The time when the job is scheduled/starts executing.
    sched_time_sec: Optional[float] = None

    def __post_init__(self):
        self.priority = self.arrival_time_sec

    def short_print(self):
        return f"[Job {self.uuid}, arrival t = {self.arrival_time_sec}]"


@dataclass(order=True)
class Subjob:
    """
    A partial job that gets mapped to a single XPU.
    """

    # Completion time of the subjob, calculated based on arrival time and duration.
    # Subjobs are sorted by their completion time.
    completion_time_sec: float = field(init=False)
    # Must be the same as the parent job's uuid.
    uuid: int
    # Size of the subjob, i.e., the number of XPU needed.
    size: Union[float, int]
    # Absolute job arrival time, not IAT.
    arrival_time_sec: float
    duration_sec: float
    # Set by the scheduler when the job is scheduled.
    sched_time_sec: float

    def __post_init__(self):
        self.completion_time_sec = self.sched_time_sec + self.duration_sec


def SplitShape(shape: str, topo: TopoType) -> Tuple[Union[float, int], ...]:
    """
    Splits the given shape string into a tuple of integers or floats.
    E.g., 1+1+1+1 -> (1, 1, 1, 1).
    """
    # Torus shape is separated by 'x', while Clos shape is separated by '+'.
    shape_delim = "+" if topo == TopoType.CLOS else "x"
    return tuple(
        map(lambda x: float(x) if FRAC_XPU else ceil(float(x)), shape.split(shape_delim))
    )
