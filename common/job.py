from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Tuple, Optional, Union, TypedDict

from common.flags import *


# Definition of type Allocation: it is a dict.
Allocation = TypedDict("Allocation", {"node": str, "num_xpu": Union[int, float]})


class TopoType(Enum):
    MESH2D = "2D Mesh"
    T2D = "2D Torus"
    MESH3D = "3D Mesh"
    T3D_NT = "3D Torus non-twisted"
    T3D_T = "3D Torus twisted"
    CLOS = "Folded Clos"


@dataclass(order=True)
class Job:
    # Priority + monotonic UUID should be unique to distinguish jobs.
    priority: float = field(init=False, repr=False)
    uuid: int
    # Absolute job arrival time, not IAT.
    arrival_time_sec: float

    # ----- resource requirements -----
    topology: TopoType
    # For 2D Torus, shape should appear as (x, y), where x and y are the
    # number of XPUs in each dimension.
    # For 3D Torus, shape is (x, y, z).
    # For folded Clos, shape should appear as a tuple of (x, y, z, ...),
    # where each x, y, z represents the number of XPUs on a single machine.
    shape: Tuple[Union[float, int], ...]
    size: Union[float, int]
    duration_sec: Optional[float] = None
    # ----- end of resource requirements -----

    # ----- allocation info -----
    # The time when the job is scheduled/starts executing.
    sched_time_sec: Optional[float] = None
    # Detailed allocation provided after a scheduling decision (admit or similar) is made.
    allocation: Optional[Allocation] = field(default_factory=dict)
    # ----- end of allocation info -----

    # ----- stats -----
    # Job queueing delay.
    queueing_delay_sec: Optional[float] = None
    # Job completion time (including queueing and other waiting time).
    completion_time_sec: Optional[float] = None
    # A ratio of JCT / job duration. 1 means no slowdown.
    slowdown: Optional[float] = None
    # ----- end of stats -----

    def __post_init__(self):
        self.priority = self.arrival_time_sec

    def short_print(self):
        return (
            f"[Job {self.uuid}, arrive t={self.arrival_time_sec}, "
            f"size={self.size}, shape={self.shape}, dur={self.duration_sec}]"
        )

    def stats(self):
        return (
            f"Job {self.uuid}, t_arr={self.arrival_time_sec}, t_sch={self.sched_time_sec}, "
            f"t_comp={self.completion_time_sec}, "
            f"queue={self.queueing_delay_sec}, "
            f"jct={self.completion_time_sec}, "
            f"slowdown=({self.completion_time_sec}/{self.duration_sec})={self.slowdown}"
        )


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


def FormShape(shape: Tuple[Union[float, int], ...], topo: TopoType) -> str:
    """
    Formats the given shape tuple into a string.
    E.g., (1, 1, 1, 1) -> 1+1+1+1
    """
    # Torus shape is separated by 'x', while Clos shape is separated by '+'.
    shape_delim = "+" if topo == TopoType.CLOS else "x"
    # Unlike SplitShape(), there is no need to worry about fractional XPUs,
    # because the given shape from a trace should already be aware of that.
    return shape_delim.join(map(str, shape))
