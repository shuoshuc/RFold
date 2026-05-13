from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from math import ceil
from typing import Iterator, Tuple, Optional, Union, TypedDict

from common.flags import FLAGS

# One physical placement record carried by Job.allocation as a value.
AllocEntry = TypedDict("AllocEntry", {"node": str, "num_xpu": Union[int, float]})


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
    # The time when the job is scheduled/starts executing. This field could contain
    # a stale time if the job has been attempted scheduling at least once.
    sched_time_sec: Optional[float] = None
    # Job completion time stamp.
    completion_time_sec: Optional[float] = None
    # Maps logical rank -> physical placement. Logical rank is a dense int
    # in [0, size). Within a single chunk/submesh ranks are x-fastest;
    # across chunks they follow the scheduler's chunk-processing order.
    allocation: dict[int, AllocEntry] = field(default_factory=dict)
    # Directed ring communication pattern over logical ranks. Populated in
    # __post_init__ for torus topologies (T2D, T3D_NT, T3D_T); left None
    # otherwise. Access via getCommPattern() to get a clear error for jobs
    # whose topology has no defined pattern. Sits inside the allocation
    # block because it needs a default, and non-default fields must
    # precede defaulted ones in the dataclass declaration order.
    _comm_pattern: Optional[list[Tuple[int, int]]] = field(
        default=None, init=False, compare=False, repr=False
    )
    # ----- end of allocation info -----

    # ----- stats -----
    # Job queueing delay.
    queueing_delay_sec: Optional[float] = None
    # Job completion time duration (including queueing and other waiting time).
    jct_sec: Optional[float] = None
    # A ratio of JCT / job duration. 1 means no slowdown.
    slowdown: Optional[float] = None
    # Reason for last job rejection. Must be one of: "resource", "shape".
    reject_reason: Optional[str] = field(default=None, compare=False)
    # Time of last rejection.
    last_reject_time: Optional[float] = field(default=None, compare=False)
    # Time spent on waiting for the shape to be available.
    wait_on_shape_sec: Optional[float] = field(default=0, compare=False)
    # Time spent on waiting for the total resources to be available.
    wait_on_resource_sec: Optional[float] = field(default=0, compare=False)
    # ----- end of stats -----

    # ----- contention model state -----
    # Ideal work completed so far (in seconds of ideal duration). Advances at rate
    # 1/current_slowdown per wall-clock second between events. Capped at duration_sec.
    work_done_ideal_sec: float = field(default=0.0, compare=False)
    # Wall-clock time at which work_done_ideal_sec was last updated. None until the
    # job is admitted and its first slowdown is applied. Used to compute the elapsed
    # wall time delta when a new slowdown factor arrives.
    last_event_time_sec: Optional[float] = field(default=None, compare=False)
    # The slowdown factor in effect for this job since last_event_time_sec.
    # 1.0 = no contention, > 1.0 = the job runs slower (e.g., 2.0 means 1 wall sec
    # advances 0.5 sec of ideal work). Recomputed by ContentionModel on each
    # admit/complete event.
    current_slowdown: float = field(default=1.0, compare=False)
    # ----- end of contention model state -----

    def __post_init__(self):
        self.priority = self.arrival_time_sec
        if self.topology in (TopoType.T2D, TopoType.T3D_NT, TopoType.T3D_T):
            # Torus shapes are integer in this codebase; coerce defensively
            # since the shape tuple is typed as Union[float, int].
            self._comm_pattern = compute_ring_comm_pattern(
                tuple(int(s) for s in self.shape)
            )

    def getCommPattern(self) -> list[Tuple[int, int]]:
        """
        Return the directed ring communication pattern over logical ranks.

        Raises ValueError if this job's topology has no defined pattern
        (mesh and Clos topologies).
        """
        if self._comm_pattern is None:
            raise ValueError(
                f"Comm pattern undefined for topology {self.topology}"
            )
        return self._comm_pattern

    def addToAllocation(
        self, node_name: str, num_xpu: Union[int, float] = 1
    ) -> None:
        """Append a placement with the next sequential logical rank."""
        self.allocation[len(self.allocation)] = {
            "node": node_name,
            "num_xpu": num_xpu,
        }

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
            f"jct={self.jct_sec}, "
            f"slowdown={self.slowdown}"
        )

    def extra_stats(self):
        return (
            f"Job {self.uuid}, "
            f"queue={self.queueing_delay_sec}, "
            f"wait_on_shape={self.wait_on_shape_sec}, "
            f"wait_on_resource={self.wait_on_resource_sec}"
        )

    def logRejectReason(self, current_time: float, reason: str):
        """
        Logs the reason for the job rejection, and track the waiting time.
        """
        # Update the time tracking. If it is the first rejection of the job, skip this
        # and only record the rejection.
        if self.reject_reason == "shape":
            self.wait_on_shape_sec += current_time - self.last_reject_time
        elif self.reject_reason == "resource":
            self.wait_on_resource_sec += current_time - self.last_reject_time
        self.reject_reason = reason
        self.last_reject_time = current_time

    def updateQueueingTime(self, current_time: float):
        """
        Updates the end-to-end queueing time and time breakdown for the job.
        """
        # Update the time tracking. If it is the first rejection of the job, skip this
        # and only record the rejection.
        # If the job has never been rejected, the queueing time is 0.
        if self.reject_reason == "shape":
            self.wait_on_shape_sec += current_time - self.last_reject_time
        elif self.reject_reason == "resource":
            self.wait_on_resource_sec += current_time - self.last_reject_time
        self.sched_time_sec = current_time
        self.queueing_delay_sec = self.sched_time_sec - self.arrival_time_sec
        # Clean up the reject reason and last reject time since the job has started execution.
        self.reject_reason = None
        self.last_reject_time = None

    def applySlowdown(self, new_slowdown: float, current_time: float):
        """
        Tell this job its slowdown for the next period.
        1) Accrue ideal work done since the last event at the prior factor.
        2) Store the new factor and the current time.
        3) Update self.priority to the projected ETA = now + remaining_ideal * new_s.
        """
        if self.last_event_time_sec is not None:
            elapsed_wall = current_time - self.last_event_time_sec
            self.work_done_ideal_sec += elapsed_wall / self.current_slowdown
        # Clamp against floating-point overshoot.
        self.work_done_ideal_sec = min(self.work_done_ideal_sec, self.duration_sec)
        self.last_event_time_sec = current_time
        self.current_slowdown = new_slowdown
        remaining_ideal = self.duration_sec - self.work_done_ideal_sec
        self.priority = current_time + remaining_ideal * new_slowdown


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


def enumerate_xmajor(
    extents: Tuple[int, ...],
    origin: Optional[Tuple[int, ...]] = None,
) -> Iterator[Tuple[int, ...]]:
    """
    Yield N-D coordinates over a box of the given extents in x-fastest
    row-major order (first dim varies fastest, then second, then third).

    For extents (sx, sy, sz):
      (0,0,0), (1,0,0), ..., (sx-1,0,0),
      (0,1,0), ..., (sx-1,sy-1,0),
      ...,
      (sx-1, sy-1, sz-1)

    If `origin` is given, each yielded coord is offset by `origin`.
    """
    if origin is None:
        origin = tuple(0 for _ in extents)
    # itertools.product enumerates the last arg fastest. Pass extents reversed
    # and reverse each yielded tuple to flip to x-fastest.
    for rev in product(*[range(e) for e in reversed(extents)]):
        coord = rev[::-1]
        yield tuple(c + o for c, o in zip(coord, origin))


def compute_ring_comm_pattern(
    shape: Tuple[int, ...],
) -> list[Tuple[int, int]]:
    """
    Build the directed ring communication pattern over logical ranks for a
    torus job of the given shape.

    Ranks are x-fastest:
      rank(c0, c1, ..., c_{n-1}) = c0 + c1*shape[0] + c2*shape[0]*shape[1] + ...

    For each axis d in 0..ndim-1, for every fiber along axis d (every fixed
    assignment of the remaining coords), emit forward ring edges:
      (rank_k, rank_{(k+1) mod shape[d]})  for k = 0 .. shape[d]-1

    Edges are listed dim-major (all axis-0 edges, then axis-1, ...). Within
    an axis, fibers are visited in lowest-remaining-index-fastest order
    (matches enumerate_xmajor on the projected shape).

    A degenerate axis (shape[d] == 1) contributes no edges (no self-loops).
    """
    ndim = len(shape)
    # strides[d] = product(shape[0..d-1]), so rank = sum(c_d * strides[d]).
    strides = [1] * ndim
    for d in range(1, ndim):
        strides[d] = strides[d - 1] * shape[d - 1]

    edges: list[Tuple[int, int]] = []
    for d in range(ndim):
        s_d = shape[d]
        if s_d <= 1:
            continue
        remaining_axes = [a for a in range(ndim) if a != d]
        remaining_extents = [shape[a] for a in remaining_axes]
        # itertools.product enumerates the last arg fastest. Reverse the
        # extents so the lowest remaining index varies fastest, then reverse
        # each yielded tuple back to original order.
        for rev in product(*[range(e) for e in reversed(remaining_extents)]):
            fiber_coords = rev[::-1]
            base_rank = 0
            for axis, coord in zip(remaining_axes, fiber_coords):
                base_rank += coord * strides[axis]
            for k in range(s_d):
                src = base_rank + k * strides[d]
                dst = base_rank + ((k + 1) % s_d) * strides[d]
                edges.append((src, dst))
    return edges


def SplitShape(shape: str, topo: TopoType) -> Tuple[Union[float, int], ...]:
    """
    Splits the given shape string into a tuple of integers or floats.
    E.g., 1+1+1+1 -> (1, 1, 1, 1).
    """
    # Torus shape is separated by 'x', while Clos shape is separated by '+'.
    shape_delim = "+" if topo == TopoType.CLOS else "x"
    return tuple(
        map(
            lambda x: float(x) if FLAGS.frac_xpu else ceil(float(x)),
            shape.split(shape_delim),
        )
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
