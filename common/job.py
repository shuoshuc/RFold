#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Union
from common.flags import *
from math import ceil


class TopoType(Enum):
    T2D = '2D Torus'
    T3D_NT = '3D Torus non-twisted'
    T3D_T = '3D Torus twisted'
    CLOS = 'Folded Clos'


@dataclass
class Job():
    uuid: int
    topology: TopoType
    # For 2D Torus, shape should appear as (x, y), where x and y are the
    # number of XPUs in each dimension.
    # For 3D Torus, shape is (x, y, z).
    # For folded Clos, shape should appear as a tuple of (x, y, z, ...),
    # where each x, y, z represents the number of XPUs on a single machine.
    shape: Tuple[Union[float, int], ...]
    size: Union[float, int]
    arrival_time_sec: float = 0
    duration_minutes: Optional[float] = None


def SplitShape(shape: str, topo: TopoType) -> Tuple[Union[float, int], ...]:
    '''
    Splits the given shape string into a tuple of integers or floats.
    E.g., 1+1+1+1 -> (1, 1, 1, 1).
    '''
    # Torus shape is separated by 'x', while Clos shape is separated by '+'.
    shape_delim = '+' if topo == TopoType.CLOS else 'x'
    return tuple(map(lambda x: float(x) if FRAC_XPU else ceil(x), shape.split(shape_delim)))
