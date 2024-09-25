#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional

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
    shape: Tuple[int, ...]
    size: int
    arrival_time_sec: float = 0
    duration_minutes: Optional[float] = None