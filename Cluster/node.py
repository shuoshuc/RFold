import logging
from typing import Union

from queue import PriorityQueue
from common.job import Subjob


class Node:
    def __init__(self, name, num_xpu):
        """
        name: FQDN of the node
        num_xpu: number of XPUs on the node
        """
        if num_xpu < 1:
            raise ValueError("Number of XPUs must be at least 1.")
        self.name = name
        self.num_xpu = num_xpu
        self.num_idle_xpu = num_xpu
        # # Each XPU has its own subjob queue.
        # self.subjob_queues: list[PriorityQueue[Subjob]] = [
        #     PriorityQueue() for _ in range(num_xpu)
        # ]

    def numXPU(self) -> int:
        """
        Returns the number of XPUs on the node.
        """
        return self.num_xpu

    def numIdleXPU(self) -> Union[int, float]:
        """
        Returns the number of idle XPUs on the node.
        """
        return self.num_idle_xpu

    def alloc(self, n_xpu: Union[int, float]):
        """
        Allocate N XPUs.
        """
        if self.num_idle_xpu < n_xpu:
            raise ValueError(
                f"Node {self.name} XPU requested exceeds available: "
                f"{n_xpu} vs {self.num_idle_xpu}."
            )
        self.num_idle_xpu -= n_xpu
        logging.info(
            f"Node {self.name} state: {self.numIdleXPU()} / {self.numXPU()} XPUs."
        )

    def free(self, n_xpu: Union[int, float]):
        """
        Free N XPUs.
        """
        if self.num_idle_xpu + n_xpu > self.num_xpu:
            raise ValueError(
                f"Node {self.name} idle XPU exceeds total number: "
                f"{self.num_idle_xpu} + {n_xpu} > {self.num_xpu}."
            )
        self.num_idle_xpu += n_xpu
        logging.info(
            f"Node {self.name} state: {self.numIdleXPU()} / {self.numXPU()} XPUs."
        )
