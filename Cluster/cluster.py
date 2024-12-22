import logging
import simpy
from typing import Union

from common.job import Job
from common.flags import *
from Cluster.node import Node


class Cluster:
    """
    A cluster is the high-level abstraction of the resources.
    It contains a collection of nodes, and also models the network topology, link
    bandwidth, etc. The cluster manager and other external components interact with
    this class to access the resources. This class also exposes certain states for
    monitoring purposes.
    """

    def __init__(self, env: simpy.core.Environment, num_nodes: int, num_xpu: int):
        """
        num_nodes: number of nodes in the cluster.
        num_xpu: number of XPUs on each node.
        """
        self.env = env
        # A map from node ID to node object.
        self.nodes = {}
        for i in range(num_nodes):
            node_id = f"n{i+1}"
            self.nodes[node_id] = Node(node_id, num_xpu)

    def execute(self, job: Job):
        """
        Executes a job on the cluster. The job is broken down into subjobs and sent to
        nodes for execution. The shape and duration of the job remain as-is and should
        always succeed execution, because the scheduler admits the job only when it can
        successfully start.
        """
        logging.info(f"t = {self.env.now}, executing job {job.short_print()}")
        if not job.allocation:
            raise ValueError(f"Job {job.uuid} allocation info is missing.")
        if len(job.shape) != len(job.allocation):
            raise ValueError(
                f"Job {job.uuid} shape and allocation mismatch: "
                f"{job.shape} vs {job.allocation}."
            )
        for node_id, num_xpu in job.allocation.items():
            self.nodes[node_id].alloc(num_xpu)
        # TODO: break down the job into subjobs, send to nodes for execution.
        # Update states.

    def complete(self, job: Job):
        """
        Handle a job's completion. Free up the resources allocated to the job.
        """
        logging.info(f"t = {self.env.now}, job {job.short_print()} completed")
        for node_id, num_xpu in job.allocation.items():
            self.nodes[node_id].free(num_xpu)
        # TODO: this method is called when a job completes at the theorectical completion
        # time. The actual completion time may be ahead or behind if we model failures or
        # runtime dynamics. Need to refactor this class to handle such cases.

    def numNodes(self) -> int:
        """
        Return the number of nodes in the cluster.
        """
        return len(self.nodes)

    def allNodes(self) -> dict:
        """
        Return all nodes in the cluster.
        """
        return self.nodes

    def getIdleXPU(self, node_id: str) -> Union[int, float]:
        """
        Return the number of idle XPUs on the given node.
        """
        return self.nodes[node_id].numIdleXPU()
