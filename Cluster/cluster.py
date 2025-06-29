import logging
import simpy
import numpy as np
from itertools import product
from math import ceil
from numpy.typing import NDArray
from typing import Optional, Union

from common.flags import FLAGS
from common.job import Job, TopoType
from common.utils import viz3D
from Cluster.topology import Port, Link, Node, Switch


class Cluster:
    """
    A cluster is the high-level abstraction of the resources.
    It contains a collection of nodes, and also models the network topology, link
    bandwidth, etc. The cluster manager and other external components interact with
    this class to access the resources. This class also exposes certain states for
    monitoring purposes.
    """

    def __init__(self, env: simpy.core.Environment, spec: dict, rsize: int = FLAGS.rsize):
        """
        env: simpy environment.
        spec: a parsed JSON object containing the cluster spec.
        """
        self.env = env
        if not spec:
            raise ValueError("No cluster spec provided.")

        # A map from node name to node object.
        self.nodes: dict[str, Node] = {}
        # A map from block coordinate to a list of node objects in that block.
        self.blocks: dict[Union[tuple[int, int], tuple[int, int, int]], list[Node]] = {}
        # A map from port name to port object.
        self.ports: dict[str, Port] = {}
        # A map from link name to link object.
        self.links: dict[str, Link] = {}
        # (Clos only) A map from pod ID to a map of node objects.
        self.pods: dict[int, dict[str, Node]] = {}
        # (Clos only) A map from switch name to T0 switch object.
        self.tier0s: dict[str, Switch] = {}
        # (Clos only) A map from switch name to T1 switch object.
        self.tier1s: dict[str, Switch] = {}

        # ----- start parsing the cluster spec -----
        self.name = spec["name"]
        self.topo = TopoType[spec["topology"]]
        # x, y, z together describe the shape of the cluster.
        # For Clos, x, y, z = n, t, s respectively.
        self.dimx = spec["dimx"]
        self.dimy = spec["dimy"]
        self.dimz = spec["dimz"]
        # Compute the number of bits per dimension for Hilbert curve.
        # For non-power-of-2 and irregular shapes, take the max and round up.
        # E.g., for a 4x4x5 3D torus, max # nodes per dimension is 5.
        # Round it to the nearest power-of-2, we get 8. So bits per dimension is log2(8)=3.
        self.bits_per_dim = ceil(np.log2(max(self.dimx, self.dimy, self.dimz)))
        # Clos-only field.
        if self.topo == TopoType.CLOS:
            self.num_pods = spec["num_pods"]
            for pod_id in range(self.num_pods):
                self.pods[pod_id] = {}

        # Iterate through the nodes and ports.
        for n in spec["nodes"]:
            node_obj = Node(
                name=n["name"],
                num_xpu=n["num_xpu"],
                topo=self.topo,
                coord=n["coordinates"],
                bits_per_dim=self.bits_per_dim,
            )
            self.nodes[node_obj.name] = node_obj
            # Group the nodes into reconfigurable blocks.
            block_coord = [
                dim // rsize
                for dim in [node_obj.dimx, node_obj.dimy, node_obj.dimz]
                if dim is not None
            ]
            self.blocks.setdefault(tuple(block_coord), []).append(node_obj)
            # (Clos-only) group nodes into pods.
            if self.topo == TopoType.CLOS:
                self.pods[node_obj.pod_id][node_obj.name] = node_obj
            for p in n["ports"]:
                port_obj = Port(
                    name=p["name"], speed_gbps=p["speed_gbps"], index=p["index"]
                )
                self.ports[port_obj.name] = port_obj
                node_obj.addPort(port_obj)
                port_obj.setParent(node_obj)
        # --- Clos-only parsing below. ---
        # Iterate through the switches and ports.
        if self.topo == TopoType.CLOS:
            for t0 in spec["tier0"]:
                t0_obj = Switch(name=t0["name"], tier=t0["tier"], coord=t0["coordinates"])
                self.tier0s[t0_obj.name] = t0_obj
                for p in t0["ports"]:
                    port_obj = Port(
                        name=p["name"], speed_gbps=p["speed_gbps"], index=p["index"]
                    )
                    self.ports[port_obj.name] = port_obj
                    t0_obj.addPort(port_obj)
                    port_obj.setParent(t0_obj)
            for s in spec["tier1"]:
                s_obj = Switch(name=s["name"], tier=s["tier"], coord=s["coordinates"])
                self.tier1s[s_obj.name] = s_obj
                for p in s["ports"]:
                    port_obj = Port(
                        name=p["name"], speed_gbps=p["speed_gbps"], index=p["index"]
                    )
                    self.ports[port_obj.name] = port_obj
                    s_obj.addPort(port_obj)
                    port_obj.setParent(s_obj)
        # --- Clos-only parsing end. ---
        # Iterate through the links.
        for l in spec["links"]:
            src_port = self.ports[l["src"]]
            dst_port = self.ports[l["dst"]]
            link_obj = Link(
                name=l["name"],
                src_port=src_port,
                dst_port=dst_port,
                speed_gbps=l["speed_gbps"],
            )
            self.links[link_obj.name] = link_obj
            src_port.setOrigLink(link_obj)
            dst_port.setTermLink(link_obj)
        # ----- finish parsing the cluster spec -----

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
        for node_id, num_xpu in job.allocation.items():
            self.nodes[node_id].alloc(num_xpu)

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

    def totalIdleXPU(self) -> Union[int, float]:
        """
        Return the total number of idle XPUs in the cluster.
        """
        # TODO: only count idle XPUs on idle nodes.
        return sum([n.numIdleXPU() for n in self.nodes.values()])

    def totalIdleNodes(self) -> int:
        """
        Return the total number of idle nodes in the cluster.
        """
        # TODO: cache the idle nodes.
        return len([n for n in self.nodes.values() if n.numIdleXPU() > 0])

    def toBlockArray(
        self, nodes: list[Node], rsize: int = FLAGS.rsize
    ) -> NDArray[np.float64]:
        """
        Return a 2D/3D array representation of the reconfigurable block availability.
        Each element corresponds to the number of idle XPUs on a node.
        Note: this method only works for 2D/3D mesh/torus topology.
        """
        if self.topo in [TopoType.MESH2D, TopoType.T2D]:
            array = np.zeros((rsize, rsize))
            for node in nodes:
                array[node.dimx % rsize, node.dimy % rsize] = node.numIdleXPU()
            return array
        elif self.topo in [TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T]:
            array = np.zeros((rsize, rsize, rsize))
            for node in nodes:
                array[
                    node.dimx % rsize,
                    node.dimy % rsize,
                    node.dimz % rsize,
                ] = node.numIdleXPU()
            return array
        else:
            raise TypeError(f"Topology {self.topo} is not supported.")

    def toArray(self) -> NDArray[np.float64]:
        """
        Return a 2D/3D array representation of the node/xpu availability.
        Each element corresponds to the number of idle XPUs on a node.
        Note: this method only works for 2D/3D mesh/torus topology.
        """
        if self.topo in [TopoType.MESH2D, TopoType.T2D]:
            array = np.zeros((self.dimx, self.dimy))
            for node in self.nodes.values():
                array[node.dimx, node.dimy] = node.numIdleXPU()
            return array
        elif self.topo in [TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T]:
            array = np.zeros((self.dimx, self.dimy, self.dimz))
            for node in self.nodes.values():
                array[node.dimx, node.dimy, node.dimz] = node.numIdleXPU()
            return array
        else:
            raise TypeError(f"Topology {self.topo} is not supported.")

    def linearAvail(self) -> NDArray[np.float64]:
        """
        Return a 1D array representation of the node/xpu availability.
        Nodes are linearized (sorted) based on their Hilbert index.
        """
        array = []
        for node in self.nodes.values():
            index, avail = node.getHilbertIndex(), node.numIdleXPU()
            # Note that valid Hilbert index can be 0.
            if index is None:
                raise ValueError(
                    "Hilbert index not set for node or topology not supported."
                )

            # Construct the array of availability. For each node, if its idle XPUs
            # are greater than 1, replicate the Hilbert index multiple times in the array.
            array.extend([index] * avail)

        return np.array(sorted(array))

    def failNodes(self, node_names: list[str]):
        """
        Fail the nodes in the cluster by setting their availability to 0.
        """
        for node_name in node_names:
            if node_name not in self.nodes:
                logging.warning(f"Failed node {node_name} not found in the cluster.")
                continue
            self.nodes[node_name].num_idle_xpu = 0

    def visualize(self):
        """
        Visualize the cluster state.
        """
        if self.topo in (TopoType.MESH2D, TopoType.T2D):
            array = np.rot90(self.toArray())
            logging.info(f"Cluster state:")
            for row in array:
                logging.info(" ".join(map(str, map(int, row))))
            logging.info("")
        elif self.topo in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
            viz3D(self.dimx, self.dimy, self.dimz, self.toArray())

    # --------------------------------------------------
    # Graph-query type of methods for entity lookup.
    # --------------------------------------------------
    def getPortByName(self, port_name: str) -> Optional[Port]:
        """
        Find the port object by its name. Return None if not found.
        """
        if port_name not in self.ports:
            return None
        return self.ports[port_name]

    def getNodeByName(self, node_name: str) -> Optional[Node]:
        """
        Find the node object by its name. Return None if not found.
        """
        if node_name not in self.nodes:
            return None
        return self.nodes[node_name]

    def findNodeOfPort(self, port_name: str) -> Optional[Node]:
        """
        Find the parent node object by the child port name. Return None if not found.
        """
        if port_name not in self.ports:
            return None
        return self.ports[port_name].getParent()

    def findPeerPortOfPort(self, port_name: str) -> Optional[Port]:
        """
        Find the peer port object by port name. Return None if not found.
        """
        if port_name not in self.ports:
            return None
        port = self.ports[port_name]
        if port.orig_link is None:
            return None
        return port.orig_link.dst_port

    def findLinksBetweenNodes(self, src_node: Node, dst_node: Node) -> list[Link]:
        """
        Find all direct links from the source node to the destination node.
        Node can also be a switch.
        """
        links = []
        src_port_names = [p.name for p in src_node._ports]
        dst_port_names = [p.name for p in dst_node._ports]
        for src, dst in list(product(src_port_names, dst_port_names)):
            link_name = f"{src}:{dst}"
            if link_name in self.links:
                links.append(self.links[link_name])
        return links

    def hasPort(self, port_name: str) -> bool:
        """
        Return True if the port exists in the cluster.
        """
        return port_name in self.ports

    def hasNode(self, node_name: str) -> bool:
        """
        Return True if the node exists in the cluster.
        """
        return node_name in self.nodes

    def hasLink(self, src_port: str, dst_port: str) -> bool:
        """
        Return True if the link exists in the cluster.
        """
        return f"{src_port}:{dst_port}" in self.links

    # Clos-only methods.
    def hasT0(self, t0_name: str) -> bool:
        """
        Return True if the T0 switch exists in the cluster.
        """
        if self.topo != TopoType.CLOS:
            return False
        return t0_name in self.tier0s

    def hasT1(self, t1_name: str) -> bool:
        """
        Return True if the T1 switch exists in the cluster.
        """
        if self.topo != TopoType.CLOS:
            return False
        return t1_name in self.tier1s

    def hasNodeInPod(self, node_name: str, pod_id: int) -> bool:
        """
        Return True if the node exists in the given pod.
        """
        if self.topo != TopoType.CLOS:
            return False
        return node_name in self.pods[pod_id]

    def getT0ByName(self, t0_name: str) -> Optional[Switch]:
        """
        Find the T0 switch object by its name. Return None if not found.
        """
        if t0_name not in self.tier0s:
            return None
        return self.tier0s[t0_name]

    def getT1ByName(self, t1_name: str) -> Optional[Switch]:
        """
        Find the T1 switch object by its name. Return None if not found.
        """
        if t1_name not in self.tier1s:
            return None
        return self.tier1s[t1_name]
