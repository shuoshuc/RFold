import logging
from abc import ABC, abstractmethod
from typing import Union

from common.job import TopoType


class Port:
    """
    A port represents a physical network port on a node or switch, also one end
    of a physical link.
    - name: port name
    - orig_link: originating link of which this port is a source port.
    - term_link: terminating link of which this port is a destination port.
    - speed_gbps: speed the port is running at.
    - index: index of the port.
    """

    def __init__(
        self,
        name: str,
        orig_link: "Link" = None,
        term_link: "Link" = None,
        speed_gbps: float = None,
        index: int = None,
    ):
        self.name = name
        self.index = index
        self.orig_link = orig_link
        self.term_link = term_link
        self.speed_gbps = speed_gbps
        # parent node this port belongs to.
        self._parent_node = None

    def setParent(self, node: "Node"):
        self._parent_node = node

    def setOrigLink(self, orig_link: "Link"):
        self.orig_link = orig_link

    def setTermLink(self, term_link: "Link"):
        self.term_link = term_link

    def getParent(self) -> "Node":
        return self._parent_node


class Link:
    """
    A link represents a uni-directional link that connects 2 ports.
    - name: link name
    - src_port: source port of the link.
    - dst_port: destination port of the link.
    - speed_gbps: speed the link is running at (in Gbps).
    """

    def __init__(
        self,
        name: str,
        src_port: Port = None,
        dst_port: Port = None,
        speed_gbps: float = None,
    ):
        self.name = name
        self.src_port = src_port
        self.dst_port = dst_port
        self.speed_gbps = speed_gbps
        # Remaining available capacity on the link.
        self._residual = speed_gbps

    def resetResidual(self):
        self._residual = 0


class BaseNode(ABC):
    @abstractmethod
    def addPort(self, port: Port):
        pass


class Node(BaseNode):
    def __init__(
        self,
        name: str,
        num_xpu: int,
        topo: TopoType,
        coord: Union[tuple[int, int], tuple[int, int, int]],
    ):
        """
        name: FQDN of the node
        num_xpu: number of XPUs on the node
        topo: topology type of the node
        coord: coordinates of the node in the cluster
        """
        if num_xpu < 1:
            raise ValueError("Number of XPUs must be at least 1.")
        self.name = name
        self.num_xpu = num_xpu
        self.num_idle_xpu = num_xpu
        # Depending on topology type, coordinates either contain (pod id, node id),
        # or (x, y) / (x, y, z).
        self.pod_id = None
        self.node_id = None
        self.dimx = None
        self.dimy = None
        self.dimz = None
        self._unpackCoord(topo, coord)
        # List of member ports on the node.
        self._ports = []

    def _unpackCoord(
        self, topo: TopoType, coord: Union[tuple[int, int], tuple[int, int, int]]
    ):
        """
        Unpack the coordinates of the node based on the topology.
        """
        if topo == TopoType.CLOS:
            self.pod_id, self.node_id = coord
        elif topo in (TopoType.MESH2D, TopoType.T2D):
            self.dimx, self.dimy = coord
        elif topo in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
            self.dimx, self.dimy, self.dimz = coord
        else:
            logging.error(f"Unknown topology type: {topo}")

    def addPort(self, port: Port):
        """
        Add a port to the node.
        """
        self._ports.append(port)

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
        logging.debug(
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
        logging.debug(
            f"Node {self.name} state: {self.numIdleXPU()} / {self.numXPU()} XPUs."
        )


class Switch(BaseNode):
    """
    A switch is another type of node that has no XPU. It only exists in Clos topology.
    """

    def __init__(self, name: str, tier: int, coord: Union[tuple[int], tuple[int, int]]):
        self.name = name
        self.tier = tier
        # A switch can either be T0 (pod, t) or T1 (s,).
        self.pod_id, self.switch_id = coord if tier == 0 else (None, *coord)
        self._ports = []

    def addPort(self, port: Port):
        """
        Add a port to the node.
        """
        self._ports.append(port)
