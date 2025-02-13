import logging
from typing import Optional, Union

from common.flags import FLAGS
from common.job import TopoType
from common.utils import dump_spec


def connect_links(
    topo: TopoType,
    x: int,
    y: int,
    z: Optional[int],
    dimx: int,
    dimy: int,
    dimz: Optional[int],
    speed_gbps: float,
) -> list[Optional[dict]]:
    """
    Connect a node to its neighbors in a 2/3D mesh/torus.
    If 2D, z coordinate is set to None and should be ignored.
    Returns a list of 6 links in fixed order: [x-, x+, y-, y+, z-, z+].
    If a corresponding link is invalid, it is a None in the returned list.
    Port mapping in 3D mesh/torus:
        port x-, index 0
        port x+, index 1
        port y-, index 2
        port y+, index 3
        port z-, index 4
        port z+, index 5
    """

    def port_name_gen(x: int, y: int, z: int, p: int) -> str:
        if z is None:
            return f"x{x}-y{y}-p{p}"
        return f"x{x}-y{y}-z{z}-p{p}"

    # Internal helper function to build a link.
    def build_link(src: str, dst: str, speed_gbps: float) -> dict:
        return {
            "name": f"{src}:{dst}",
            "src": src,
            "dst": dst,
            "speed_gbps": speed_gbps,
        }

    if topo == TopoType.T3D_T:
        raise NotImplementedError("Twisted 3D topology is not supported.")

    connected_links = [None] * 6
    x_minus, x_plus = x - 1, x + 1
    y_minus, y_plus = y - 1, y + 1
    # Connect along the x-axis.
    if x_minus >= 0 or topo in (TopoType.T2D, TopoType.T3D_NT):
        connected_links[0] = build_link(
            src=port_name_gen(x, y, z, 0),
            dst=port_name_gen(x_minus % dimx, y, z, 1),
            speed_gbps=speed_gbps,
        )
    if x_plus < dimx or topo in (TopoType.T2D, TopoType.T3D_NT):
        connected_links[1] = build_link(
            src=port_name_gen(x, y, z, 1),
            dst=port_name_gen(x_plus % dimx, y, z, 0),
            speed_gbps=speed_gbps,
        )
    # Connect along the y-axis.
    if y_minus >= 0 or topo in (TopoType.T2D, TopoType.T3D_NT):
        connected_links[2] = build_link(
            src=port_name_gen(x, y, z, 2),
            dst=port_name_gen(x, y_minus % dimy, z, 3),
            speed_gbps=speed_gbps,
        )
    if y_plus < dimy or topo in (TopoType.T2D, TopoType.T3D_NT):
        connected_links[3] = build_link(
            src=port_name_gen(x, y, z, 3),
            dst=port_name_gen(x, y_plus % dimy, z, 2),
            speed_gbps=speed_gbps,
        )

    # No z coordinate set, this is a 2D mesh/torus. Finish here.
    if z is None:
        return connected_links

    z_minus, z_plus = z - 1, z + 1
    # Connect along the z-axis.
    if z_minus >= 0 or topo in (TopoType.T3D_NT,):
        connected_links[4] = build_link(
            src=port_name_gen(x, y, z, 4),
            dst=port_name_gen(x, y, z_minus % dimz, 5),
            speed_gbps=speed_gbps,
        )
    if z_plus < dimz or topo in (TopoType.T3D_NT,):
        connected_links[5] = build_link(
            src=port_name_gen(x, y, z, 5),
            dst=port_name_gen(x, y, z_plus % dimz, 4),
            speed_gbps=speed_gbps,
        )
    return connected_links


def zip_connect_links(
    port_list1: list[str],
    port_list2: list[str],
    speed_gbps: float,
) -> list[dict]:
    """
    Zip connect two lists of ports with link pairs (in both directions).
    It works like the zip() function.
    """
    if len(port_list1) != len(port_list2):
        logging.error(
            f"Port lists must have the same length: {len(port_list1)} != {len(port_list2)}"
        )
    links = []
    for p1, p2 in zip(port_list1, port_list2):
        link_p1_p2 = {
            "name": f"{p1}:{p2}",
            "src": p1,
            "dst": p2,
            "speed_gbps": speed_gbps,
        }
        link_p2_p1 = {
            "name": f"{p2}:{p1}",
            "src": p2,
            "dst": p1,
            "speed_gbps": speed_gbps,
        }
        links.extend([link_p1_p2, link_p2_p1])
    return links


def build_2d_model(
    name: str,
    topo: TopoType,
    dimx: int,
    dimy: int,
    xpu_per_node: int,
    port_per_node: int,
    port_speed_gbps: float,
    output: str,
) -> dict:
    """
    Construct a 2D mesh/torus model.
    """
    tot_xpu = dimx * dimy
    cluster = {
        "name": name,
        "topology": topo.name,
        "dimx": dimx,
        "dimy": dimy,
        "dimz": 0,
        "total_nodes": int(tot_xpu / xpu_per_node),
        "nodes": [],
        "links": [],
    }
    for x in range(dimx):
        for y in range(dimy):
            node = {
                "name": f"x{x}-y{y}",
                "num_xpu": xpu_per_node,
                "coordinates": (x, y),
                "ports": [],
            }
            for p in range(port_per_node):
                port = {
                    "name": f"x{x}-y{y}-p{p}",
                    "index": p,
                    "speed_gbps": port_speed_gbps,
                }
                node["ports"].append(port)
            cluster["nodes"].append(node)
            # Connect the node to its neighbors. Expect to get no more than 4 links.
            links = connect_links(
                topo=topo,
                x=x,
                y=y,
                z=None,
                dimx=dimx,
                dimy=dimy,
                dimz=None,
                speed_gbps=port_speed_gbps,
            )
            for link in links:
                if link:
                    cluster["links"].append(link)

    # If an output file is specified, dump to the json file. Otherwise, just return the
    # in-memory struct.
    dump_spec(cluster, output)
    return cluster


def build_3d_model(
    name: str,
    topo: TopoType,
    dimx: int,
    dimy: int,
    dimz: int,
    xpu_per_node: int,
    port_per_node: int,
    port_speed_gbps: float,
    output: str,
) -> dict:
    """
    Construct a 3D mesh/torus model.
    """
    tot_xpu = dimx * dimy * dimz
    cluster = {
        "name": name,
        "topology": topo.name,
        "dimx": dimx,
        "dimy": dimy,
        "dimz": dimz,
        "total_nodes": int(tot_xpu / xpu_per_node),
        "nodes": [],
        "links": [],
    }
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                node = {
                    "name": f"x{x}-y{y}-z{z}",
                    "num_xpu": xpu_per_node,
                    "coordinates": (x, y, z),
                    "ports": [],
                }
                for p in range(port_per_node):
                    port = {
                        "name": f"x{x}-y{y}-z{z}-p{p}",
                        "index": p,
                        "speed_gbps": port_speed_gbps,
                    }
                    node["ports"].append(port)
                cluster["nodes"].append(node)
                # Connect the node to its neighbors. Expect to get no more than 6 links.
                links = connect_links(
                    topo=topo,
                    x=x,
                    y=y,
                    z=z,
                    dimx=dimx,
                    dimy=dimy,
                    dimz=dimz,
                    speed_gbps=port_speed_gbps,
                )
                for link in links:
                    if link:
                        cluster["links"].append(link)

    # If an output file is specified, dump the json file. Otherwise, just return the
    # in-memory struct.
    dump_spec(cluster, output)
    return cluster


def build_clos_model(
    name: str,
    num_node: int,
    num_t0: int,
    num_t1: int,
    xpu_per_node: int,
    port_per_node: int,
    port_speed_gbps: float,
    t1_reserved: int,
    output: str,
) -> dict:
    """
    Construct a Clos model. Only 1-tier and 2-tier are supported.
    The definitions of input parameters are:
    - name: name of the cluster.
    - num_node: number of nodes in each pod.
    - num_t0: number of tier-0 (ToR) switches in each pod.
    - num_t1: number of tier-1 (spine) switches in the cluster.
    - xpu_per_node: number of XPUs per node.
    - port_per_node: number of ports per node.
    - port_speed_gbps: speed of each port in Gbps (same speed bi-directional).
    - t1_reserved: number of ports reserved on each T1 switch for uplinks.
    - output: output file to dump the cluster topology.

    NB: This function constructs a rail-optimized non-blocking Clos topology.
    A few assumptions are made:
    1. num_t0 == xpu_per_node == port_per_node.
    2. num_node == number of node-facing ports on each T0.
    3. num_t1 == number of T1-facing ports on each T0.
    4. T0 switches are 1:1 oversubscribed (# node-facing ports == # T1-facing ports).
    5. T0 switches and T1 switches are the same (i.e., radix, capacity).
    6. Each T0 has one link to each T1, a pod has `num_t0` links to each T1.

    A pod is a virtual concept, it is not explicitly modeled as the parent of nodes and
    T0 switches. Each node and T0 has a coordinate indicating the pod it belongs to.
    """
    # Total number of ports on each T0/T1 switch.
    radix = num_node * 2 if num_t1 == 0 else num_node + num_t1
    # 1-tier Clos is always 1 pod.
    num_pod = 1 if num_t1 == 0 else (num_node + num_t1 - t1_reserved) / num_t0
    if int(num_pod) != num_pod:
        raise ValueError("Number of pods must be an integer.")
    num_pod = int(num_pod)
    cluster = {
        "name": name,
        "topology": "CLOS",
        "dimx": num_node,
        "dimy": num_t0,
        "dimz": num_t1,
        "total_nodes": int(num_node * num_pod),
        "num_pods": num_pod,  # Clos-only field.
        "nodes": [],
        "tier0": [],  # Clos-only field.
        "tier1": [],  # Clos-only field.
        "links": [],
    }
    for pod in range(num_pod):
        for n in range(num_node):
            # The nodes are named as pod0-n0, pod0-n1, ...
            # The coordinates are (pod index, node index).
            node = {
                "name": f"pod{pod}-n{n}",
                "num_xpu": xpu_per_node,
                "coordinates": (pod, n),
                "ports": [],
            }
            # The ports are named as pod0-n0-p0, pod0-n0-p1, ...
            for p in range(port_per_node):
                port = {
                    "name": f"pod{pod}-n{n}-p{p}",
                    "index": p,
                    "speed_gbps": port_speed_gbps,
                }
                node["ports"].append(port)
            cluster["nodes"].append(node)
        for t in range(num_t0):
            # The T0 switches are named as pod0-t0, pod0-t1, ...
            t0 = {
                "name": f"pod{pod}-t{t}",
                "coordinates": (pod, t),
                "tier": 0,
                "ports": [],
            }
            for p in range(radix):
                # The ports are named as pod0-t0-p0, pod0-t0-p1, ...
                # Even indices (e.g., 0, 2, 4) are node-facing, odd indices
                # (e.g., 1, 3, 5) are T1-facing. If 1-tier, odd ports are unconnected.
                port = {
                    "name": f"pod{pod}-t{t}-p{p}",
                    "index": p,
                    "speed_gbps": port_speed_gbps,
                }
                t0["ports"].append(port)
            cluster["tier0"].append(t0)

            # Connect the T0 switch to the nodes in the same pod.
            # p0 on all nodes connect to switch t0, p1 on all nodes connect to t1, ...
            # Note that order of the ports is important.
            t0_ports = [port["name"] for port in t0["ports"] if port["index"] % 2 == 0]
            node_ports = [f"pod{pod}-n{i}-p{t}" for i in range(num_node)]
            cluster["links"].extend(
                zip_connect_links(t0_ports, node_ports, port_speed_gbps)
            )

    for s in range(num_t1):
        # The T1 switches are uniquely named as s0, s1, ...
        t1 = {
            "name": f"s{s}",
            "coordinates": (s,),
            "tier": 1,
            "ports": [],
        }
        for p in range(radix):
            # The ports are named as s0-p0, s0-p1, ...
            # All ports are T0-facing.
            port = {
                "name": f"s{s}-p{p}",
                "index": p,
                "speed_gbps": port_speed_gbps,
            }
            t1["ports"].append(port)
        cluster["tier1"].append(t1)

        # Connect the T1 switch to the T0 switches in each pod.
        # p1 on all T0 connect to switch s0, p3 on all T0 connect to s1, ...
        # Note that order of the ports is important.
        t1_ports = [port["name"] for port in t1["ports"]]
        t0_ports = [
            f"pod{pd}-t{j}-p{2 * s + 1}" for pd in range(num_pod) for j in range(num_t0)
        ]
        cluster["links"].extend(zip_connect_links(t1_ports, t0_ports, port_speed_gbps))

    # If an output file is specified, dump the json file. Otherwise, just return the
    # in-memory struct.
    dump_spec(cluster, output)
    return cluster


def build(
    topo: TopoType,
    name: str,
    dimension: Union[tuple[int, int], tuple[int, int, int]],
    xpu_per_node: int,
    port_per_node: int,
    port_speed_gbps: float,
    output: str = None,
    t1_reserved: int = FLAGS.t1_reserved_ports,
) -> dict:
    """
    The builder function to construct a cluster topology.
    Specify the following parameters:
    - topo: enum topology type.
    - name: name of the cluster.
    - dimension: a tuple representing the dimension of the topology. If Torus, (x, y)
      for 2D, (x, y, z) for 3D. If CLOS, (n, t, s) for a 2-tier network, where n is the
      number of nodes in a pod, t is the number of tier-0 switches in a pod, s is the
      number of tier-1 switches in the cluster.
    - xpu_per_node: number of XPUs per node.
    - port_per_node: number of ports per node.
    - port_speed_gbps: speed of each port in Gbps (same speed bi-directional).
    - output: output file to dump the cluster topology
    - t1_reserved (Clos-only param): number of ports reserved on each T1 for uplinks.
    """
    if not all(isinstance(i, int) and i > 0 for i in dimension):
        raise ValueError("Dimension must contain positive integers.")
    if xpu_per_node < 1:
        raise ValueError("At least 1 XPU per node is required.")
    if port_per_node < 1:
        raise ValueError("At least 1 port per node is required.")
    if port_speed_gbps < 0:
        raise ValueError("Port speed cannot be negative.")

    if topo in (TopoType.T2D, TopoType.MESH2D):
        if len(dimension) != 2:
            raise ValueError("*2D: dimension must be a 2-tuple.")
        if xpu_per_node != 1:
            raise ValueError("*2D: only 1 XPU per node is supported.")
        if port_per_node != 4:
            raise ValueError("*2D: each node should have 4 ports.")
        (dimx, dimy) = dimension
        cluster = build_2d_model(
            name=name,
            topo=topo,
            dimx=dimx,
            dimy=dimy,
            xpu_per_node=xpu_per_node,
            port_per_node=port_per_node,
            port_speed_gbps=port_speed_gbps,
            output=output,
        )
        return cluster
    elif topo in (TopoType.T3D_NT, TopoType.T3D_T, TopoType.MESH3D):
        if len(dimension) != 3:
            raise ValueError("*3D: dimension must be a 3-tuple.")
        if xpu_per_node != 1:
            raise ValueError("*3D: only 1 XPU per node is supported.")
        if port_per_node != 6:
            raise ValueError("*3D: each node should have 6 ports.")
        (dimx, dimy, dimz) = dimension
        cluster = build_3d_model(
            name=name,
            topo=topo,
            dimx=dimx,
            dimy=dimy,
            dimz=dimz,
            xpu_per_node=xpu_per_node,
            port_per_node=port_per_node,
            port_speed_gbps=port_speed_gbps,
            output=output,
        )
        return cluster
    elif topo == TopoType.CLOS:
        # Only 1-tier and 2-tier Clos are currently supported.
        if len(dimension) != 2 and len(dimension) != 3:
            raise NotImplementedError(f"Dimension not supported in Clos.")
        num_node, num_t0, num_t1 = dimension if len(dimension) == 3 else (*dimension, 0)
        cluster = build_clos_model(
            name=name,
            num_node=num_node,
            num_t0=num_t0,
            num_t1=num_t1,
            xpu_per_node=xpu_per_node,
            port_per_node=port_per_node,
            port_speed_gbps=port_speed_gbps,
            t1_reserved=t1_reserved,
            output=output,
        )
        return cluster
    else:
        raise ValueError(f"Unknown topology: {topo}")
