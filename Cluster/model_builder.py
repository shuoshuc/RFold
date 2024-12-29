import json

from common.job import TopoType


def build_2d_model(
    name: str, topo: TopoType, dimx: int, dimy: int, xpu_per_node: int, output: str
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
    }
    for x in range(dimx):
        for y in range(dimy):
            node = {"name": f"x{x}-y{y}", "num_xpu": xpu_per_node, "coordinates": (x, y)}
            cluster["nodes"].append(node)
            # TODO: connect the nodes differently for mesh and torus.
    # If an output file is specified, dump to the json file. Otherwise, just return the
    # in-memory struct.
    if output:
        with open(output, "w") as f:
            json.dump(cluster, f, indent=4)
    return cluster


def build_3d_model(
    name: str,
    topo: TopoType,
    dimx: int,
    dimy: int,
    dimz: int,
    xpu_per_node: int,
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
    }
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                node = {
                    "name": f"x{x}-y{y}-z{z}",
                    "num_xpu": xpu_per_node,
                    "coordinates": (x, y, z),
                }
                cluster["nodes"].append(node)
    # If an output file is specified, dump the json file. Otherwise, just return the
    # in-memory struct.
    if output:
        with open(output, "w") as f:
            json.dump(cluster, f, indent=4)
    return cluster


def build_clos_model(
    name: str,
    num_node: int,
    num_t0: int,
    num_t1: int,
    xpu_per_node: int,
    output: str,
) -> dict:
    """
    Construct a Clos model.
    """
    # No need for num_t1 to find out the number of nodes in the cluster.
    tot_node = num_node * num_t0
    cluster = {
        "name": name,
        "topology": "CLOS",
        "dimx": num_node,
        "dimy": num_t0,
        "dimz": num_t1,
        "total_nodes": int(tot_node),
        "nodes": [],
    }
    for t in range(num_t0):
        for n in range(num_node):
            # The nodes are named as t0-n0, t0-n1, ...
            # The T0 switches are uniquely named as t0, t1, ...
            # The coordinates are (n, t). No need for s (tier-1 switch index) to
            # uniquely identify a node since T0 and T1 are connected as a Clos.
            node = {
                "name": f"t{t}-n{n}",
                "num_xpu": xpu_per_node,
                "coordinates": (n, t),
            }
            cluster["nodes"].append(node)
    # If an output file is specified, dump the json file. Otherwise, just return the
    # in-memory struct.
    if output:
        with open(output, "w") as f:
            json.dump(cluster, f, indent=4)
    return cluster


def build(
    topo: TopoType,
    name: str,
    dimension: tuple[int],
    xpu_per_node: int = 1,
    output: str = None,
) -> dict:
    """
    The builder function to construct a cluster topology.
    Specify the following parameters:
    - topo: enum topology type.
    - name: name of the cluster.
    - dimension: a tuple representing the dimension of the topology. If Torus, (x, y)
      for 2D, (x, y, z) for 3D. If CLOS, (n, t, s) for a 2-tier network, where n is the
      number of host under tier-0 switch, t is the number of tier-0 switch, s is the
      number of tier-1 switch.
      switch.
    - xpu_per_node: number of XPUs per node.
    - output: output file to dump the cluster topology
    """
    if not all(isinstance(i, int) and i > 0 for i in dimension):
        raise ValueError("Dimension must contain positive integers.")
    if xpu_per_node < 1:
        raise ValueError("At least 1 XPU per node is required.")

    if topo in (TopoType.T2D, TopoType.MESH2D):
        if len(dimension) != 2:
            raise ValueError("*2D: dimension must be a 2-tuple.")
        if xpu_per_node != 1:
            raise ValueError("*2D: only 1 XPU per node is supported.")
        (dimx, dimy) = dimension
        cluster = build_2d_model(
            name=name,
            topo=topo,
            dimx=dimx,
            dimy=dimy,
            xpu_per_node=xpu_per_node,
            output=output,
        )
        return cluster
    elif topo in (TopoType.T3D_NT, TopoType.T3D_T, TopoType.MESH3D):
        if len(dimension) != 3:
            raise ValueError("*3D: dimension must be a 3-tuple.")
        if xpu_per_node != 1:
            raise ValueError("*3D: only 1 XPU per node is supported.")
        (dimx, dimy, dimz) = dimension
        cluster = build_3d_model(
            name=name,
            topo=topo,
            dimx=dimx,
            dimy=dimy,
            dimz=dimz,
            xpu_per_node=xpu_per_node,
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
            output=output,
        )
        return cluster
    else:
        raise ValueError(f"Unknown topology: {topo}")
