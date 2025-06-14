import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


DEFAULT_COLOR = "silver"


def draw_split_cuboid(
    ax,
    dims=(4, 8, 4),
    split_info=None,
    highlighted_nodes=None,
    non_highlighted_edges=[],
):
    """
    Draws a cuboid, with a visual gap, and ensures correctly colored nodes without any overlap.
    """
    if highlighted_nodes is None:
        highlighted_nodes = {}

    Nx, Ny, Nz = dims
    # Create the grid points. These represent the "logical" coordinates.
    x_logical, y_logical, z_logical = np.mgrid[0:Nx, 0:Ny, 0:Nz]

    # Create a separate copy of the coordinates for plotting.
    x_plot, y_plot, z_plot = (
        x_logical.astype(float),
        y_logical.astype(float),
        z_logical.astype(float),
    )

    # --- 1. Apply Coordinate Gap Transformation to the Plotting Coordinates ---
    gap_size = 0
    if split_info and split_info["axis"] == "y":
        split_index = split_info["index"]
        gap_size = split_info["gap"]
        # Modify ONLY the plotting coordinates, leaving the logical ones intact.
        y_plot[y_plot >= split_index] += gap_size

    # --- 2. Identify and Separate Nodes Before Plotting (The Robust Fix) ---
    is_visible_surface = (x_logical == 0) | (y_logical == 0) | (z_logical == Nz - 1)

    # Prepare lists to hold the PLOT coordinates for each group
    standard_nodes_coords = []

    # Get the set of nodes to be highlighted for fast lookups
    highlight_keys = set(highlighted_nodes.keys())

    # Iterate through all nodes in the grid
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Check if the current node is on a visible surface
                if is_visible_surface[i, j, k]:
                    logical_coord = (
                        x_logical[i, j, k],
                        y_logical[i, j, k],
                        z_logical[i, j, k],
                    )
                    plot_coord = (x_plot[i, j, k], y_plot[i, j, k], z_plot[i, j, k])

                    # If this node is NOT in the highlight list, add its PLOT coordinates
                    # to the standard list. Otherwise, do nothing with it here.
                    if logical_coord not in highlight_keys:
                        standard_nodes_coords.append(plot_coord)

    # --- 3. Plot the Separated Node Groups ---

    # Plot ONLY the standard (non-highlighted) nodes in blue.
    if standard_nodes_coords:
        std_x, std_y, std_z = zip(*standard_nodes_coords)
        ax.scatter(std_x, std_y, std_z, c=DEFAULT_COLOR, s=100, alpha=1.0)

    # Plot ONLY the highlighted nodes with their custom colors.
    for (hx, hy, hz), color in highlighted_nodes.items():
        # This check is technically redundant now but good for safety
        if (hx == 0) or (hy == 0) or (hz == Nz - 1):
            # Calculate the plot coordinate for this specific highlighted node
            plot_y_highlight = float(hy)
            if split_info and split_info["axis"] == "y" and hy >= split_info["index"]:
                plot_y_highlight += gap_size
            # Draw the single, correctly-colored circle.
            ax.scatter(hx, plot_y_highlight, hz, c=color, s=100, alpha=1.0, zorder=10)

    # --- 4. Connect All Nodes Using Lines ---
    def check_visible_surface(pi, pj, pk):
        return (pi == 0) or (pj == 0) or (pk == Nz - 1)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Check connection in X direction
                if i + 1 < Nx:
                    # Check if the line is on the visible surface
                    if check_visible_surface(i, j, k) and check_visible_surface(
                        i + 1, j, k
                    ):
                        coord1 = (i, j, k)
                        coord2 = (i + 1, j, k)
                        # Check if both endpoints are highlighted with the same color
                        if (
                            coord1 in highlight_keys
                            and coord2 in highlight_keys
                            and highlighted_nodes[coord1] == highlighted_nodes[coord2]
                            and sorted([coord1, coord2]) not in non_highlighted_edges
                        ):
                            line_color = highlighted_nodes[coord1]
                            line_width = 2
                            z_ord = 15
                        else:
                            line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 1
                        ax.plot(
                            x_plot[i : i + 2, j, k],
                            y_plot[i : i + 2, j, k],
                            z_plot[i : i + 2, j, k],
                            color=line_color,
                            lw=line_width,
                            zorder=z_ord,
                        )

                # Check connection in Y direction
                if j + 1 < Ny:
                    if check_visible_surface(i, j, k) and check_visible_surface(
                        i, j + 1, k
                    ):
                        coord1 = (i, j, k)
                        coord2 = (i, j + 1, k)
                        if (
                            coord1 in highlight_keys
                            and coord2 in highlight_keys
                            and highlighted_nodes[coord1] == highlighted_nodes[coord2]
                            and sorted([coord1, coord2]) not in non_highlighted_edges
                        ):
                            line_color = highlighted_nodes[coord1]
                            line_width = 2
                            z_ord = 15
                        else:
                            line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 1
                        ax.plot(
                            x_plot[i, j : j + 2, k],
                            y_plot[i, j : j + 2, k],
                            z_plot[i, j : j + 2, k],
                            color=line_color,
                            lw=line_width,
                            zorder=z_ord,
                        )

                # Check connection in Z direction
                if k + 1 < Nz:
                    if check_visible_surface(i, j, k) and check_visible_surface(
                        i, j, k + 1
                    ):
                        coord1 = (i, j, k)
                        coord2 = (i, j, k + 1)
                        if (
                            coord1 in highlight_keys
                            and coord2 in highlight_keys
                            and highlighted_nodes[coord1] == highlighted_nodes[coord2]
                            and sorted([coord1, coord2]) not in non_highlighted_edges
                        ):
                            line_color = highlighted_nodes[coord1]
                            line_width = 2
                            z_ord = 15
                        else:
                            line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 1
                        ax.plot(
                            x_plot[i, j, k : k + 2],
                            y_plot[i, j, k : k + 2],
                            z_plot[i, j, k : k + 2],
                            color=line_color,
                            lw=line_width,
                            zorder=z_ord,
                        )

    # --- 5. Clean up the view ---
    ax.axis("off")
    ax.set_box_aspect((Nx - 1, (Ny - 1) + gap_size, Nz - 1), zoom=0.9)
    ax.view_init(elev=23, azim=-155)


def draw_split_cuboid_3d(
    ax,
    dims=(4, 8, 4),
    split_info=None,
    highlighted_nodes=None,
    non_highlighted_edges=[],
):
    """
    Draws a cuboid, with a visual gap, and ensures correctly colored nodes without any overlap.
    """
    if highlighted_nodes is None:
        highlighted_nodes = {}

    Nx, Ny, Nz = dims
    # Create the grid points. These represent the "logical" coordinates.
    x_logical, y_logical, z_logical = np.mgrid[0:Nx, 0:Ny, 0:Nz]

    # Create a separate copy of the coordinates for plotting.
    x_plot, y_plot, z_plot = (
        x_logical.astype(float),
        y_logical.astype(float),
        z_logical.astype(float),
    )

    # --- 1. Apply Coordinate Gap Transformation to the Plotting Coordinates ---
    gap_size = 0
    if split_info and split_info["axis"] == "y":
        split_index = split_info["index"]
        gap_size = split_info["gap"]
        # Modify ONLY the plotting coordinates, leaving the logical ones intact.
        y_plot[y_plot >= split_index] += gap_size

    # --- 2. Identify and Separate Nodes Before Plotting (The Robust Fix) ---
    is_visible_surface = (x_logical == 0) | (y_logical == 0) | (z_logical == Nz - 1)

    # Prepare lists to hold the PLOT coordinates for each group
    standard_nodes_coords = []

    # Get the set of nodes to be highlighted for fast lookups
    highlight_keys = set(highlighted_nodes.keys())

    # Iterate through all nodes in the grid
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Check if the current node is on a visible surface
                if is_visible_surface[i, j, k]:
                    logical_coord = (
                        x_logical[i, j, k],
                        y_logical[i, j, k],
                        z_logical[i, j, k],
                    )
                    plot_coord = (x_plot[i, j, k], y_plot[i, j, k], z_plot[i, j, k])

                    # If this node is NOT in the highlight list, add its PLOT coordinates
                    # to the standard list. Otherwise, do nothing with it here.
                    if logical_coord not in highlight_keys:
                        standard_nodes_coords.append(plot_coord)

    # --- 3. Plot the Separated Node Groups ---

    # Plot ONLY the standard (non-highlighted) nodes in blue.
    if standard_nodes_coords:
        std_x, std_y, std_z = zip(*standard_nodes_coords)
        ax.scatter(std_x, std_y, std_z, c=DEFAULT_COLOR, s=100, alpha=1.0, zorder=1)

    # Plot ONLY the highlighted nodes with their custom colors.
    for (hx, hy, hz), color in highlighted_nodes.items():
        # This check is technically redundant now but good for safety
        if (hx == 0) or (hy == 0) or (hz == Nz - 1):
            # Calculate the plot coordinate for this specific highlighted node
            plot_y_highlight = float(hy)
            if split_info and split_info["axis"] == "y" and hy >= split_info["index"]:
                plot_y_highlight += gap_size
            # Draw the single, correctly-colored circle.
            ax.scatter(hx, plot_y_highlight, hz, c=color, s=100, alpha=1.0, zorder=1)

    # --- 4. Connect All Nodes Using Lines ---
    def check_visible_surface(pi, pj, pk):
        return (pi == 0) or (pj == 0) or (pk == Nz - 1)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Check connection in X direction
                if i + 1 < Nx:
                    # Check if the line is on the visible surface
                    if check_visible_surface(i, j, k) and check_visible_surface(
                        i + 1, j, k
                    ):
                        coord1 = (i, j, k)
                        coord2 = (i + 1, j, k)
                        # Check if both endpoints are highlighted with the same color
                        if (
                            coord1 in highlight_keys
                            and coord2 in highlight_keys
                            and highlighted_nodes[coord1] == highlighted_nodes[coord2]
                            and sorted([coord1, coord2]) not in non_highlighted_edges
                        ):
                            line_color = highlighted_nodes[coord1]
                            line_width = 2
                            z_ord = 15
                        else:
                            line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 1
                        ax.plot(
                            x_plot[i : i + 2, j, k],
                            y_plot[i : i + 2, j, k],
                            z_plot[i : i + 2, j, k],
                            color=line_color,
                            lw=line_width,
                            zorder=z_ord,
                        )

                # Check connection in Y direction
                if j + 1 < Ny:
                    if check_visible_surface(i, j, k) and check_visible_surface(
                        i, j + 1, k
                    ):
                        coord1 = (i, j, k)
                        coord2 = (i, j + 1, k)
                        if (
                            coord1 in highlight_keys
                            and coord2 in highlight_keys
                            # and highlighted_nodes[coord1] == highlighted_nodes[coord2]
                            and sorted([coord1, coord2]) not in non_highlighted_edges
                        ):
                            line_color = highlighted_nodes[coord2]
                            line_width = 2
                            z_ord = 5
                        else:
                            line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 1
                        ax.plot(
                            x_plot[i, j : j + 2, k],
                            y_plot[i, j : j + 2, k],
                            z_plot[i, j : j + 2, k],
                            color=line_color,
                            lw=line_width,
                            zorder=z_ord,
                        )

                # Check connection in Z direction
                if k + 1 < Nz:
                    if check_visible_surface(i, j, k) and check_visible_surface(
                        i, j, k + 1
                    ):
                        coord1 = (i, j, k)
                        coord2 = (i, j, k + 1)
                        if (
                            coord1 in highlight_keys
                            and coord2 in highlight_keys
                            and highlighted_nodes[coord1] == highlighted_nodes[coord2]
                            and sorted([coord1, coord2]) not in non_highlighted_edges
                        ):
                            line_color = highlighted_nodes[coord1]
                            if sorted([coord1, coord2]) in [
                                sorted([(0, 1, 1), (0, 1, 2)]),
                                sorted([(0, 2, 1), (0, 2, 2)]),
                            ]:
                                line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 15
                        else:
                            line_color = DEFAULT_COLOR
                            line_width = 2
                            z_ord = 1
                        ax.plot(
                            x_plot[i, j, k : k + 2],
                            y_plot[i, j, k : k + 2],
                            z_plot[i, j, k : k + 2],
                            color=line_color,
                            lw=line_width,
                            zorder=z_ord,
                        )

    # --- 5. Clean up the view ---
    ax.axis("off")
    ax.set_box_aspect((Nx - 1, (Ny - 1) + gap_size, Nz - 1), zoom=0.9)
    ax.view_init(elev=23, azim=-155)


def draw_elliptical_arrow_3d(
    ax,
    center,
    radius_a,
    radius_b,
    start_angle,
    end_angle,
    plane="xz",
    plane_val=0,
    clockwise=True,
    **kwargs
):
    """
    Draws a 3D elliptical arrow on the given axes.
    """
    # 1. Generate the points for the elliptical arc body
    theta = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), 100)

    # The logic is updated to use two different radii for an ellipse
    if plane == "xz":
        x = center[0] + radius_a * np.cos(theta)
        z = center[1] + radius_b * np.sin(theta)
        y = np.full_like(x, plane_val)
    elif plane == "xy":
        x = center[0] + radius_a * np.cos(theta)
        y = center[1] + radius_b * np.sin(theta)
        z = np.full_like(x, plane_val)
    elif plane == "yz":
        y = center[0] + radius_a * np.cos(theta)
        z = center[1] + radius_b * np.sin(theta)
        x = np.full_like(y, plane_val)

    # 2. Plot the arc body
    ax.plot(x, y, z, **kwargs)

    # 3. Add the arrowhead using quiver
    if clockwise:
        loc1 = -1
        loc2 = -2
    else:
        loc1 = 0
        loc2 = 1
    x_head, y_head, z_head = x[loc1], y[loc1], z[loc1]
    dx, dy, dz = x[loc1] - x[loc2], y[loc1] - y[loc2], z[loc1] - z[loc2]

    ax.quiver(
        x_head,
        y_head,
        z_head,
        dx,
        dy,
        dz,
        length=0.4,
        arrow_length_ratio=0.9,
        normalize=True,
        **kwargs,
    )


def plot_folding_1d2d(ax):
    cuboid_dims = (4, 8, 4)
    split_parameters = {"axis": "y", "index": 4, "gap": 1.1}
    nodes_to_highlight = {}
    # Red color for 4x6x1 job.
    for x in range(1):
        for y in range(2, 8):
            for z in range(4):
                nodes_to_highlight[(x, y, z)] = "#1f77b4"
    # Blue color for 2x3x4 job.
    for x in range(4):
        for y in range(2):
            for z in range(3):
                nodes_to_highlight[(x, y, z)] = "#ff7f0e"
    # 1D snaking job.
    nodes_to_highlight[(0, 0, 3)] = "#2ca02c"
    nodes_to_highlight[(1, 0, 3)] = "#2ca02c"
    nodes_to_highlight[(2, 0, 3)] = "#2ca02c"
    nodes_to_highlight[(0, 1, 3)] = "#2ca02c"
    nodes_to_highlight[(1, 1, 3)] = "#2ca02c"
    nodes_to_highlight[(2, 1, 3)] = "#2ca02c"
    nodes_to_highlight[(1, 2, 3)] = "#2ca02c"
    nodes_to_highlight[(2, 2, 3)] = "#2ca02c"
    nodes_to_highlight[(3, 2, 3)] = "#2ca02c"
    nodes_to_highlight[(1, 3, 3)] = "#2ca02c"
    nodes_to_highlight[(3, 3, 3)] = "#2ca02c"
    nodes_to_highlight[(1, 4, 3)] = "#2ca02c"
    nodes_to_highlight[(3, 4, 3)] = "#2ca02c"
    nodes_to_highlight[(1, 5, 3)] = "#2ca02c"
    nodes_to_highlight[(2, 5, 3)] = "#2ca02c"
    nodes_to_highlight[(3, 5, 3)] = "#2ca02c"
    nodes_to_highlight[(2, 6, 3)] = "#2ca02c"
    nodes_to_highlight[(3, 6, 3)] = "#2ca02c"

    # Non-highlighted edges to avoid double lines
    non_highlighted_edges = [
        sorted([(1, 0, 3), (1, 1, 3)]),
        sorted([(1, 1, 3), (2, 1, 3)]),
        sorted([(1, 2, 3), (2, 2, 3)]),
        sorted([(2, 5, 3), (3, 5, 3)]),
    ]

    draw_split_cuboid(
        ax,
        dims=cuboid_dims,
        split_info=split_parameters,
        highlighted_nodes=nodes_to_highlight,
        non_highlighted_edges=non_highlighted_edges,
    )
    ax.set_xlim([0, cuboid_dims[0] - 1])
    ax.set_ylim([-0.4, cuboid_dims[1] - 1 + split_parameters["gap"]])
    ax.set_zlim([0, cuboid_dims[2] - 1])

    draw_elliptical_arrow_3d(
        ax,
        center=(0.5, 0.9),  # (y, z) center of the ellipse
        radius_a=0.23,  # Radius in the y-direction
        radius_b=0.7,  # Radius in the z-direction
        start_angle=120,
        end_angle=380,
        plane="yz",  # Draw in a plane parallel to the YZ plane
        plane_val=0,  # Position the plane on the side face (x=0)
        color="black",  # Use the same orange color
        lw=1.5,
        zorder=20,
    )
    # y-axis arrow
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [6.5, 3],
            [-0.4, -0.4],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    # z-axis arrow
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [8.6, 8.6],
            [0.3, 2.0],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    # x-axis arrow
    ax.add_artist(
        Arrow3D(
            [0, 2.5],
            [-0.5, -0.5],
            [0, 0],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )

    ax.text(
        -1.4,
        4,
        0,
        "Y",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        0.5,
        1.7,
        "Y'",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.7,
        -0.9,
        0,
        "X",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        8.6,
        2.3,
        "Z",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )


def plot_folding_3d(ax):
    cuboid_dims = (4, 8, 4)
    split_parameters = {"axis": "y", "index": 4, "gap": 1.1}
    nodes_to_highlight = {}
    # Red color for 4x4x4 job.
    for x in range(4):
        for y in range(4):
            for z in range(4):
                nodes_to_highlight[(x, y, z)] = "#d62728"
    # Lighter color for original unfolded job.
    for x in range(4):
        for y in range(4, 8):
            for z in range(2, 4):
                nodes_to_highlight[(x, y, z)] = "#eb9293"

    # Non-highlighted edges to avoid double lines
    non_highlighted_edges = [
        # sorted([(1, 0, 3), (1, 1, 3)]),
    ]

    draw_split_cuboid_3d(
        ax,
        dims=cuboid_dims,
        split_info=split_parameters,
        highlighted_nodes=nodes_to_highlight,
        non_highlighted_edges=non_highlighted_edges,
    )
    ax.set_xlim([0, cuboid_dims[0] - 1])
    ax.set_ylim([-0.4, cuboid_dims[1] - 1 + split_parameters["gap"]])
    ax.set_zlim([0, cuboid_dims[2] - 1])

    # z-axis arrow
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [8.5, 8.5],
            [2.0, 2.9],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    ax.add_artist(
        Arrow3D(
            [3, 3],
            [-0.4, -0.4],
            [2.2, 3.1],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    ax.add_artist(
        Arrow3D(
            [3, 3],
            [-0.4, -0.4],
            [0.2, 1.1],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    # x-axis arrow
    ax.add_artist(
        Arrow3D(
            [0, 2.5],
            [-0.5, -0.5],
            [0, 0],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    # y-axis arrow
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [7, 4.5],
            [2.7, 2.7],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [7, 4.5],
            [1.7, 1.7],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [2.8, 0.5],
            [2.7, 2.7],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    ax.add_artist(
        Arrow3D(
            [0, 0],
            [0.7, 2.8],
            [0.3, 0.3],
            arrowstyle="->",
            color="black",
            lw=1.5,
            mutation_scale=10,
        )
    )
    draw_elliptical_arrow_3d(
        ax,
        center=(1.5, 1.45),  # (y, z) center of the ellipse
        radius_a=1.3,  # Radius in the y-direction
        radius_b=0.25,  # Radius in the z-direction
        start_angle=115,
        end_angle=415,
        plane="yz",  # Draw in a plane parallel to the YZ plane
        plane_val=0,  # Position the plane on the side face (x=0)
        clockwise=True,  # Draw clockwise
        color="black",  # Use the same orange color
        lw=1.5,
        zorder=20,
    )
    # The two wrap-around links.
    draw_elliptical_arrow_3d(
        ax,
        center=(3.36, 1.45),  # (y, z) center of the ellipse
        radius_a=0.35,  # Radius in the y-direction
        radius_b=2.0,  # Radius in the z-direction
        start_angle=235,
        end_angle=465,
        plane="yz",  # Draw in a plane parallel to the YZ plane
        plane_val=0,  # Position the plane on the side face (x=0)
        clockwise=True,  # Draw clockwise
        color="black",  # Use the same orange color
        lw=1.5,
        zorder=200,
    )
    draw_elliptical_arrow_3d(
        ax,
        center=(0.33, 1.45),  # (y, z) center of the ellipse
        radius_a=0.3,  # Radius in the y-direction
        radius_b=1.96,  # Radius in the z-direction
        start_angle=256,
        end_angle=485,
        plane="yz",  # Draw in a plane parallel to the YZ plane
        plane_val=0,  # Position the plane on the side face (x=0)
        clockwise=False,  # Draw counter-clockwise
        color="black",  # Use the same orange color
        lw=1.5,
        zorder=5000,
    )

    ax.text(
        0.9,
        -0.9,
        0,
        "X",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        3.5,
        -0.5,
        0.3,
        "Z",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        3.5,
        -0.5,
        2.3,
        "Z",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        8.85,
        2.3,
        "Z",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        5.6,
        2.4,
        "Y1",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        5.6,
        1.4,
        "Y2",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        1.45,
        1.5,
        "Y2'",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0,
        4.15,
        0.5,
        "Y1'",
        color="black",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        -1,
        6.0,
        -0.4,
        "4x4x4\nTPU nodes (cube)",
        color="dimgray",
        zorder=25,
        ha="center",
        va="center",
        fontsize=10,
    )


if __name__ == "__main__":
    # Set the style for the plots
    plt.rcParams["font.size"] = 10
    latex_line_width = 7
    fig = plt.figure(
        figsize=(latex_line_width, latex_line_width * 0.6), constrained_layout=True
    )
    ax1 = fig.add_subplot(121, projection="3d")
    plot_folding_1d2d(ax1)
    ax2 = fig.add_subplot(122, projection="3d")
    plot_folding_3d(ax2)

    job_1d_color = "#2ca02c"
    job_2d_color = "#1f77b4"
    job_2d_folded_color = "#ff7f0e"
    job_3d_color = "#d62728"
    color_unused = DEFAULT_COLOR

    # Create proxy artists with visible lines
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=job_1d_color,
            label="Folded 1D job\n(18x1x1)",
            markerfacecolor=job_1d_color,
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=job_2d_color,
            label="Original 2D job\n(1x6x4)",
            markerfacecolor=job_2d_color,
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=job_2d_folded_color,
            label="Folded 3D job\n(4x2x3)",
            markerfacecolor=job_2d_folded_color,
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=job_3d_color,
            label="Folded 3D job\n(4x4x4)",
            markerfacecolor=job_3d_color,
            markersize=8,
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_unused,
            label="Unused",
            markerfacecolor=color_unused,
            markersize=8,
            linestyle="-",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.87),
        fontsize=9.5,
        ncol=5,
        columnspacing=0.9,
    )

    # plt.show()
    plot_file_name = "folding_design.pdf"
    plt.savefig(plot_file_name, bbox_inches="tight")

    # Now crop the PDF.
    # 1 inch = 72 points in PDF coordinate system
    inches_from_top = 0.28
    inches_from_bottom = 0.75

    reader = PdfReader(plot_file_name)
    writer = PdfWriter()
    for page in reader.pages:
        media_box = page.mediabox
        page.cropbox = media_box
        page.cropbox.bottom = media_box.bottom + inches_from_bottom * 72
        page.cropbox.top = media_box.top - inches_from_top * 72
        writer.add_page(page)

    # Write the cropped PDF to the output file
    with open(plot_file_name, "wb") as f:
        writer.write(f)
