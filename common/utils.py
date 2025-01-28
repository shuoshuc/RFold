import json
import logging
import simpy
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class Signal:
    """
    A wrapper class for simpy events.
    """

    def __init__(self, env: simpy.core.Environment):
        self.env = env
        self.event = env.event()

    def trigger(self):
        """
        Trigger once and reloads the event.
        """
        self.event.succeed()
        self.event = self.env.event()

    def signal(self):
        """
        Expose the underlying event.
        """
        return self.event


class PrettyForm(logging.Formatter):
    """
    Custom log formatter to make padding and alignment easier.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        record.module = f"{record.module}::{record.funcName}():{record.lineno}"
        return super().format(record)


def spec_parser(specfile: str) -> dict:
    """
    Parse the cluster spec file.
    """
    with open(specfile, "r") as f:
        return json.load(f)


def dump_spec(spec: dict, specfile: str):
    """
    Dump the cluster spec to a file if `specfile` is specified.
    """
    if specfile:
        with open(specfile, "w") as f:
            json.dump(spec, f, indent=4)


def viz3D(dimx: int, dimy: int, dimz: int, array: NDArray[np.float64]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Generate coordinates for the corners of the cube
    x, y, z = np.meshgrid(np.arange(dimx), np.arange(dimy), np.arange(dimz))

    # Plot circles at each corner
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                alpha = 1 if array[i, j, k] > 0 else 0.2
                ax.scatter(i, j, k, s=100, alpha=alpha)

    # Connect circles with lines
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if i < dimx - 1:
                    ax.plot(
                        [x[i, j, k], x[i + 1, j, k]],
                        [y[i, j, k], y[i + 1, j, k]],
                        [z[i, j, k], z[i + 1, j, k]],
                        "k-",
                        color="gray",
                    )
                if j < dimy - 1:
                    ax.plot(
                        [x[i, j, k], x[i, j + 1, k]],
                        [y[i, j, k], y[i, j + 1, k]],
                        [z[i, j, k], z[i, j + 1, k]],
                        "k-",
                        color="gray",
                    )
                if k < dimz - 1:
                    ax.plot(
                        [x[i, j, k], x[i, j, k + 1]],
                        [y[i, j, k], y[i, j, k + 1]],
                        [z[i, j, k], z[i, j, k + 1]],
                        "k-",
                        color="gray",
                    )

    # Set the aspect of the plot to be equal
    ax.set_box_aspect([1, 1, 0.9])
    plt.show()
