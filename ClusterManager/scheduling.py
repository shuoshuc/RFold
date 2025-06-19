import logging
import random
import simpy
import numpy as np
from enum import Enum
from hilbert import decode as hdecode
from numpy.typing import NDArray
from typing import Optional, Generator
from itertools import permutations, combinations, product

from common.flags import FLAGS
from common.job import Job, TopoType
from Cluster.cluster import Cluster
from ClusterManager.torus import (
    real_shape_dimension,
    find_simple_path_helper,
    find_1D_cycle_helper,
    fold,
    folded_shape_helper,
)


class SchedDecision(Enum):
    # Admit a job immediately.
    ADMIT = 1
    # Reject a job immediately.
    REJECT = 2
    # Preempt running jobs to admit a job.
    # May take some time to migrate.
    PREEMPT = 3
    # Reconfigure the cluster topology to admit a job.
    # May take some time.
    RECONFIGURE = 4


class SchedulingPolicy:
    def __init__(self, env: simpy.core.Environment, cluster: Cluster):
        self.env = env
        self.cluster = cluster
        random.seed(42)

    def randDecision(self, job: Job) -> tuple[SchedDecision, Job]:
        """
        Makes a random scheduling decision and does not modify the given job.
        To be used for testing purposes.
        """
        return random.choice([d for d in SchedDecision]), job

    def check_total_xpu(self, job: Job) -> bool:
        """
        Check if the total number of XPUs required by the job is available.
        If not, return False. Otherwise, return True.
        """
        return job.size <= self.cluster.totalIdleXPU()

    def check_total_node(self, job: Job) -> bool:
        """
        Check if the total number of nodes required by the job is available.
        If not, return False. Otherwise, return True.
        """
        req_node_cnt = len(job.shape) if job.topology == TopoType.CLOS else job.size
        return req_node_cnt <= self.cluster.totalIdleNodes()

    def _find_submesh(
        self, array: NDArray, shape: tuple[int, ...]
    ) -> Optional[tuple[int, ...]]:
        """
        Finds a feasible submesh of `shape` in the 2D/3D mesh/torus.

        Parameters:
            array: numpy array, where 0 means no XPU available.
            shape: job shape along each dimension (e.g., (a, b) or (a, b, c)).

        Returns:
            A tuple indicating the starting coordinates if a submesh is found.
            Otherwise None.
        """
        ndim = array.ndim
        if len(shape) != ndim:
            logging.error(f"Job shape {len(shape)} mismatches array dimension {ndim}D.")
            return None

        cluster_dims = (self.cluster.dimx, self.cluster.dimy, self.cluster.dimz)[:ndim]
        if any(s > c for s, c in zip(shape, cluster_dims)):
            logging.debug(f"Job shape {shape} exceeds cluster dimension {cluster_dims}.")
            return None

        is_torus = False
        if ndim == 2 and self.cluster.topo == TopoType.T2D:
            is_torus = True
        elif ndim == 3 and self.cluster.topo in (TopoType.T3D_NT, TopoType.T3D_T):
            is_torus = True

        # The search range for torus should consider wrap-around.
        search_upper_bounds = [
            cluster_dims[i] if is_torus else cluster_dims[i] - shape[i] + 1
            for i in range(ndim)
        ]
        for start_coord in product(*[range(ub) for ub in search_upper_bounds]):
            # If all nodes in the submesh have available XPUs, return the start position.
            if all(
                array[
                    tuple(
                        (start_coord[i] + offset[i]) % cluster_dims[i]
                        for i in range(ndim)
                    )
                ]
                > 0
                for offset in product(*[range(s) for s in shape])
            ):
                return start_coord

        return None

    def _firstfit(self, job: Job) -> tuple[SchedDecision, Job]:
        """
        First-fit scheduling policy.
        """
        # TODO: generalize to other topologies.
        if job.topology in (TopoType.CLOS,):
            logging.debug(f"Job {job.uuid} rejected, unsupported topology.")
            return SchedDecision.REJECT, job

        dims = (
            self.cluster.dimx,
            self.cluster.dimy,
            self.cluster.dimz,
        )[: len(job.shape)]
        prefix = ("x", "y", "z")[: len(job.shape)]

        for shape in list(permutations(job.shape)):
            loc = self._find_submesh(self.cluster.toArray(), shape)
            if not loc:
                continue
            job.shape = shape
            for coord in product(
                *[range(base, base + offset) for base, offset in zip(loc, shape)]
            ):
                wrapped_coord = tuple((c % d) for c, d in zip(coord, dims))
                job.allocation[
                    "-".join([f"{p}{c}" for p, c in zip(prefix, wrapped_coord)])
                ] = 1
            return SchedDecision.ADMIT, job

        logging.debug(f"Job {job.uuid} rejected, no feasible placement found.")
        job.logRejectReason(self.env.now, "shape")
        return SchedDecision.REJECT, job

    def _find_slices_loc(
        self,
        candidates: list[tuple[int, ...]],
        blocks_needed: int,
        partial: tuple[int, ...],
        loc: tuple[int, ...],
        rsize: int,
    ) -> tuple[list[tuple], list[str]]:
        """
        Find `blocks_needed` number of `partial` shaped submesh.
        Must be at the specified corner (`loc`).
        Returns the set of blocks picked along with a list of node names to allocate.
        """
        ndim = len(loc)
        prefix = ("x", "y", "z")[:ndim]
        blocks_to_allocate = []
        nodes_to_allocate = []

        if ndim != len(partial):
            raise RuntimeError(f"Mismatch: loc {len(loc)}D, partial {len(partial)}D.")

        for block_coord in candidates:
            if len(blocks_to_allocate) >= blocks_needed:
                break

            nodes = self.cluster.blocks[block_coord]
            avail = self.cluster.toBlockArray(nodes)
            subarray = avail[
                tuple(slice(loc[i], loc[i] + partial[i]) for i in range(ndim))
            ]
            if subarray.size > 0 and np.all(subarray > 0):
                blocks_to_allocate.append(block_coord)
                for local_coord in product(
                    *[range(l, l + p) for l, p in zip(loc, partial)]
                ):
                    global_coord = tuple(
                        b * rsize + l for b, l in zip(block_coord, local_coord)
                    )
                    nodes_to_allocate.append(
                        "-".join([f"{p}{g}" for p, g in zip(prefix, global_coord)])
                    )
        return (blocks_to_allocate, nodes_to_allocate)

    def _reconfig_param_map(
        self, shape: tuple[int, ...], rsize: int
    ) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[int, ...]]]:
        """
        Given a job shape, calculates the number of blocks needed and associated params.
        """
        if len(shape) == 2:
            # A job shape of (x, y) can be divided into several chunks:
            # (1) full_x * full_y completely empty reconfigurable blocks.
            # (2) full_x partially empty blocks, each of shape (RSIZE, partial_y).
            # (3) full_y partially empty blocks, each of shape (partial_x, RSIZE).
            # (4) one partially empty block, of shape (partial_x, partial_y).
            full_x, full_y = map(int, np.array(shape) // rsize)
            partial_x, partial_y = map(int, np.array(shape) % rsize)

            num_needed_2d = {
                "full_block": full_x * full_y,
                "partial_x": full_y if partial_x > 0 else 0,
                "partial_y": full_x if partial_y > 0 else 0,
                "corner": 1 if partial_x > 0 and partial_y > 0 else 0,
            }
            num_found_2d = {
                "full_block": 0,
                "partial_x": 0,
                "partial_y": 0,
                "corner": 0,
            }
            partial_2d = {
                "full_block": (rsize, rsize),
                "partial_x": (partial_x, rsize),
                "partial_y": (rsize, partial_y),
                "corner": (partial_x, partial_y),
            }
            return (num_needed_2d, num_found_2d, partial_2d)
        elif len(shape) == 3:
            # A job shape of (x, y, z) can be divided into several chunks:
            # (1) full_x * full_y completely empty reconfigurable blocks.
            # (2) xy face: (RSIZE, RSIZE, partial_z) * full_x * full_y partially empty blocks.
            # (3) xz face: (RSIZE, partial_y, RSIZE) * full_x * full_z partially empty blocks.
            # (4) yz face: (partial_x, RSIZE, RSIZE) * full_y * full_z partially empty blocks.
            # (5) x edge: (RSIZE, partial_y, partial_z) * full_x partially empty blocks.
            # (6) y edge: (partial_x, RSIZE, partial_z) * full_y partially empty blocks.
            # (7) z edge: (partial_x, partial_y, RSIZE) * full_z partially empty blocks.
            # (8) corner: (partial_x, partial_y, partial_z) * 1 partially empty block.
            full_x, full_y, full_z = map(int, np.array(shape) // rsize)
            partial_x, partial_y, partial_z = map(int, np.array(shape) % rsize)

            num_needed_3d = {
                "full_block": full_x * full_y * full_z,
                "face_xy": full_x * full_y if partial_z > 0 else 0,
                "face_xz": full_x * full_z if partial_y > 0 else 0,
                "face_yz": full_y * full_z if partial_x > 0 else 0,
                "edge_x": full_x if partial_y * partial_z > 0 else 0,
                "edge_y": full_y if partial_x * partial_z > 0 else 0,
                "edge_z": full_z if partial_x * partial_y > 0 else 0,
                "corner": 1 if partial_x * partial_y * partial_z > 0 else 0,
            }
            num_found_3d = {
                "full_block": 0,
                "face_xy": 0,
                "face_xz": 0,
                "face_yz": 0,
                "edge_x": 0,
                "edge_y": 0,
                "edge_z": 0,
                "corner": 0,
            }
            partial_3d = {
                "full_block": (rsize, rsize, rsize),
                "face_xy": (rsize, rsize, partial_z),
                "face_xz": (rsize, partial_y, rsize),
                "face_yz": (partial_x, rsize, rsize),
                "edge_x": (rsize, partial_y, partial_z),
                "edge_y": (partial_x, rsize, partial_z),
                "edge_z": (partial_x, partial_y, rsize),
                "corner": (partial_x, partial_y, partial_z),
            }
            return (num_needed_3d, num_found_3d, partial_3d)
        else:
            raise ValueError(f"Unsupported shape dimension: {len(shape)}D")

    def _reconfig_loc_helper(
        self, shape: tuple[int, ...], rsize: int
    ) -> list[tuple[int, ...]]:
        """
        Lists all feasible locations for reconfigurable scheduling in 2D/3D torus.
        """
        if len(shape) == 2:
            x, y = shape
            top_left = (0, 0)
            top_right = (0, rsize - y)
            bottom_left = (rsize - x, 0)
            bottom_right = (rsize - x, rsize - y)
            return [top_left, top_right, bottom_left, bottom_right]
        elif len(shape) == 3:
            x, y, z = shape
            loc1 = (0, 0, 0)
            loc2 = (rsize - x, 0, 0)
            loc3 = (0, rsize - y, 0)
            loc4 = (0, 0, rsize - z)
            loc5 = (rsize - x, rsize - y, 0)
            loc6 = (rsize - x, 0, rsize - z)
            loc7 = (0, rsize - y, rsize - z)
            loc8 = (rsize - x, rsize - y, rsize - z)
            return [loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8]
        else:
            raise ValueError(f"Unsupported shape dimension: {len(shape)}D")

    def _reconfig(
        self, job: Job, rsize: int, free_loc: bool = False
    ) -> tuple[SchedDecision, Job]:
        """
        Reconfigurable scheduling policy.
        """
        # TODO: generalize to other topologies.
        if job.topology in (TopoType.CLOS,):
            logging.debug(f"Job {job.uuid} rejected, unsupported topology.")
            return SchedDecision.REJECT, job

        for shape in permutations(job.shape):
            num_needed, num_found, partial = self._reconfig_param_map(shape, rsize)
            allocated = set()
            candidates = list(self.cluster.blocks.keys())
            for counter_name in num_needed.keys():
                # Skip the blocks that are not needed.
                if num_needed[counter_name] <= 0:
                    continue

                # De-duplicate the locations.
                locations = set(self._reconfig_loc_helper(partial[counter_name], rsize))
                # Extend the location list if OCS links are not required.
                if free_loc and num_needed["full_block"] < 1:
                    block_dim_sizes = (rsize, rsize, rsize)[: len(shape)]
                    search_bounds = [
                        s - p + 1 for s, p in zip(block_dim_sizes, partial[counter_name])
                    ]
                    locations = [
                        loc for loc in product(*[range(ub) for ub in search_bounds])
                    ]
                for loc in locations:
                    blocks_to_alloc, nodes_to_alloc = self._find_slices_loc(
                        candidates=candidates,
                        blocks_needed=num_needed[counter_name],
                        partial=partial[counter_name],
                        loc=loc,
                        rsize=rsize,
                    )
                    # Enough blocks found at this location. Done for this partial shape.
                    if len(blocks_to_alloc) >= num_needed[counter_name]:
                        num_found[counter_name] = len(blocks_to_alloc)
                        for block_coord in blocks_to_alloc:
                            allocated.add(block_coord)
                            candidates.remove(block_coord)
                            for node in nodes_to_alloc:
                                job.allocation[node] = 1
                        break

            # Check if there are enough blocks to satisfy the job shape.
            if all(num_found[key] >= num_needed[key] for key in num_needed):
                job.shape = shape
                return SchedDecision.ADMIT, job
            else:
                # Reset the job allocation and retry.
                job.allocation = {}

        # If we reach here, allocation is unsuccessful.
        logging.debug(f"Job {job.uuid} rejected, not enough empty blocks found")
        job.logRejectReason(self.env.now, "shape")
        return SchedDecision.REJECT, job

    def block_candidates(
        self,
        used_blocks_avail: dict[tuple, NDArray],
        empty_blocks_avail: dict[tuple, NDArray],
        job_size: int,
    ) -> list[tuple[int, ...]]:
        """
        Find the best block candidates for a job of a given size.
        The selection logic is as follows:
        (1) If used blocks do not provide enough nodes, then add a few empty blocks.
        (2) Sort the blocks from least utilized to most utilized.
        (3) Select the blocks until the job size is satisfied. This ensures minimum blocks used.

        Returns:
            A list of block coordinates that can host the job.
            If no blocks are found, return an empty list.
        """
        used_block_map = used_blocks_avail.copy()
        empty_block_map = empty_blocks_avail.copy()
        while sum(np.sum(block) for block in used_block_map.values()) < job_size:
            # Infeasible to fit the job in current blocks.
            if not empty_block_map:
                return []
            coord, array = empty_block_map.popitem()
            used_block_map[coord] = array
        # Sort blocks from least utilized to most utilized.
        used_block_map = dict(
            sorted(
                used_block_map.items(),
                key=lambda item: np.sum(item[1]),
                reverse=True,
            )
        )
        curr_sum = 0
        candidates = []
        for coord, array in used_block_map.items():
            curr_sum += np.sum(array)
            candidates.append(coord)
            if curr_sum >= job_size:
                return candidates

        # If we reach here, it means we have not found enough blocks.
        return []

    def _rfold(self, job: Job, rsize: int) -> tuple[SchedDecision, Job]:
        """
        RFold scheduling policy.
        """
        if real_shape_dimension(job.shape) == 1:
            used_blocks_avail = {}
            empty_blocks_avail = {}
            for coord, nodes in self.cluster.blocks.items():
                avail = self.cluster.toBlockArray(nodes, rsize)
                # Skip blocks that are almost full.
                if np.sum(avail) <= 1:
                    continue
                if np.sum(avail) >= rsize ** len(job.shape):
                    empty_blocks_avail[coord] = avail
                else:
                    used_blocks_avail[coord] = avail
            block_coords = self.block_candidates(
                used_blocks_avail, empty_blocks_avail, job.size
            )
            if block_coords:
                logging.debug(f"Candidate block coordinates: {block_coords}")
                for coord in block_coords[:-1]:
                    for node in self.cluster.blocks[coord]:
                        if node.numIdleXPU() > 0:
                            job.allocation[node.name] = 1
                last_block = {**used_blocks_avail, **empty_blocks_avail}[block_coords[-1]]
                for i in reversed(range(last_block.ndim)):
                    logging.debug(
                        f"last block: {block_coords[-1]}, axis={i}, nodes needed: "
                        f"{job.size - len(job.allocation)}, nodes avail: {np.sum(last_block)}"
                    )
                    path = find_simple_path_helper(
                        block_coords[-1],
                        last_block,
                        job.size - len(job.allocation),
                        i,
                        rsize,
                        10,
                    )
                    if path:
                        logging.debug(f"Found path: {path}")
                        for node_coord in path:
                            prefix = ("x", "y", "z")[: len(node_coord)]
                            job.allocation[
                                "-".join([f"{p}{c}" for p, c in zip(prefix, node_coord)])
                            ] = 1
                        return SchedDecision.ADMIT, job
                    else:
                        logging.info(f"[WARNING] No folded path found: job {job.uuid}")
                # Reset the job allocation and fall back to normal scheduling.
                job.allocation = {}
        else:
            # Regular folding for 2D/3D jobs.
            base_num_needed, _, _ = self._reconfig_param_map(job.shape, rsize)
            folded_option_list = []
            for folded_shape in fold(job.shape, rsize):
                num_needed, _, _ = self._reconfig_param_map(folded_shape, rsize)
                sum_blocks = sum(num_needed.values())
                if sum_blocks > sum(base_num_needed.values()):
                    continue
                folded_option_list.append((folded_shape, sum_blocks))
            if folded_option_list:
                job.shape = sorted(folded_option_list, key=lambda x: x[1])[0][0]

        return self._reconfig(job=job, rsize=rsize, free_loc=True)

    def _foldonly(self, job: Job, rsize: int) -> tuple[SchedDecision, Job]:
        """
        Folding-only scheduling policy.
        """

        def fold_func(job_shape: tuple[int, ...], rsize: int) -> list[tuple[int, ...]]:
            if real_shape_dimension(job.shape) == 1:
                return folded_shape_helper(job_shape)
            else:
                return fold(job_shape, rsize)

        folded_option_list = []
        for folded_shape in fold_func(job.shape, rsize):
            # Only feasible folded shapes are useful.
            if any(sz > rsize for sz in folded_shape):
                continue
            folded_option_list.append(folded_shape)
        if folded_option_list:
            # Pick the shape with smallest maximum dimension size.
            # This indiates it is closer to a square/cube shape.
            job.shape = sorted(folded_option_list, key=lambda x: max(x))[0]

        # # Optional: find 1D cycle for 1D jobs.
        # if real_shape_dimension(job.shape) == 1 and job.size >= 4:
        #     avail = self.cluster.toArray()
        #     cycle = find_1D_cycle_helper(avail, job.size, 10)
        #     if cycle:
        #         for node_coord in cycle:
        #             prefix = ("x", "y", "z")[: len(node_coord)]
        #             job.allocation[
        #                 "-".join([f"{p}{c}" for p, c in zip(prefix, node_coord)])
        #             ] = 1
        #         return SchedDecision.ADMIT, job
        #     else:
        #         logging.info(f"[WARNING] No folded path found: job {job.uuid}")

        return self._firstfit(job=job)

    def _slurm_hilbert(self, job: Job) -> tuple[SchedDecision, Job]:
        """
        The SLURM scheduling policy using Hilbert curve. Each node in N-dimensional topology
        is mapping to a linear curve using the Hilbert index. Nodes that are close to each other
        on the Hilbert curve are allocated to the job.
        """
        # TODO: generalize to other topologies.
        if job.topology in (TopoType.CLOS,):
            logging.debug(f"Job {job.uuid} rejected, unsupported topology.")
            return SchedDecision.REJECT, job

        avail_nodes = self.cluster.linearAvail()
        best_start, best_range = None, float("inf")

        # Try to find the best contiguous allocation, where "best" is defined as
        # the smallest range in Hilbert index. This implies a tight fit with
        # low communication overhead.
        for i in range(len(avail_nodes) - job.size + 1):
            start = avail_nodes[i]
            end = avail_nodes[i + job.size - 1]
            range_size = end - start

            if range_size < best_range:
                best_start, best_range = i, range_size

        # Allocate the best block if found.
        if best_start is not None:
            for index in avail_nodes[best_start : best_start + job.size]:
                if job.topology in (TopoType.MESH2D, TopoType.T2D):
                    x, y = hdecode(index, 2, self.cluster.bits_per_dim)[0]
                    # Make sure to handle wrap-around indices for torus.
                    job.allocation[
                        f"x{x % self.cluster.dimx}-y{y % self.cluster.dimy}"
                    ] = 1
                elif job.topology in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
                    x, y, z = hdecode(index, 3, self.cluster.bits_per_dim)[0]
                    # Make sure to handle wrap-around indices for torus.
                    job.allocation[
                        f"x{x % self.cluster.dimx}-"
                        f"y{y % self.cluster.dimy}-"
                        f"z{z % self.cluster.dimz}"
                    ] = 1
            return SchedDecision.ADMIT, job

        job.logRejectReason(self.env.now, "shape")
        return SchedDecision.REJECT, job

    def place(
        self, job: Job, policy: str = FLAGS.place_policy, rsize: int = FLAGS.rsize
    ) -> tuple[SchedDecision, Job]:
        """
        Make a scheduling decision for a job. Note that the job (e.g., shape, duration)
        could be modifed to achieve a more desirable scheduling decision. But if the job
        is rejected, it is not modified.
        The (modified) job is returned along with the decision.
        """
        if job.topology != self.cluster.topo:
            raise ValueError("Job topology mismatches cluster topology.")
        # Jobs that fail the total XPU check get rejected.
        if not self.check_total_xpu(job):
            logging.debug(f"Job {job.uuid} rejected, insufficient total number of XPUs.")
            job.logRejectReason(self.env.now, "resource")
            return SchedDecision.REJECT, job
        # Jobs that fail the total node check get rejected.
        if not self.check_total_node(job):
            logging.debug(f"Job {job.uuid} rejected, insufficient number of idle nodes.")
            job.logRejectReason(self.env.now, "resource")
            return SchedDecision.REJECT, job

        if policy == "firstfit":
            return self._firstfit(job)
        elif policy == "reconfig":
            return self._reconfig(job, rsize)
        elif policy == "rfold":
            return self._rfold(job, rsize)
        elif policy == "folding":
            return self._foldonly(job, min(FLAGS.dim))
        elif policy == "slurm_hilbert":
            return self._slurm_hilbert(job)
        # The default fallback is to reject all jobs.
        return SchedDecision.REJECT, job
