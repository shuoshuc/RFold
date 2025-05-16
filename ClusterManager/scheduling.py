import logging
import random
import simpy
import numpy as np
from enum import Enum
from hilbert import decode as hdecode
from math import ceil, prod
from numpy.typing import NDArray
from typing import Optional, Generator

from common.flags import FLAGS
from common.job import Job, TopoType
from common.utils import find_simple_path_helper
from Cluster.cluster import Cluster
from itertools import permutations, combinations


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

    def _find_submesh(
        self, array: NDArray[np.float64], a: int, b: int
    ) -> Optional[tuple[int, int]]:
        """
        Finds a feasible AxB shaped submesh in the 2D mesh/torus.

        Parameters:
        array: 2D numpy array, where 0 means no XPU available.
        a: job shape along dimension x
        b: job shape along dimension y

        Returns:
        A tuple of (x, y) indicating the starting coordinates if a submesh is found.
        Otherwise None.
        """
        if a > self.cluster.dimx or b > self.cluster.dimy:
            logging.debug(f"Job shape ({a}, {b}) exceeds cluster dimension.")
            return None

        # The search range for torus should consider wrap-around.
        if self.cluster.topo in (TopoType.T2D,):
            xrange = self.cluster.dimx
            yrange = self.cluster.dimy
        else:
            xrange = self.cluster.dimx - a + 1
            yrange = self.cluster.dimy - b + 1
        for x in range(xrange):
            for y in range(yrange):
                # Check if all nodes in the submesh have available XPUs.
                if all(
                    array[(x + i) % self.cluster.dimx][(y + j) % self.cluster.dimy] > 0
                    for i in range(a)
                    for j in range(b)
                ):
                    return x, y
        return None

    def _find_slice(
        self, array: NDArray[np.float64], a: int, b: int, c: int
    ) -> Optional[tuple[int, int, int]]:
        """
        Finds a feasible AxBxC shaped submesh in the 3D mesh/torus.

        Parameters:
        array: 3D numpy array, where 0 means no XPU available.
        a: job shape along dimension x
        b: job shape along dimension y
        c: job shape along dimension z

        Returns:
        A tuple of (x, y, z) indicating the starting coordinates if a slice is found.
        Otherwise None.
        """
        if a > self.cluster.dimx or b > self.cluster.dimy or c > self.cluster.dimz:
            logging.debug(f"Job shape ({a}, {b}, {c}) exceeds cluster dimension.")
            return None

        # The search range for torus should consider wrap-around.
        if self.cluster.topo in (TopoType.T3D_NT, TopoType.T3D_T):
            xrange = self.cluster.dimx
            yrange = self.cluster.dimy
            zrange = self.cluster.dimz
        else:
            xrange = self.cluster.dimx - a + 1
            yrange = self.cluster.dimy - b + 1
            zrange = self.cluster.dimz - c + 1
        for x in range(xrange):
            for y in range(yrange):
                for z in range(zrange):
                    # Check if all nodes in the slice have available XPUs.
                    if all(
                        array[(x + i) % self.cluster.dimx][(y + j) % self.cluster.dimy][
                            (z + k) % self.cluster.dimz
                        ]
                        > 0
                        for i in range(a)
                        for j in range(b)
                        for k in range(c)
                    ):
                        return x, y, z
        return None

    def _fit_2d(
        self, avail_array: NDArray[np.float64], job: Job
    ) -> tuple[SchedDecision, Job]:
        """
        Fits a 2D job into the cluster.
        Performs transposition if the original placement fails.
        """
        a, b = job.shape
        loc = self._find_submesh(avail_array, a, b)
        loc_t = self._find_submesh(avail_array, b, a)
        if not loc and not loc_t:
            logging.debug(f"Job {job.uuid} rejected, no feasible placement found.")
            job.logRejectReason(self.env.now, "shape")
            return SchedDecision.REJECT, job
        # If the placement is after transposition, update the job shape.
        base_x, base_y = loc if loc else loc_t
        a, b = (a, b) if loc else (b, a)
        job.shape = (a, b)
        for i in range(base_x, base_x + a):
            for j in range(base_y, base_y + b):
                # Make sure to handle wrap-around indices for torus.
                job.allocation[f"x{i % self.cluster.dimx}-y{j % self.cluster.dimy}"] = 1
        return SchedDecision.ADMIT, job

    def _fit_3d(
        self, avail_array: NDArray[np.float64], job: Job
    ) -> tuple[SchedDecision, Job]:
        """
        Fits a 3D job into the cluster.
        Performs transposition if the original placement fails.
        """
        shapes = list(permutations(job.shape))
        # A shape should have 6 permutations in 3D space. For now, they are treated as equivalent.
        # But ideally, the best shape should be the one that generates least fragmentation.
        # Need some form of evaluation to determine the best shape.
        for a, b, c in shapes:
            loc = self._find_slice(avail_array, a, b, c)
            if not loc:
                continue
            base_x, base_y, base_z = loc
            # Update the job shape since the placement could be after transposition.
            job.shape = (a, b, c)
            for i in range(base_x, base_x + a):
                for j in range(base_y, base_y + b):
                    for k in range(base_z, base_z + c):
                        # Make sure to handle wrap-around indices for torus.
                        job.allocation[
                            f"x{i % self.cluster.dimx}-"
                            f"y{j % self.cluster.dimy}-"
                            f"z{k % self.cluster.dimz}"
                        ] = 1
            return SchedDecision.ADMIT, job

        logging.debug(f"Job {job.uuid} rejected, no feasible placement found.")
        job.logRejectReason(self.env.now, "shape")
        return SchedDecision.REJECT, job

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

    def _firstfit(self, job: Job) -> tuple[SchedDecision, Job]:
        """
        First-fit scheduling policy.
        """
        # TODO: generalize to other topologies.
        if job.topology in (TopoType.CLOS,):
            logging.debug(f"Job {job.uuid} rejected, unsupported topology.")
            return SchedDecision.REJECT, job

        if job.topology in (TopoType.MESH2D, TopoType.T2D):
            return self._fit_2d(self.cluster.toArray(), job)
        elif job.topology in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
            return self._fit_3d(self.cluster.toArray(), job)

    def _find_slices_loc_2d(
        self,
        candidates: list[tuple[int, int]],
        blocks_needed: int,
        partial_x: int,
        partial_y: int,
        loc: tuple[int, int],
        rsize: int,
    ) -> tuple[list[tuple], list[str]]:
        """
        Find `blocks_needed` number of partial_x * partial_y shaped submesh.
        Must be at the specified corner (`loc`).
        Returns the set of blocks picked along with a list of node names to allocate.
        """
        blocks_to_allocate = []
        nodes_to_allocate = []
        for block_coord in candidates:
            if len(blocks_to_allocate) >= blocks_needed:
                break
            nodes = self.cluster.blocks[block_coord]
            avail = self.cluster.toBlockArray(nodes)
            i, j = loc
            if np.all(avail[i : i + partial_x, j : j + partial_y] > 0):
                blocks_to_allocate.append(block_coord)
                base_x, base_y = block_coord
                for local_x in range(i, i + partial_x):
                    for local_y in range(j, j + partial_y):
                        # Cast local coordinates to global coordinates.
                        x_coord = base_x * rsize + local_x
                        y_coord = base_y * rsize + local_y
                        nodes_to_allocate.append(f"x{x_coord}-y{y_coord}")
        return (blocks_to_allocate, nodes_to_allocate)

    def _reconfig_2d(self, job: Job, rsize: int) -> tuple[SchedDecision, Job]:
        for shape in permutations(job.shape):
            # A job shape of (x, y) can be divided into several chunks:
            # (1) full_x * full_y completely empty reconfigurable blocks.
            # (2) full_x partially empty blocks, each of shape (RSIZE, partial_y).
            # (3) full_y partially empty blocks, each of shape (partial_x, RSIZE).
            # (4) one partially empty block, of shape (partial_x, partial_y).
            full_x, full_y = map(int, np.array(shape) // rsize)
            partial_x, partial_y = map(int, np.array(shape) % rsize)

            num_needed = {
                "full_block": full_x * full_y,
                "partial_x": full_y if partial_x > 0 else 0,
                "partial_y": full_x if partial_y > 0 else 0,
                "corner": 1 if partial_x > 0 and partial_y > 0 else 0,
            }
            num_found = {
                "full_block": 0,
                "partial_x": 0,
                "partial_y": 0,
                "corner": 0,
            }

            allocated = set()
            candidates = list(self.cluster.blocks.keys())
            for counter_name, x, y in [
                ("full_block", rsize, rsize),
                ("partial_x", partial_x, rsize),
                ("partial_y", rsize, partial_y),
                ("corner", partial_x, partial_y),
            ]:
                # Skip the blocks that are not needed.
                if num_needed[counter_name] <= 0:
                    continue

                top_left = (0, 0)
                top_right = (0, rsize - y)
                bottom_left = (rsize - x, 0)
                bottom_right = (rsize - x, rsize - y)
                # De-duplicate the locations.
                for loc in set([top_left, top_right, bottom_left, bottom_right]):
                    blocks_to_alloc, nodes_to_alloc = self._find_slices_loc_2d(
                        candidates=candidates,
                        blocks_needed=num_needed[counter_name],
                        partial_x=x,
                        partial_y=y,
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

    def _find_slices_loc_3d(
        self,
        candidates: list[tuple[int, int, int]],
        blocks_needed: int,
        partial_x: int,
        partial_y: int,
        partial_z: int,
        loc: tuple[int, int, int],
        rsize: int,
    ) -> tuple[list[tuple], list[str]]:
        """
        Find `blocks_needed` number of partial_x * partial_y * partial_z shaped submesh.
        Must be at the specified corner (`loc`).
        Returns the set of blocks picked along with a list of node names to allocate.
        """
        blocks_to_allocate = []
        nodes_to_allocate = []
        for block_coord in candidates:
            if len(blocks_to_allocate) >= blocks_needed:
                break
            nodes = self.cluster.blocks[block_coord]
            avail = self.cluster.toBlockArray(nodes)
            i, j, k = loc
            if np.all(avail[i : i + partial_x, j : j + partial_y, k : k + partial_z] > 0):
                blocks_to_allocate.append(block_coord)
                base_x, base_y, base_z = block_coord
                for local_x in range(i, i + partial_x):
                    for local_y in range(j, j + partial_y):
                        for local_z in range(k, k + partial_z):
                            # Cast local coordinates to global coordinates.
                            x_coord = base_x * rsize + local_x
                            y_coord = base_y * rsize + local_y
                            z_coord = base_z * rsize + local_z
                            nodes_to_allocate.append(f"x{x_coord}-y{y_coord}-z{z_coord}")
        return (blocks_to_allocate, nodes_to_allocate)

    def _reconfig_3d(self, job: Job, rsize: int) -> tuple[SchedDecision, Job]:
        for shape in permutations(job.shape):
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

            num_needed = {
                "full_block": full_x * full_y * full_z,
                "face_xy": full_x * full_y if partial_z > 0 else 0,
                "face_xz": full_x * full_z if partial_y > 0 else 0,
                "face_yz": full_y * full_z if partial_x > 0 else 0,
                "edge_x": full_x if partial_y * partial_z > 0 else 0,
                "edge_y": full_y if partial_x * partial_z > 0 else 0,
                "edge_z": full_z if partial_x * partial_y > 0 else 0,
                "corner": 1 if partial_x * partial_y * partial_z > 0 else 0,
            }
            num_found = {
                "full_block": 0,
                "face_xy": 0,
                "face_xz": 0,
                "face_yz": 0,
                "edge_x": 0,
                "edge_y": 0,
                "edge_z": 0,
                "corner": 0,
            }

            allocated = set()
            candidates = list(self.cluster.blocks.keys())
            for counter_name, x, y, z in [
                ("full_block", rsize, rsize, rsize),
                ("face_xy", rsize, rsize, partial_z),
                ("face_xz", rsize, partial_y, rsize),
                ("face_yz", partial_x, rsize, rsize),
                ("edge_x", rsize, partial_y, partial_z),
                ("edge_y", partial_x, rsize, partial_z),
                ("edge_z", partial_x, partial_y, rsize),
                ("corner", partial_x, partial_y, partial_z),
            ]:
                # Skip the blocks that are not needed.
                if num_needed[counter_name] <= 0:
                    continue

                loc1 = (0, 0, 0)
                loc2 = (rsize - x, 0, 0)
                loc3 = (0, rsize - y, 0)
                loc4 = (0, 0, rsize - z)
                loc5 = (rsize - x, rsize - y, 0)
                loc6 = (rsize - x, 0, rsize - z)
                loc7 = (0, rsize - y, rsize - z)
                loc8 = (rsize - x, rsize - y, rsize - z)
                # De-duplicate the locations.
                for loc in set([loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8]):
                    blocks_to_alloc, nodes_to_alloc = self._find_slices_loc_3d(
                        candidates=candidates,
                        blocks_needed=num_needed[counter_name],
                        partial_x=x,
                        partial_y=y,
                        partial_z=z,
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

    def _reconfig(self, job: Job, rsize: int) -> tuple[SchedDecision, Job]:
        """
        Reconfigurable scheduling policy.
        """
        # TODO: generalize to other topologies.
        if job.topology in (TopoType.CLOS,):
            logging.debug(f"Job {job.uuid} rejected, unsupported topology.")
            return SchedDecision.REJECT, job

        if job.topology in (TopoType.MESH2D, TopoType.T2D):
            return self._reconfig_2d(job, rsize)
        elif job.topology in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
            return self._reconfig_3d(job, rsize)

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

    def block_candidates_generator(
        self,
        all_block_avail: dict[tuple, np.ndarray],
        job_size: int,
    ) -> Generator[tuple[tuple[int, ...]], None, None]:
        """
        Generates all possible combinations of block indices that can host a job
        of a given size, yielding them in a sorted order.

        The sorting criteria are (in order of priority):
        1. Fewer blocks are ranked higher.
        2. If # blocks is the same, combinations with total available nodes
        closest to (and >=) job_size are ranked higher.
        3. If still tied, combinations where the higher minimum current utilization
        among its constituent blocks are ranked higher.

        Args:
            all_block_avail: A dict of NumPy arrays representing block availability.
            job_size: The number of nodes the job requires.
        """
        # Iterate through the number of blocks in a combination.
        # This handles the primary metric (fewer blocks first) naturally.
        min_blocks = ceil(job_size / next(iter(all_block_avail.values())).size)
        max_blocks = ceil(job_size / next(iter(all_block_avail.values())).shape[0])
        for nblock in range(min_blocks, max_blocks + 1):
            candidates_with_metrics = []
            for coords in combinations(all_block_avail.keys(), nblock):
                # print(f"Finding combination of {nblock} blocks: {coords}")
                combo_total_free = sum(np.sum(all_block_avail[coord]) for coord in coords)
                # This combination can host the job.
                if combo_total_free >= job_size:
                    # Metric 2: diff between free nodes and job size
                    diff = combo_total_free - job_size

                    combo_util = []
                    for coord in coords:
                        num_free = np.sum(all_block_avail[coord])
                        block_capacity = all_block_avail[coord].size
                        util = (block_capacity - num_free) / block_capacity
                        combo_util.append(util)
                    # Metric 3: Minimum block utilization in combo
                    # Higher actual minimum utilization is better.
                    # Negative utilization ranks higher in ascending order.
                    min_util = -min(combo_util)

                    candidates_with_metrics.append((diff, min_util, coords))
                if len(candidates_with_metrics) > 1000:
                    # Limit the number of candidates to avoid excessive computation.
                    break

            # Sort candidates based on metric 2 and 3.
            candidates_with_metrics.sort(key=lambda x: (x[0], x[1]))

            # Yield the sorted combinations for the current nblock.
            for _, _, coords in candidates_with_metrics:
                yield coords

    def _folding(self, job: Job, rsize: int) -> tuple[SchedDecision, Job]:
        """
        Torus folding scheduling policy.
        """

        def _real_job_dimension(shape: tuple) -> int:
            """
            Check if the job is 1D, 2D or 3D.
            Note: single node job is technically 1D but we return a 0.
            """
            return sum(dim != 1 for dim in shape)

        if _real_job_dimension(job.shape) == 1:
            logging.info(f"1D job, shape: {job.shape}")
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

        if job.topology in (TopoType.MESH2D, TopoType.T2D):
            return self._reconfig_2d(job, rsize)
        elif job.topology in (TopoType.MESH3D, TopoType.T3D_NT, TopoType.T3D_T):
            return self._reconfig_3d(job, rsize)

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
        elif policy == "folding":
            return self._folding(job, rsize)
        elif policy == "slurm_hilbert":
            return self._slurm_hilbert(job)
        # The default fallback is to reject all jobs.
        return SchedDecision.REJECT, job
