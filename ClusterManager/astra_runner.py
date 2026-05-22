"""
AstraSimRunner — the single I/O seam between RFold and the rfold-astra
container. Owns a ProcessPoolExecutor sized at os.cpu_count() - 2 (or a
caller-injected Executor) and the per-uuid filesystem layout under
tmp_root.

runOne is used by ContentionModel.runAstraSim for the ideal (no
contention) call; runMany dispatches per-impacted-job real-JCT calls in
parallel on every contention event.
"""

import logging
import os
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ClusterManager import astra_sim


@dataclass(frozen=True)
class AstraJobSpec:
    """One unit of work for AstraSimRunner.runMany."""
    uuid: int
    shape: tuple
    bw: list
    lt: list


class AstraSimRunner:
    def __init__(
        self,
        tmp_root: Path,
        max_workers: Optional[int] = None,
        executor: Optional[Executor] = None,
    ):
        """
        tmp_root: directory under which per-uuid inputs/outputs are
          materialized (tmp_root/<uuid>/{inputs,outputs}).
        max_workers: pool size override. Defaults to os.cpu_count() - 2.
          Ignored when `executor` is supplied.
        executor: caller-supplied Executor (e.g. ThreadPoolExecutor for
          hermetic tests). Defaults to ProcessPoolExecutor(max_workers).
        """
        self._tmp_root = Path(tmp_root)
        if executor is None:
            if max_workers is None:
                max_workers = os.cpu_count() - 2
            executor = ProcessPoolExecutor(max_workers=max_workers)
        self._pool = executor

    def runOne(
        self,
        uuid: int,
        shape: tuple,
        bw: list,
        lt: list,
    ) -> float:
        """Submit a single astra-sim job through the pool and block until
        it returns the parsed JCT in nanoseconds."""
        future = self._pool.submit(
            astra_sim.run_astra,
            uuid=uuid,
            shape=shape,
            bw_matrix=bw,
            lt_matrix=lt,
            tmp_root=self._tmp_root,
        )
        return future.result()

    def runMany(self, specs: list) -> dict:
        """Submit one future per spec to the pool; block until all
        complete; return {uuid: jct_nsec}. Re-raises the first worker
        exception encountered; remaining pending futures are cancelled.
        """
        if not specs:
            return {}
        t0 = time.monotonic()
        futures = {
            self._pool.submit(
                astra_sim.run_astra,
                uuid=spec.uuid,
                shape=spec.shape,
                bw_matrix=spec.bw,
                lt_matrix=spec.lt,
                tmp_root=self._tmp_root,
            ): spec.uuid
            for spec in specs
        }
        results: dict = {}
        try:
            for fut in as_completed(futures):
                uuid = futures[fut]
                results[uuid] = fut.result()  # re-raises worker exception
        except BaseException:
            for pending in futures:
                pending.cancel()
            raise
        wall_ms = (time.monotonic() - t0) * 1000.0
        logging.info(
            "astra_runner batch: n=%d wall_ms=%.1f", len(specs), wall_ms
        )
        return results

    def shutdown(self) -> None:
        """Release worker resources. Safe to call multiple times."""
        self._pool.shutdown(wait=True, cancel_futures=True)
