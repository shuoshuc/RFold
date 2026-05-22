import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ClusterManager import astra_runner
from ClusterManager.astra_runner import AstraJobSpec, AstraSimRunner


class TestAstraJobSpec(unittest.TestCase):
    def test_frozen_and_carries_uuid_shape_matrices(self):
        spec = AstraJobSpec(
            uuid=1,
            shape=(2, 2),
            bw=[[0.0, 1.0], [1.0, 0.0]],
            lt=[[0.0, 100.0], [100.0, 0.0]],
        )
        self.assertEqual(spec.uuid, 1)
        self.assertEqual(spec.shape, (2, 2))
        with self.assertRaises(Exception):
            # Frozen dataclass: attribute assignment must raise.
            spec.uuid = 999  # type: ignore[misc]


class TestAstraSimRunnerSingleCall(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="astra_runner_test_")
        # Inject a hermetic ThreadPoolExecutor so mock.patch on
        # astra_sim.run_astra propagates to the worker.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.runner = AstraSimRunner(
            tmp_root=Path(self.tmpdir), executor=self.executor
        )

    def tearDown(self):
        self.runner.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_one_invokes_run_astra_with_caller_matrices(self):
        bw = [[0.0, 5.0], [5.0, 0.0]]
        lt = [[0.0, 9.0], [9.0, 0.0]]
        with patch.object(
            astra_runner.astra_sim, "run_astra", return_value=42.0
        ) as mock_ra:
            result = self.runner.runOne(uuid=3, shape=(2,), bw=bw, lt=lt)
        self.assertEqual(result, 42.0)
        mock_ra.assert_called_once()
        kwargs = mock_ra.call_args.kwargs
        self.assertEqual(kwargs["uuid"], 3)
        self.assertEqual(kwargs["shape"], (2,))
        self.assertEqual(kwargs["bw_matrix"], bw)
        self.assertEqual(kwargs["lt_matrix"], lt)
        self.assertEqual(Path(kwargs["tmp_root"]), Path(self.tmpdir))


class TestAstraSimRunnerBatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="astra_runner_test_")
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.runner = AstraSimRunner(
            tmp_root=Path(self.tmpdir), executor=self.executor
        )

    def tearDown(self):
        self.runner.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_many_returns_uuid_to_ns_map(self):
        specs = [
            AstraJobSpec(
                uuid=10, shape=(2,),
                bw=[[0.0, 1.0], [1.0, 0.0]],
                lt=[[0.0, 1.0], [1.0, 0.0]],
            ),
            AstraJobSpec(
                uuid=11, shape=(2,),
                bw=[[0.0, 2.0], [2.0, 0.0]],
                lt=[[0.0, 2.0], [2.0, 0.0]],
            ),
        ]
        # Return ns proportional to uuid so we can distinguish outputs.
        def fake_run_astra(**kwargs):
            return float(kwargs["uuid"]) * 100.0
        with patch.object(
            astra_runner.astra_sim, "run_astra", side_effect=fake_run_astra
        ):
            result = self.runner.runMany(specs)
        self.assertEqual(result, {10: 1000.0, 11: 1100.0})

    def test_run_many_reraises_first_worker_failure(self):
        specs = [
            AstraJobSpec(
                uuid=1, shape=(2,),
                bw=[[0.0, 1.0], [1.0, 0.0]],
                lt=[[0.0, 1.0], [1.0, 0.0]],
            ),
        ]
        with patch.object(
            astra_runner.astra_sim,
            "run_astra",
            side_effect=RuntimeError("simulated worker failure"),
        ):
            with self.assertRaises(RuntimeError):
                self.runner.runMany(specs)

    def test_run_many_with_empty_specs_returns_empty_dict(self):
        # No specs => no submits, immediate empty map.
        self.assertEqual(self.runner.runMany([]), {})


class TestAstraSimRunnerShutdown(unittest.TestCase):
    def test_shutdown_idempotent_and_blocks_subsequent_runs(self):
        executor = ThreadPoolExecutor(max_workers=1)
        runner = AstraSimRunner(tmp_root=Path("/tmp"), executor=executor)
        runner.shutdown()
        # A second shutdown() must not raise.
        runner.shutdown()
        # Submitting after shutdown must raise (executor refuses new work).
        with self.assertRaises(RuntimeError):
            runner.runMany([
                AstraJobSpec(
                    uuid=1, shape=(2,),
                    bw=[[0.0, 1.0], [1.0, 0.0]],
                    lt=[[0.0, 1.0], [1.0, 0.0]],
                )
            ])


class TestAstraSimRunnerDefaults(unittest.TestCase):
    def test_default_executor_is_process_pool_sized_by_cpu_count(self):
        # Build a runner without injecting an executor, immediately
        # shut it down. Verify the pool was created with the expected
        # max_workers.
        import os
        from concurrent.futures import ProcessPoolExecutor

        runner = AstraSimRunner(tmp_root=Path("/tmp"))
        try:
            self.assertIsInstance(runner._pool, ProcessPoolExecutor)
            # Implementation detail: ProcessPoolExecutor stores its
            # configured max workers in _max_workers.
            self.assertEqual(runner._pool._max_workers, os.cpu_count() - 2)
        finally:
            runner.shutdown()


if __name__ == "__main__":
    unittest.main()
