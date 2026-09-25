'Concurrency regression tests for the native index binding.'

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.native_index import NativeIndex, TaskState


def label_for(value: int) -> str:
    'Create a deterministic canonical test label.'
    return f"{value:0{512}x}"


class NativeIndexConcurrencyTest(unittest.TestCase):
    'Verify the ABI, idempotence, and state-machine races.'

    @classmethod
    def setUpClass(cls) -> None:
        'Compile the exact source under test to the packaged runtime path.'
        if shutil.which("g++") is None:
            raise unittest.SkipTest("g++ is required for native-index tests /  g++")
        suffix = ".dll" if sys.platform == "win32" else ".so"
        cls.library_path = PROJECT_ROOT / "src" / "dbtfl" / "native" / f"atomic_word{suffix}"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_native.py",
                "--output",
                str(cls.library_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def new_index(self) -> NativeIndex:
        'Create an index sized for all concurrency cases.'
        return NativeIndex(128, 64, 256, library_path=self.library_path)

    def test_native_and_python_reject_noncanonical_labels(self) -> None:
        'Reject uppercase labels at both FFI boundaries.'
        with self.new_index() as index:
            task_id = ctypes.c_int()
            accepted = index._library.dbt_index_register_label(
                index._require_open(), b"A" * 512, 1, 0, ctypes.byref(task_id)
            )
            self.assertEqual(accepted, 0)
            with self.assertRaises(ValueError):
                index.register_label("a" * 511 + "A", 1, 0)

    def test_concurrent_retries_create_exactly_one_owner_edge(self) -> None:
        'Ensure same-client concurrent retries remain idempotent.'
        label = label_for(11)
        with self.new_index() as index:
            workers = 32
            start = threading.Barrier(workers)

            def register_once() -> int:
                start.wait()
                return index.register_label(label, 1, 3)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                task_ids = list(executor.map(lambda _: register_once(), range(workers)))

            self.assertEqual(set(task_ids), {task_ids[0]})
            self.assertEqual(index.find_label(label), task_ids[0])
            self.assertEqual(index.task_label(task_ids[0]), label)
            self.assertEqual(index.owners(task_ids[0]), (1,))
            self.assertEqual(index.client_tasks(1), (task_ids[0],))
            self.assertEqual(index.edge_count, 1)

    def test_concurrent_claim_has_one_winner_and_owner_only_commits(self) -> None:
        'Verify CAS grants one trainer and rejects stale completion.'
        label = label_for(12)
        with self.new_index() as index:
            task_id = index.register_label(label, 1, 0)
            trainers = tuple(range(1, 33))
            for trainer in trainers[1:]:
                self.assertEqual(index.register_label(label, trainer, 0), task_id)

            start = threading.Barrier(len(trainers))

            def claim_once(trainer: int) -> tuple[int, bool]:
                start.wait()
                return trainer, index.try_claim(task_id, trainer)

            with ThreadPoolExecutor(max_workers=len(trainers)) as executor:
                outcomes = list(executor.map(claim_once, trainers))

            winners = [trainer for trainer, claimed in outcomes if claimed]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            snapshot = index.snapshot(task_id)
            self.assertEqual(snapshot.state, TaskState.PENDING)
            self.assertEqual(snapshot.trainer, winner)
            loser = next(trainer for trainer in trainers if trainer != winner)
            self.assertFalse(index.mark_committed(task_id, loser))
            self.assertTrue(index.mark_committed(task_id, winner))
            self.assertEqual(index.snapshot(task_id).state, TaskState.COMMITTED)
            self.assertFalse(index.release_if_trainer(task_id, winner))

    def test_release_and_reset_increment_versions(self) -> None:
        'Verify release and reset preserve ABA-version progression.'
        with self.new_index() as index:
            task_id = index.register_label(label_for(13), 1, 7)
            self.assertEqual(index.snapshot(task_id).version, 0)
            self.assertTrue(index.try_claim(task_id, 1))
            snapshot = index.snapshot(task_id)
            self.assertEqual(
                (snapshot.state, snapshot.trainer, snapshot.version),
                (TaskState.PENDING, 1, 1),
            )
            self.assertTrue(index.release_if_trainer(task_id, 1))
            snapshot = index.snapshot(task_id)
            self.assertEqual(
                (snapshot.state, snapshot.trainer, snapshot.version),
                (TaskState.EMPTY, 0, 2),
            )
            self.assertTrue(index.try_claim(task_id, 2))
            self.assertTrue(index.mark_committed(task_id, 2))
            snapshot = index.snapshot(task_id)
            self.assertEqual(
                (snapshot.state, snapshot.trainer, snapshot.version),
                (TaskState.COMMITTED, 2, 4),
            )
            index.reset(task_id)
            snapshot = index.snapshot(task_id)
            self.assertEqual(
                (snapshot.state, snapshot.trainer, snapshot.version),
                (TaskState.EMPTY, 0, 5),
            )
            self.assertGreater(index.memory_bytes, 0)


if __name__ == "__main__":
    unittest.main()
