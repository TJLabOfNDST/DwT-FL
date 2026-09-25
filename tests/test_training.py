'Cross-platform tests for preparation, FedAvg math, and GPU job isolation.'

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.federation import FedAvgError, fedavg_arrays
from dbtfl.training import (
    ClientTrainingJob,
    DistilledGptTrainingConfig,
    iter_prepared_records,
    materialize_hot_training_split,
    mps_partitioning_status,
    prepare_csv_dataset,
    run_client_training_jobs,
)
from dbtfl.training.gpt import _resolve_gpu_memory_fraction


class TrainingPreparationTest(unittest.TestCase):
    'Verify deterministic text-only preparation without GPU dependencies.'

    def test_preparation_ignores_export_index_and_keeps_metadata_local(self) -> None:
        'Ensure only configured text becomes JSONL text for model input.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["Unnamed: 0", "id", "processed_title", "ups", "keywords"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Unnamed: 0": "0",
                        "id": "a",
                        "processed_title": "first poem",
                        "ups": "5",
                        "keywords": "first,poem",
                    }
                )
                writer.writerow(
                    {
                        "Unnamed: 0": "1",
                        "id": "b",
                        "processed_title": "second poem",
                        "ups": "2",
                        "keywords": "second,poem",
                    }
                )
            manifest = prepare_csv_dataset(source, root / "prepared", validation_fraction=0.0)
            records = list(iter_prepared_records(root / "prepared" / "records.jsonl"))

            self.assertEqual((manifest.total_records, manifest.train_records), (2, 2))
            self.assertEqual([record.text for record in records], ["first poem", "second poem"])
            self.assertEqual(records[0].metadata, {"ups": "5", "keywords": "first,poem"})
            self.assertNotIn(
                "Unnamed: 0",
                (root / "prepared" / "records.jsonl").read_text(encoding="utf-8"),
            )
            manifest_payload = json.loads(
                (root / "prepared" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest_payload["source_sha256"], manifest.source_sha256)

    def test_hot_split_uses_only_train_records_and_removes_local_duplicates(self) -> None:
        'Ensure DEDUP records cannot enter the client-local model input.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            source.write_text(
                "id,processed_title,keywords\n"
                "one,hot poem,hot\n"
                "two,cold poem,cold\n"
                "three,hot poem,duplicate\n",
                encoding="utf-8",
            )
            prepare_csv_dataset(source, root / "prepared", validation_fraction=0.0)
            selected_count = materialize_hot_training_split(
                root / "prepared" / "train.jsonl",
                [b"hot poem"],
                root / "client-hot.jsonl",
            )
            selected_records = list(iter_prepared_records(root / "client-hot.jsonl"))

            self.assertEqual(selected_count, 1)
            self.assertEqual([record.text for record in selected_records], ["hot poem"])


class FedAvgTest(unittest.TestCase):
    'Verify weighted aggregation independently from checkpoint libraries.'

    def test_weighted_fedavg_and_nonfloating_buffer_guard(self) -> None:
        'Verify sample weighting and reject incompatible integer buffers.'
        first = {
            "weight": numpy.array([1.0, 3.0], dtype=numpy.float32),
            "counter": numpy.array([7], dtype=numpy.int64),
        }
        second = {
            "weight": numpy.array([5.0, 9.0], dtype=numpy.float32),
            "counter": numpy.array([7], dtype=numpy.int64),
        }
        averaged = fedavg_arrays([first, second], [1, 3])
        numpy.testing.assert_allclose(averaged["weight"], numpy.array([4.0, 7.5]))
        self.assertEqual(averaged["counter"].tolist(), [7])

        second["counter"] = numpy.array([8], dtype=numpy.int64)
        with self.assertRaises(FedAvgError):
            fedavg_arrays([first, second], [1, 3])


class GpuSchedulerTest(unittest.TestCase):
    'Verify physical GPU isolation through subprocess environment variables.'

    def test_trainer_records_the_scheduler_supplied_fixed_share(self) -> None:
        "Resolve the scheduler's fixed share before CUDA training starts."
        with patch.dict(os.environ, {"DBTFL_GPU_MEMORY_FRACTION": "0.20"}, clear=False):
            self.assertEqual(
                _resolve_gpu_memory_fraction(
                    DistilledGptTrainingConfig(output_directory=Path("artifacts"))
                ),
                0.20,
            )

    def test_scheduler_exposes_one_assigned_gpu_to_each_process(self) -> None:
        'Ensure each queued job sees exactly its assigned CUDA device.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jobs = [
                ClientTrainingJob(
                    client_id=f"client-{index}",
                    command=(
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
                    ),
                    log_path=root / f"client-{index}.log",
                )
                for index in range(3)
            ]
            results = run_client_training_jobs(jobs, [2, 7])

            self.assertEqual(len(results), 3)
            self.assertTrue(all(result.return_code == 0 for result in results))
            self.assertTrue(all(result.gpu_id in {2, 7} for result in results))
            for result in results:
                self.assertEqual(
                    result.log_path.read_text(encoding="utf-8").strip(),
                    str(result.gpu_id),
                )

    def test_scheduler_notifies_before_each_training_child_starts(self) -> None:
        'Expose a real process-start boundary for fault injection.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jobs = [
                ClientTrainingJob(
                    client_id=f"callback-client-{index}",
                    command=(sys.executable, "-c", "print('trained')"),
                    log_path=root / f"callback-client-{index}.log",
                )
                for index in range(2)
            ]
            started: list[str] = []
            results = run_client_training_jobs(
                jobs,
                [0],
                on_job_start=lambda job: started.append(job.client_id),
            )

            self.assertTrue(all(result.return_code == 0 for result in results))
            self.assertCountEqual(started, [job.client_id for job in jobs])

    def test_scheduler_evenly_shares_four_clients_across_two_gpus(self) -> None:
        'Verify two concurrent bounded slots are created on each GPU.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jobs = [
                ClientTrainingJob(
                    client_id=f"shared-client-{index}",
                    command=(
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['DBTFL_GPU_MEMORY_FRACTION'])",
                    ),
                    log_path=root / f"shared-client-{index}.log",
                )
                for index in range(4)
            ]
            results = run_client_training_jobs(jobs, [0, 1], clients_per_gpu=2)

            self.assertEqual([result.gpu_id for result in results].count(0), 2)
            self.assertEqual([result.gpu_id for result in results].count(1), 2)
            self.assertTrue(all(result.gpu_memory_fraction == 0.5 for result in results))
            for result in results:
                self.assertEqual(result.log_path.read_text(encoding="utf-8").strip(), "0.5")

    def test_scheduler_terminates_a_timed_out_client_process(self) -> None:
        'Bound a hung trainer so a later evaluation case can still execute.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            job = ClientTrainingJob(
                client_id="timed-out-client",
                command=(sys.executable, "-c", "import time; time.sleep(5)"),
                log_path=root / "timed-out-client.log",
            )
            result = run_client_training_jobs([job], [0], timeout_seconds=0.05)[0]

            self.assertTrue(result.timed_out)
            self.assertEqual(result.return_code, 124)
            self.assertIn("timed out", result.log_path.read_text(encoding="utf-8"))

    def test_scheduler_cancels_only_a_deliberately_disconnected_trainer(self) -> None:
        'Terminate a faulted child without classifying it as a timeout.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            job = ClientTrainingJob(
                client_id="disconnected-client",
                command=(sys.executable, "-c", "import time; time.sleep(5)"),
                log_path=root / "disconnected-client.log",
            )
            requested_at = time.monotonic() + 0.05
            result = run_client_training_jobs(
                [job],
                [0],
                cancel_requested=lambda _job: time.monotonic() >= requested_at,
            )[0]

            self.assertTrue(result.cancelled)
            self.assertFalse(result.timed_out)
            self.assertNotEqual(result.return_code, 0)
            self.assertIn("deliberately cancelled", result.log_path.read_text(encoding="utf-8"))

    def test_fixed_twenty_percent_share_runs_ten_client_scale_without_queueing(self) -> None:
        'Give every client the same 20-percent share in the scale suite.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jobs = [
                ClientTrainingJob(
                    client_id=f"scale-client-{index}",
                    command=(
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['DBTFL_GPU_MEMORY_FRACTION'])",
                    ),
                    log_path=root / f"scale-client-{index}.log",
                )
                for index in range(10)
            ]
            results = run_client_training_jobs(
                jobs,
                [0, 1],
                clients_per_gpu=5,
                gpu_memory_fraction=0.20,
            )

            self.assertEqual(len(results), 10)
            self.assertTrue(all(result.return_code == 0 for result in results))
            self.assertTrue(all(result.gpu_memory_fraction == 0.20 for result in results))
            for gpu_id in (0, 1):
                gpu_results = [result for result in results if result.gpu_id == gpu_id]
                self.assertEqual(len(gpu_results), 5)
                self.assertEqual({result.slot_index for result in gpu_results}, {0, 1, 2, 3, 4})
            for result in results:
                self.assertEqual(result.log_path.read_text(encoding="utf-8").strip(), "0.2")

    def test_fixed_share_requires_mps_before_claiming_compute_partitioning(self) -> None:
        'Reject a fairness run when CUDA MPS cannot enforce compute shares.'
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            job = ClientTrainingJob(
                client_id="mps-client",
                command=(sys.executable, "-c", "print('unreachable')"),
                log_path=root / "mps-client.log",
            )
            with patch(
                "dbtfl.training.scheduler.mps_partitioning_status",
                return_value={"available": False, "reason": "test MPS unavailable"},
            ):
                with self.assertRaisesRegex(RuntimeError, "MPS"):
                    run_client_training_jobs(
                        [job],
                        [0],
                        clients_per_gpu=5,
                        gpu_memory_fraction=0.20,
                        require_mps_partitioning=True,
                    )

    def test_mps_probe_rejects_zero_exit_when_control_reports_missing_daemon(self) -> None:
        'Do not trust a misleading zero exit status from an absent MPS daemon.'
        with patch("dbtfl.training.scheduler.sys.platform", "linux"), patch(
            "dbtfl.training.scheduler.shutil.which", return_value="mps-control"
        ), patch(
            "dbtfl.training.scheduler.subprocess.run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="Cannot find MPS control daemon process",
            ),
        ):
            status = mps_partitioning_status()
        self.assertFalse(status["available"])
        self.assertIn("Cannot find", str(status["reason"]))


if __name__ == "__main__":
    unittest.main()
