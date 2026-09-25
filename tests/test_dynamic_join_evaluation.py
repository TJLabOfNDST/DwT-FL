'Unit tests for the paired DwT-FL dynamic-join evaluator.\nDwT-FL'

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.evaluation.dynamic_join_evaluation import (
    DynamicJoinEvaluationPlan,
    DynamicJoinEvaluationRunner,
    _load_or_create_arrival_schedule,
    _trimmed_result,
    _write_markdown,
)
from dbtfl.evaluation.oprf_precompute import allocate_joining_records
from dbtfl.evaluation.runner import _Case, _prepared_records_for_case
from dbtfl.evaluation.runner import EvaluationRunner
from dbtfl.training import PreparedRecord


class DynamicJoinScheduleTest(unittest.TestCase):
    'Verify schedule reuse and trimmed reporting without GPU execution.'

    def test_reuses_ndss_compatible_fixed_schedule(self) -> None:
        'The persisted 1--5 second schedule is loaded verbatim on rerun.'
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = DynamicJoinEvaluationPlan(
                output_directory=root / "out",
                prepared_data_path=root / "train.jsonl",
                arrival_schedule_path=root / "shared-arrivals.json",
            )
            first = _load_or_create_arrival_schedule(plan)
            second = _load_or_create_arrival_schedule(plan)
            self.assertEqual(first, second)
            self.assertEqual(set(first["scheduled_delay_seconds"]), {
                "joining-client-0", "joining-client-1", "joining-client-2",
                "joining-client-3", "joining-client-4", "joining-client-5",
                "joining-client-6",
            })
            self.assertTrue(all(1.0 <= value <= 5.0 for value in first["scheduled_delay_seconds"].values()))
            self.assertEqual(json.loads(plan.arrival_schedule_path.read_text(encoding="utf-8")), first)

    def test_trimmed_result_discards_endpoint_extremes(self) -> None:
        'Four runs retain only the middle two by the declared endpoint.'
        rows = []
        for value in (10.0, 20.0, 30.0, 40.0):
            rows.append({
                "value": 3,
                "configuration": {},
                "oprf_suite": "test",
                "base_client_count": 10,
                "joining_client_count": 3,
                "input_record_count": 3072,
                "arrival_schedule": {},
                "ks_oprf_delta": {"oprf_evaluated_element_count": value, "oprf_evaluation_compute_seconds": value},
                "first_join_started_after_base_seconds": value,
                "last_join_registered_after_base_seconds": value,
                "join_arrival_span_seconds": value,
                "join_end_to_end_to_training_start_seconds": value,
                "join_dedup_after_first_join_seconds": value,
                "join_dedup_after_all_joined_seconds": value,
                "join_global_model_download_wall_seconds": value,
                "join_training_launch_span_seconds": value,
                "join_training_completion_wall_seconds": value,
                "join_training_completion_accumulated_seconds": value,
            })
        result = _trimmed_result(rows)
        self.assertEqual(result["trimmed_repetitions"], 2)
        self.assertEqual(result["join_end_to_end_to_training_start_seconds"], 25.0)

    def test_fixed_base_and_seven_joiners_fit_the_real_training_pool(self) -> None:
        'The 10+7 allocation must fit 14,414 prepared training records.\n        10+7'
        source = tuple(
            PreparedRecord(str(index), f"record-{index}", {})
            for index in range(14_414)
        )
        case = _Case(
            suite="dynamic_join_base_allocation",
            variable="base_clients",
            value=10,
            clients=10,
            request_workers=10,
            duplicate_ratio=0.30,
            backend_workers=4,
            records_per_client=1024,
        )
        base = _prepared_records_for_case(case, 0, source, 0.30)
        joiners = allocate_joining_records(
            source, base, joining_clients=7, duplicate_ratio=0.30, seed=17
        )
        self.assertEqual({len(values) for values in base.values()}, {1024})
        self.assertEqual({len(values) for values in joiners.values()}, {1024})

    def test_resume_retains_a_completed_prefix_and_markdown_uses_root_directory(self) -> None:
        'A report-write interruption must not repeat completed GPU cases.'
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = DynamicJoinEvaluationPlan(
                output_directory=root / "out",
                prepared_data_path=root / "train.jsonl",
                arrival_schedule_path=root / "arrivals.json",
            )
            plan.output_directory.mkdir()
            completed = {
                "status": "completed",
                "suite": "dynamic_join",
                "value": 1,
                "join_end_to_end_to_training_start_seconds": 1.0,
            }
            (plan.output_directory / "results.json").write_text(
                json.dumps({"results": [completed]}), encoding="utf-8"
            )
            runner = DynamicJoinEvaluationRunner(plan, progress=lambda _message: None)
            self.assertEqual(runner._load_completed_prefix(), [completed])
            _write_markdown(plan.output_directory, [completed])
            self.assertTrue((plan.output_directory / "README.md").is_file())

    def test_raw_repetition_log_is_durable_before_a_later_failure(self) -> None:
        'A completed repeat must survive a later repeat failure.'
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan = DynamicJoinEvaluationPlan(
                output_directory=root / "out",
                prepared_data_path=root / "train.jsonl",
                arrival_schedule_path=root / "arrivals.json",
            )
            runner = DynamicJoinEvaluationRunner(plan, progress=lambda _message: None)
            runner._append_raw_repeat({
                "status": "completed",
                "joining_clients": 7,
                "repetition": 1,
                "result": {"join_end_to_end_to_training_start_seconds": 1.0},
            })
            runner._append_raw_repeat({
                "status": "failed",
                "joining_clients": 7,
                "repetition": 2,
                "failure": {"type": "TransportError"},
            })
            log = plan.output_directory / "raw_dynamic_join_repetitions.jsonl"
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["completed", "failed"])
            self.assertEqual(rows[0]["result"]["join_end_to_end_to_training_start_seconds"], 1.0)

    def test_base_upload_fanout_can_be_bounded_to_as_capacity(self) -> None:
        'The base round retains parallel uploads but honors the AS worker cap.'
        clients = [
            SimpleNamespace(config=SimpleNamespace(client_id=f"client-{index}"))
            for index in range(5)
        ]
        claims = {
            "values": {
                client.config.client_id: ([SimpleNamespace(operation="TRAIN")], 0.0)
                for client in clients
            }
        }
        queues = {
            client.config.client_id: SimpleNamespace(hot_records=("record",))
            for client in clients
        }

        def fake_parallel_phase(selected, workers, _operation):
            'Return complete upload records without exercising network I/O.'
            self.assertEqual(workers, 4)
            return {
                "wall_seconds": 0.0,
                "accumulated_seconds": 0.0,
                "values": {
                    client.config.client_id: ({
                        "sid": index + 1,
                        "decisions": (),
                        "sample_count": 1,
                        "ownership_retrain_count": 0,
                        "submitted": True,
                        "upload_elapsed_seconds": 0.0,
                        "trained_task_count": 1,
                        "trained_sample_count": 1,
                    }, 0.0)
                    for index, client in enumerate(selected)
                },
            }

        with patch("dbtfl.evaluation.runner._parallel_phase", side_effect=fake_parallel_phase):
            result = EvaluationRunner._submit_updates(
                object(),
                Path("."),
                clients,
                claims,
                queues,
                {"checkpoints": {}},
                repetition=0,
                max_submit_workers=4,
            )
        self.assertEqual(result["submitted_client_count"], 5)


if __name__ == "__main__":
    unittest.main()
