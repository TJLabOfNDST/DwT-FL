'Integration checks for the paper-metric evaluation runner.'

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
import unittest
import json
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.evaluation.runner import (
    EvaluationPlan,
    EvaluationRunner,
    _Case,
    _TrainingFailureContext,
    _build_cases,
    _communication_metrics,
    _current_training_participants,
    _failure_victim_count,
    _heartbeat_instruction_snapshots,
    _merged_interval_seconds,
    _select_training_failure_victims,
    _global_model_distribution_workers,
    _load_completed_resume_prefix,
    _json_plan,
    _round_count_for_case,
    _selected_cases,
    _synthetic_records_for_case,
    _trimmed_mean_case_result,
    _write_reports,
)
from dbtfl.entities import (
    AggregationServerConfig,
    AggregationServerEntity,
    ClientConfig,
    ClientEntity,
    KeyServerConfig,
    KeyServerEntity,
    LocalTrainingQueues,
    ModelUpdateOwnershipLostError,
    RoundInstruction,
    TaskClaimDecision,
)
from dbtfl.training import PreparedRecord
from dbtfl.communication import (
    AggregationServerPath,
    TrafficRecord,
    TrafficRecorder,
    TransportError,
)
from dbtfl.communication.endpoints import KeyServerPath
from dbtfl.oprf import native_backend_available


def _completed_resume_result(
    case: _Case,
    *,
    repetition: int,
    marker: str,
) -> dict[str, object]:
    'Build the smallest reportable successful row for resume tests.'
    return {
        "schema_version": "1.0",
        "status": "completed",
        "suite": case.suite,
        "variable": case.variable,
        "value": case.value,
        "repetition": repetition,
        "marker": marker,
        "training_mode": "simulated",
        "oprf_suite": "test-suite",
        "service_mode": "isolated",
        "configuration": {
            "clients": case.clients,
            "request_workers": case.request_workers,
            "duplicate_ratio": case.duplicate_ratio,
            "backend_workers": case.backend_workers,
            "records_per_client": case.records_per_client,
            "failure_phase": case.failure_phase,
            "joining_clients": case.joining_clients,
            "ablation": case.ablation,
        },
        "total_completion_seconds": 1.0,
        "dedup_wall_seconds": 0.2,
        "dedup_accumulated_seconds": 0.3,
        "training_wall_seconds": 0.4,
        "training_accumulated_seconds": 0.5,
        "recovery_latency_seconds": None,
        "aggregation": {"total_repetitions": 1},
        "submitted_client_count": case.clients,
        "ownership_retrain_count": 0,
        "federated_rounds": 1,
        "model_upload_accumulated_seconds": 0.0,
        "metadata_bytes": {
            "protocol_metadata_bytes": 16,
            "total_metadata_bytes": 16,
            "model_artifact_bytes": 0,
        },
    }


class EvaluationRunnerIntegrationTest(unittest.TestCase):
    'Verify one separate AS process produces complete, formatted evidence.'

    def test_arrival_schedule_is_reproducible_and_asynchronous(self) -> None:
        'Keep paired client arrivals reproducible without serial ordering.'
        runner = EvaluationRunner(EvaluationPlan(
            output_directory=Path(tempfile.gettempdir()) / "dbtfl-arrival-test",
            training_mode="simulated",
            client_arrival_min_delay_seconds=1.0,
            client_arrival_max_delay_seconds=5.0,
        ))
        clients = [
            SimpleNamespace(config=SimpleNamespace(client_id=f"client-{index}"))
            for index in range(4)
        ]
        case = _Case("parallel_client_scale", "clients", 4, 4, 4, 0.3, 4, 16)
        first = runner._arrival_delays(clients, case, 0)
        second = runner._arrival_delays(clients, case, 0)
        self.assertEqual(first, second)
        self.assertGreaterEqual(min(first.values()), 1.0)
        self.assertLessEqual(max(first.values()), 5.0)
        self.assertGreater(len(set(first.values())), 1)

    def test_training_fault_executes_every_planned_disconnect_before_recovery_polling(self) -> None:
        'Never truncate a multi-client fault when one heartbeat RPC is slow.'
        class _FakeClient:
            'Minimal connected client used to isolate the event schedule.'

            def __init__(self, client_id: str, sid: int) -> None:
                self.config = SimpleNamespace(client_id=client_id, as_base_url="http://test")
                self.as_session = SimpleNamespace(sid=sid)
                self.last_round_instructions: list[object] = []
                self.stop_count = 0
                self.reconnect_count = 0
                self.heartbeat_count = 0

            def _stop_heartbeat_worker(self) -> None:
                self.stop_count += 1

            def connect_to_as(self) -> None:
                self.reconnect_count += 1

            def send_as_heartbeat(self) -> None:
                self.heartbeat_count += 1
                return None

            def send_as_heartbeat_with_instructions(self) -> tuple[None, tuple[object, ...]]:
                'Return the response-local empty instruction snapshot.'
                self.heartbeat_count += 1
                return None, ()

        runner = EvaluationRunner(EvaluationPlan(
            output_directory=Path(tempfile.gettempdir()) / "dbtfl-staggered-fault-test",
            training_mode="simulated",
            heartbeat_interval_seconds=0.01,
            failure_heartbeat_timeout_seconds=0.02,
            failure_disconnect_initial_delay_seconds=0.01,
            failure_disconnect_interval_seconds=0.01,
        ))
        victims = tuple(_FakeClient(f"client-{index}", index + 1) for index in range(4))
        survivor = _FakeClient("survivor", 99)
        context = _TrainingFailureContext(
            victims=victims,  # type: ignore[arg-type]
            survivors=(survivor,),  # type: ignore[arg-type]
            victim_task_ids={victim.config.client_id: set() for victim in victims},
            victim_sids_before_disconnect={
                victim.config.client_id: victim.as_session.sid for victim in victims
            },
            disconnect_started_at_by_client={},
            recovery_target_ids=set(),
            started=threading.Event(),
            completed=threading.Event(),
            lock=threading.Lock(),
        )

        # The metrics endpoint is reporting-only. A transient read failure
        # cannot invalidate a complete recovery protocol observation.
        # metrics
        with patch(
            "dbtfl.evaluation.runner._fetch_as_metrics",
            side_effect=TransportError("metrics unavailable / "),
        ):
            runner._run_training_failure_monitor(context)

        self.assertIsNone(context.error)
        self.assertEqual(len(context.disconnect_events or []), len(victims))
        self.assertEqual([victim.stop_count for victim in victims], [1, 1, 1, 1])
        self.assertEqual([victim.reconnect_count for victim in victims], [0, 0, 0, 0])
        self.assertGreaterEqual(survivor.heartbeat_count, 2)

    def test_recovery_instruction_snapshots_dispatch_all_survivors_in_parallel(self) -> None:
        'Dispatch one recovery heartbeat concurrently for every live SID.'
        client_count = 4
        gate = threading.Event()
        lock = threading.Lock()
        state = {"entered": 0, "active": 0, "maximum_active": 0}

        class _LiveClient:
            'Minimal live client exposing the protocol dispatch boundary.'

            def __init__(self, client_id: str) -> None:
                self.config = SimpleNamespace(client_id=client_id)

            def send_as_heartbeat_with_instructions(self) -> tuple[None, tuple[object, ...]]:
                'Wait until every live SID has entered its heartbeat request.'
                with lock:
                    state["entered"] += 1
                    state["active"] += 1
                    state["maximum_active"] = max(
                        state["maximum_active"], state["active"]
                    )
                    if state["entered"] == client_count:
                        gate.set()
                if not gate.wait(timeout=1.0):
                    raise RuntimeError("heartbeat dispatch was not parallel")
                # Keep the request overlap observable after the final caller.
                
                time.sleep(0.01)
                with lock:
                    state["active"] -= 1
                return None, ()

        clients = [_LiveClient(f"client-{index}") for index in range(client_count)]
        snapshots = _heartbeat_instruction_snapshots(clients)  # type: ignore[arg-type]

        self.assertEqual(set(snapshots), {client.config.client_id for client in clients})
        self.assertEqual(state["maximum_active"], client_count)

    def test_formal_arrival_schedule_pins_a_late_client_at_nineteen_seconds(self) -> None:
        'Require the 0--20-second formal workload to contain the late client.'
        runner = EvaluationRunner(EvaluationPlan(
            output_directory=Path(tempfile.gettempdir()) / "dbtfl-arrival-anchor-test",
            training_mode="simulated",
        ))
        clients = [
            SimpleNamespace(config=SimpleNamespace(client_id=f"client-{index}"))
            for index in range(4)
        ]
        case = _Case("parallel_client_scale", "clients", 4, 4, 4, 0.3, 4, 1024)
        delays = runner._arrival_delays(clients, case, 2)

        self.assertEqual(delays["client-0"], 0.0)
        self.assertEqual(delays["client-3"], 19.0)

    def test_arrival_delay_bounds_reject_an_inverted_interval(self) -> None:
        'Reject an arrival range whose maximum precedes its minimum.'
        with self.assertRaisesRegex(ValueError, "maximum delay"):
            EvaluationPlan(
                output_directory=Path(tempfile.gettempdir()) / "dbtfl-arrival-invalid",
                training_mode="simulated",
                client_arrival_min_delay_seconds=5.0,
                client_arrival_max_delay_seconds=1.0,
            )

    def test_active_interval_union_excludes_idle_arrival_gaps(self) -> None:
        'Keep arrival gaps out of the active protocol-time metric.'
        self.assertAlmostEqual(
            _merged_interval_seconds(((1.0, 2.0), (1.5, 3.0), (5.0, 5.5))),
            2.5,
        )

    def test_model_submissions_are_started_concurrently(self) -> None:
        'Require independently trained clients to enter upload concurrently.'
        runner = EvaluationRunner(EvaluationPlan(
            output_directory=Path(tempfile.gettempdir()) / "dbtfl-upload-test",
            training_mode="simulated",
        ))
        barrier = threading.Barrier(2, timeout=1.0)
        clients = []
        claims = {"values": {}}
        queues = {}
        checkpoints = {}
        for index in range(2):
            identifier = f"client-{index}"
            decision = TaskClaimDecision("a" * 512 if index == 0 else "b" * 512,
                                         index + 1, "TRAIN", "PENDING")
            client = Mock()
            client.config = SimpleNamespace(client_id=identifier)
            client.as_session = SimpleNamespace(sid=index + 1)
            client.send_as_heartbeat.return_value = client.as_session
            client.submit_model_update_at_as.side_effect = (
                lambda *_args, **_kwargs: barrier.wait()
            )
            clients.append(client)
            claims["values"][identifier] = ([decision], 0.0)
            queues[identifier] = LocalTrainingQueues((f"record-{index}".encode(),), ())
            checkpoints[identifier] = Path(tempfile.gettempdir()) / f"{identifier}.safetensors"
        result = runner._submit_updates(
            Path(tempfile.gettempdir()), clients, claims, queues,
            {"checkpoints": checkpoints, "wall_seconds": 0.0, "accumulated_seconds": 0.0},
            repetition=0,
        )
        self.assertEqual(result["submitted_client_count"], 2)
        self.assertEqual(set(result["submitted_sids"]), {1, 2})
        self.assertLess(result["wall_seconds"], 1.0)

    def test_training_dropout_prefers_a_trainer_with_an_online_duplicate_owner(self) -> None:
        'Do not inject a recoverability test into exclusive-only work.'
        exclusive = SimpleNamespace(config=SimpleNamespace(client_id="exclusive"))
        recoverable = SimpleNamespace(config=SimpleNamespace(client_id="recoverable"))
        alternate = SimpleNamespace(config=SimpleNamespace(client_id="alternate"))
        idle = SimpleNamespace(config=SimpleNamespace(client_id="idle"))
        claims = {
            "values": {
                "exclusive": ([TaskClaimDecision("a" * 512, 1, "TRAIN", "PENDING")], 0.0),
                "recoverable": ([TaskClaimDecision("b" * 512, 2, "TRAIN", "PENDING")], 0.0),
                "alternate": ([TaskClaimDecision("b" * 512, 2, "DEDUP", "PENDING")], 0.0),
                "idle": ([], 0.0),
            },
        }

        victims = _select_training_failure_victims(
            [exclusive, recoverable, alternate, idle], claims, requested_count=1
        )

        self.assertEqual([client.config.client_id for client in victims], ["recoverable"])

    def test_communication_metrics_keep_model_transport_out_of_protocol_metadata(self) -> None:
        'Separate actual OPRF/control JSON bytes from model JSON bytes.'
        recorder = TrafficRecorder()
        for path, request_bytes, response_bytes in (
            (KeyServerPath.EVALUATE_OPRF.value, 100, 80),
            (AggregationServerPath.HEARTBEAT.value, 20, 30),
            (AggregationServerPath.SUBMIT_MODEL_UPDATE.value, 1000, 50),
        ):
            recorder.record(TrafficRecord(
                base_url="http://127.0.0.1:1",
                path=path,
                method="POST",
                request_message_type="test.request",
                response_message_type="test.response",
                request_body_bytes=request_bytes,
                response_body_bytes=response_bytes,
                request_payload_bytes=request_bytes,
                response_payload_bytes=response_bytes,
                status_code=200,
                elapsed_seconds=0.01,
                error_type=None,
            ))
        client = SimpleNamespace(config=SimpleNamespace(traffic_recorder=recorder))
        metrics = _communication_metrics([client], {"oprf_evaluation_compute_seconds": 0.005})

        self.assertEqual(metrics["bytes"]["oprf_communication_bytes"], 180)
        self.assertEqual(metrics["bytes"]["heartbeat_communication_bytes"], 50)
        self.assertEqual(metrics["bytes"]["protocol_metadata_communication_bytes"], 230)
        self.assertEqual(metrics["bytes"]["model_transport_communication_bytes"], 1050)
        self.assertEqual(metrics["bytes"]["total_client_service_communication_bytes"], 1280)
        self.assertEqual(metrics["counts"]["oprf_http_exchange_count"], 1)

    @unittest.skipUnless(
        shutil.which("g++") and native_backend_available(),
        "g++ and native libsodium Ristretto255 are required for evaluation tests / "
        " g++  libsodium Ristretto255",
    )
    def test_one_protocol_case_writes_required_metric_fields(self) -> None:
        'Exercise OPRF, CAS, submission, metadata, and report rendering.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            plan = EvaluationPlan(
                output_directory=root / "report",
                heartbeat_interval_seconds=0.05,
                heartbeat_timeout_seconds=3.0,
                training_mode="simulated",
            )
            case = _Case("smoke", "clients", 2, clients=2, request_workers=2,
                         duplicate_ratio=0.5, backend_workers=2, records_per_client=2)
            result = EvaluationRunner(plan)._run_case(case, repetition=0)
            output = _write_reports(plan, [result])

            self.assertGreater(result["total_completion_seconds"], 0.0)
            self.assertGreater(result["dedup_accumulated_seconds"], 0.0)
            self.assertEqual(result["submitted_client_count"], 2)
            self.assertGreater(result["metadata_bytes"]["as_native_index_bytes"], 0)
            self.assertTrue((output / "results.json").is_file())
            self.assertTrue((output / "case_metrics.csv").is_file())
            self.assertTrue((output / "REPORT.en.md").is_file())
            self.assertTrue((output / "REPORT.zh-CN.md").is_file())

    @unittest.skipUnless(
        shutil.which("g++") and native_backend_available(),
        "g++ and native libsodium Ristretto255 are required for evaluation tests / "
        " g++  libsodium Ristretto255",
    )
    def test_training_dropout_starts_with_a_long_setup_lease_then_recovers(self) -> None:
        'Exercise the lease-transition order through a complete local case.\n        This regression intentionally makes the failure lease far shorter than\n        normal setup work. It would fail at label registration if the short\n        lease were supplied while the AS starts, which is the production bug\n        this test prevents.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            plan = EvaluationPlan(
                output_directory=root / "report",
                training_mode="simulated",
                heartbeat_interval_seconds=0.03,
                heartbeat_timeout_seconds=2.0,
                failure_heartbeat_timeout_seconds=0.10,
                rpc_timeout_seconds=3.0,
            )
            case = _Case(
                "fault_training", "failure_rate", 0.5,
                clients=4, request_workers=4, duplicate_ratio=1.0,
                backend_workers=4, records_per_client=2, failure_phase="training",
            )

            result = EvaluationRunner(plan)._run_case(case, repetition=0)

            self.assertEqual(result["failure_victim_count"], 2)
            self.assertEqual(len(result["fault_recoveries"]), 2)
            self.assertTrue(any(
                item["recovery_latency_seconds"] is not None
                for item in result["fault_recoveries"]
            ), result["fault_recoveries"])

    def test_resume_from_case_retains_only_the_validated_completed_prefix(self) -> None:
        'Continue from case two without re-running or duplicating case one.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name) / "report"
            first = _Case("first", "clients", 2, 2, 2, 0.3, 2, 4)
            second = _Case("second", "clients", 4, 4, 4, 0.3, 2, 4)
            preserved = _completed_resume_result(first, repetition=0, marker="preserved")
            root.mkdir(parents=True)
            plan = EvaluationPlan(
                output_directory=root,
                training_mode="simulated",
                start_case=2,
                repetitions=1,
            )
            (root / "results.json").write_text(
                json.dumps({"plan": _json_plan(plan), "results": [preserved]}),
                encoding="utf-8",
            )
            runner = EvaluationRunner(plan, progress=lambda _message: None)
            replacement = _completed_resume_result(second, repetition=0, marker="new")
            with patch("dbtfl.evaluation.runner._build_cases", return_value=(first, second)), patch.object(
                runner, "_ensure_prepared_data"
            ), patch("dbtfl.evaluation.runner._write_run_metadata"), patch.object(
                runner, "_run_case", return_value=replacement
            ) as run_case:
                runner.run()

            run_case.assert_called_once_with(second, 0)
            stored = json.loads((root / "results.json").read_text(encoding="utf-8"))[
                "results"
            ]
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[0]["marker"], "preserved")
            self.assertEqual(stored[1]["marker"], "new")
            status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["completed_cases"], 2)

    def test_resume_rejects_a_failed_or_mismatched_prefix(self) -> None:
        'Reject stale prefix rows before any new case starts.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            case = _Case("first", "clients", 2, 2, 2, 0.3, 2, 4)
            invalid = _completed_resume_result(case, repetition=0, marker="failed")
            invalid["status"] = "failed"
            plan = EvaluationPlan(output_directory=root, training_mode="simulated", start_case=2, repetitions=1)
            (root / "results.json").write_text(
                json.dumps({"plan": _json_plan(plan), "results": [invalid]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not completed"):
                _load_completed_resume_prefix(root, plan, ((case, 0),), start_case=2)

            completed = _completed_resume_result(case, repetition=0, marker="completed")
            incompatible_plan = EvaluationPlan(
                output_directory=root,
                training_mode="simulated",
                rpc_timeout_seconds=99.0,
                start_case=2,
                repetitions=1,
            )
            (root / "results.json").write_text(
                json.dumps({"plan": _json_plan(incompatible_plan), "results": [completed]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "plan does not match"):
                _load_completed_resume_prefix(root, plan, ((case, 0),), start_case=2)

    def test_gpt_assignment_uses_real_prepared_text_and_controlled_overlap(self) -> None:
        'Ensure the real mode never constructs synthetic training text.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            source = root / "train.jsonl"
            prepared = tuple(
                PreparedRecord(str(index), f"actual haiku {index}", {"source": "haiku"})
                for index in range(12)
            )
            source.write_text(
                "".join(json.dumps(record.to_json_object()) + "\n" for record in prepared),
                encoding="utf-8",
            )
            plan = EvaluationPlan(
                output_directory=root / "report",
                training_mode="gpt",
                prepared_data_path=source,
            )
            case = _Case(
                "real-data",
                "clients",
                2,
                clients=2,
                request_workers=1,
                duplicate_ratio=0.5,
                backend_workers=1,
                records_per_client=4,
                joining_clients=1,
            )
            assignments = EvaluationRunner(plan)._records_for_case(case, repetition=0)

            source_texts = {record.text for record in prepared}
            self.assertEqual(set(assignments), {"client-0", "client-1", "client-2"})
            self.assertTrue(all(
                set(records).issubset(source_texts) for records in assignments.values()
            ))
            # Reference-style overlaps are pairwise. No universal shared pool
            # may make one duplicate appear in every client.
            
            first = set(assignments["client-0"])
            second = set(assignments["client-1"])
            third = set(assignments["client-2"])
            self.assertEqual(
                len(first.intersection(second))
                + len(first.intersection(third))
                + len(second.intersection(third)),
                2,
            )
            self.assertFalse(first.intersection(second).intersection(third))
            self.assertFalse(any("shared-" in record or "unique-" in record
                                 for records in assignments.values() for record in records))

    @unittest.skipUnless(
        shutil.which("g++") and native_backend_available(),
        "g++ and native libsodium Ristretto255 are required for evaluation tests / "
        " g++  libsodium Ristretto255",
    )
    def test_remote_mode_uses_existing_as_and_ks_without_local_children(self) -> None:
        'Verify the real evaluator targets supplied endpoints and resets AS state.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            suffix = ".dll" if sys.platform == "win32" else ".so"
            library_path = (
                PROJECT_ROOT / "results" / "native-test-artifacts" /
                f"atomic_word_remote_evaluation{suffix}"
            )
            subprocess.run(
                [sys.executable, "scripts/build_native.py", "--output", str(library_path)],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            as_entity = AggregationServerEntity(AggregationServerConfig(
                capacity=32,
                max_clients=8,
                max_edges=64,
                host="127.0.0.1",
                port=0,
                heartbeat_interval_seconds=0.5,
                heartbeat_timeout_seconds=2.0,
                native_library_path=library_path,
                model_update_directory=root / "as-updates",
                evaluation_reset_token="remote-test-token",
            ))
            key_server = KeyServerEntity(KeyServerConfig(
                key_path=root / "ks-key.json", host="127.0.0.1", port=0
            ))
            as_entity.start()
            key_server.start()
            try:
                plan = EvaluationPlan(
                    output_directory=root / "report",
                    training_mode="simulated",
                    service_mode="remote",
                    as_base_url=as_entity.base_url,
                    ks_base_url=key_server.base_url,
                    evaluation_reset_token="remote-test-token",
                    rpc_timeout_seconds=2.0,
                    heartbeat_interval_seconds=0.5,
                    heartbeat_timeout_seconds=2.0,
                )
                case = _Case("remote-smoke", "clients", 2, clients=2, request_workers=2,
                             duplicate_ratio=0.5, backend_workers=2, records_per_client=2)

                result = EvaluationRunner(plan)._run_case(case, repetition=0)

                self.assertEqual(result["service_mode"], "remote")
                self.assertEqual(result["submitted_client_count"], 2)
                self.assertEqual(as_entity.service.heartbeat_timeout_seconds, 2.0)
                self.assertIn(result["as_resources"]["status"], {"available", "unavailable"})
                self.assertGreater(result["metadata_bytes"]["ks_private_key_bytes"], 0)
            finally:
                key_server.close()
                as_entity.close()

    def test_dynamic_join_cost_is_written_from_matched_base_run(self) -> None:
        'Verify the paper definition C_inc equals T_join minus T_base.'
        with tempfile.TemporaryDirectory() as temporary_name:
            plan = EvaluationPlan(Path(temporary_name))
            common = {
                "metadata_bytes": {"total_metadata_bytes": 0}, "value": 1,
                "dedup_wall_seconds": 0.0, "dedup_accumulated_seconds": 0.0,
                "training_wall_seconds": 0.0, "training_accumulated_seconds": 0.0,
                "recovery_latency_seconds": None,
            }
            results = [
                {"suite": "dynamic_join_base", "repetition": 0, "total_completion_seconds": 2.0,
                 **common},
                {"suite": "dynamic_join", "repetition": 0, "total_completion_seconds": 2.75,
                 **common},
            ]
            _write_reports(plan, results)
            self.assertEqual(results[1]["incremental_cost_seconds"], 0.75)

    def test_four_run_trimmed_mean_excludes_end_to_end_extremes(self) -> None:
        'Retain only middle two observations in a formal four-run case.'
        rows = [
            {
                "schema_version": "1.0", "status": "completed", "suite": "scale",
                "variable": "clients", "value": 2, "configuration": {"clients": 2},
                "training_mode": "gpt", "oprf_suite": "ristretto", "service_mode": "isolated",
                "repetition": index, "total_completion_seconds": float(index),
                "dedup_wall_seconds": float(index * 10),
            }
            for index in range(1, 5)
        ]
        result = _trimmed_mean_case_result(rows)
        self.assertEqual(result["aggregation"]["retained_repetitions"], [2, 3])
        self.assertEqual(result["aggregation"]["discarded_repetitions"], [1, 4])
        self.assertEqual(result["total_completion_seconds"], 2.5)
        self.assertEqual(result["dedup_wall_seconds"], 25.0)
        self.assertNotIn("repetition", result)

    def test_training_dropout_activates_short_lease_only_after_protocol_setup(self) -> None:
        'Keep registration/CAS on the long lease before fault injection.'
        with tempfile.TemporaryDirectory() as temporary_name:
            plan = EvaluationPlan(
                output_directory=Path(temporary_name),
                heartbeat_interval_seconds=5.0,
                heartbeat_timeout_seconds=300.0,
                failure_heartbeat_timeout_seconds=15.0,
            )
            runner = EvaluationRunner(plan)
            dropout = _Case(
                "fault_training", "failure_rate", 0.5, 2, 1, 0.5, 1, 2,
                failure_phase="training",
            )

            # _run_case must initialize both ordinary and dropout cases with
            # plan.heartbeat_timeout_seconds. The short value is consumed only
            # inside _configure_training_failure_lease after CAS completes.
            # _run_case
            
            self.assertEqual(runner._initial_heartbeat_timeout_seconds(dropout), 300.0)
            self.assertEqual(runner.plan.failure_heartbeat_timeout_seconds, 15.0)

    @patch("dbtfl.entities.client.JsonHttpClient")
    def test_client_heartbeat_transport_does_not_inherit_a_long_model_timeout(
        self,
        transport_type: Mock,
    ) -> None:
        'Bound heartbeat connection stalls independently from large RPCs.'
        with tempfile.TemporaryDirectory() as temporary_name:
            ClientEntity(ClientConfig(
                client_id="timeout-client",
                ks_base_url="http://127.0.0.1:18081",
                as_base_url="http://127.0.0.1:18080",
                label_store_path=Path(temporary_name) / "labels.json",
                timeout_seconds=300.0,
                heartbeat_rpc_timeout_seconds=2.0,
            ))

        timeouts = [call.kwargs["timeout_seconds"] for call in transport_type.call_args_list]
        self.assertEqual(timeouts, [300.0, 300.0, 2.0, 2.0])

    @patch("dbtfl.evaluation.runner.JsonHttpClient")
    def test_training_fault_lease_switch_does_not_emit_a_heartbeat_burst(
        self,
        transport_type: Mock,
    ) -> None:
        'Let AS atomically refresh leases without ten duplicate heartbeats.'
        runner = EvaluationRunner(EvaluationPlan(
            output_directory=Path(tempfile.gettempdir()) / "dbtfl-lease-switch-test",
            heartbeat_interval_seconds=0.1,
            heartbeat_timeout_seconds=300.0,
            failure_heartbeat_timeout_seconds=0.5,
        ))
        clients = []
        for index in range(10):
            client = Mock()
            client.config = SimpleNamespace(as_base_url="http://127.0.0.1:18080")
            clients.append(client)
        transport_type.return_value.send.return_value = SimpleNamespace(
            message_type="as.evaluation.lease.response",
            payload={
                "heartbeat_interval_seconds": 0.1,
                "heartbeat_timeout_seconds": 0.5,
            },
        )

        runner._configure_training_failure_lease(clients, "control-token")

        for client in clients:
            client.send_as_heartbeat.assert_not_called()
        transport_type.return_value.send.assert_called_once()

    def test_dedup_dropout_defers_all_as_work_until_the_client_rejoins(self) -> None:
        'Keep a dedup-phase victim outside AS registration and CAS.'
        with tempfile.TemporaryDirectory() as temporary_name:
            runner = EvaluationRunner(EvaluationPlan(output_directory=Path(temporary_name)))
            victim = Mock()
            victim.config = SimpleNamespace(client_id="client-0")
            survivor = Mock()
            survivor.config = SimpleNamespace(client_id="client-1")
            case = _Case(
                "fault_dedup",
                "failure_rate",
                0.1,
                2,
                2,
                0.3,
                2,
                2,
                failure_phase="dedup",
            )

            _, victims = runner._inject_dedup_dropout(
                case,
                [victim, survivor],
                {"client-0": ["record"], "client-1": ["other"]},
            )

            self.assertEqual(victims, (victim,))
            victim.generate_protected_labels.assert_called_once_with(["record"])
            victim.close.assert_called_once_with()
            victim.connect_to_as.assert_not_called()

    def test_dedup_rejoin_uses_parallel_protocol_requests_without_redundant_heartbeats(self) -> None:
        'Reuse the fresh rejoin lease for registration and CAS.'
        with tempfile.TemporaryDirectory() as temporary_name:
            runner = EvaluationRunner(EvaluationPlan(output_directory=Path(temporary_name)))
            client = Mock()
            client.config = SimpleNamespace(client_id="client-0")
            client.register_records_with_as.return_value = ["registered-label"]
            client.claim_registered_labels_at_as.return_value = []
            client.route_claim_decisions.return_value = LocalTrainingQueues((), ())

            runner._rejoin_dedup_dropouts(
                [client],
                {"client-0": ["record"]},
            )

        client.connect_to_as.assert_called_once_with()
        client.register_records_with_as.assert_called_once_with(
            ["record"],
            created_round=1,
            refresh_lease=False,
        )
        client.claim_registered_labels_at_as.assert_called_once_with(
            ["registered-label"],
            refresh_lease=False,
        )

    def test_paper_scale_lite_defaults_cover_required_axes_and_real_failure_counts(self) -> None:
        'Keep the local default aligned with feasible reference-paper axes.'
        with tempfile.TemporaryDirectory() as temporary_name:
            plan = EvaluationPlan(output_directory=Path(temporary_name))
            cases = tuple(_build_cases(plan))
            self.assertEqual(plan.base_clients, 10)
            self.assertEqual(plan.client_counts, (2, 4, 6, 8, 10))
            self.assertEqual(plan.duplicate_ratios, (0.0, 0.10, 0.30, 0.50, 0.70, 0.90))
            self.assertEqual(plan.failure_rates, (0.10, 0.20, 0.40, 0.60, 0.80))
            self.assertEqual(plan.join_client_counts, (1, 2, 4, 6))
            self.assertEqual(plan.ablation_failure_rate, 0.40)
            self.assertEqual(plan.clients_per_gpu, 5)
            self.assertEqual(plan.gpu_memory_fraction_per_client, 0.20)
            self.assertEqual(plan.repetitions, 4)
            self.assertEqual(plan.client_arrival_min_delay_seconds, 0.0)
            self.assertEqual(plan.client_arrival_max_delay_seconds, 20.0)
            self.assertEqual(plan.global_model_download_workers, 10)
            self.assertEqual(plan.records_per_client_values[-1], 1536)
            self.assertEqual(len(cases), 39)
            dropout = next(
                case for case in cases
                if case.suite == "fault_training" and case.value == 0.80
            )
            self.assertEqual(_failure_victim_count(dropout, dropout.clients), 8)
            self.assertEqual(
                {case.ablation for case in cases if case.suite == "ablation"},
                {
                    "full_dwtfl_single_round", "without_cas", "without_inverse_index",
                    "full_dwtfl_history", "without_history_scheduling",
                },
            )
            self.assertEqual(
                [case.joining_clients for case in cases if case.suite == "dynamic_join"],
                [1, 2, 4, 6],
            )
            history = next(
                case for case in cases if case.ablation == "without_history_scheduling"
            )
            self.assertEqual(_round_count_for_case(plan, history), 2)
            simulated_plan = EvaluationPlan(
                output_directory=Path(temporary_name) / "simulation",
                training_mode="simulated",
            )
            self.assertEqual(_round_count_for_case(simulated_plan, history), 1)

    def test_formal_plan_rejects_cached_oprf_without_explicit_diagnostic_opt_in(self) -> None:
        'Prevent a cache directory from silently contaminating formal results.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            with self.assertRaisesRegex(ValueError, "formal evaluation requires live OPRF"):
                EvaluationPlan(
                    output_directory=root / "formal",
                    precomputed_oprf_directory=root / "cached-labels",
                )
            diagnostic = EvaluationPlan(
                output_directory=root / "diagnostic",
                precomputed_oprf_directory=root / "cached-labels",
                require_live_oprf=False,
            )
            self.assertFalse(diagnostic.require_live_oprf)

    def test_global_model_distribution_has_a_separate_bounded_read_concurrency(self) -> None:
        'Limit read-only model distribution without altering protocol fan-out.'
        with tempfile.TemporaryDirectory() as temporary_name:
            plan = EvaluationPlan(
                output_directory=Path(temporary_name), global_model_download_workers=2
            )
            high_backend = _Case("distribution", "clients", 10, 10, 10, 0.3, 8, 2)
            single_backend = _Case("distribution", "backend", 1, 10, 10, 0.3, 1, 2)

            self.assertEqual(_global_model_distribution_workers(plan, high_backend, 10), 2)
            self.assertEqual(_global_model_distribution_workers(plan, single_backend, 10), 2)
            self.assertEqual(_global_model_distribution_workers(plan, high_backend, 1), 1)

    def test_focused_recovery_and_ablation_plan_selects_only_prior_failed_cases(self) -> None:
        'Select configured recovery rates and their full-flow ablation controls.'
        with tempfile.TemporaryDirectory() as temporary_name:
            plan = EvaluationPlan(
                output_directory=Path(temporary_name),
                failure_rates=(0.10, 0.30, 0.50, 0.70),
                included_suites=("fault_training", "ablation"),
            )
            cases = _selected_cases(plan)

        self.assertEqual(len(cases), 9)
        self.assertEqual(
            [(case.suite, case.value) for case in cases if case.suite == "fault_training"],
            [("fault_training", 0.10), ("fault_training", 0.30),
             ("fault_training", 0.50), ("fault_training", 0.70)],
        )
        self.assertEqual(
            {case.ablation for case in cases if case.suite == "ablation"},
            {
                "full_dwtfl_single_round", "without_cas", "without_inverse_index",
                "full_dwtfl_history", "without_history_scheduling",
            },
        )

    def test_post_upload_train_assignment_replaces_cumulative_checkpoint(self) -> None:
        'Incrementally train a new heartbeat task before the fixed roster aggregates.\n        This unit-level runner regression isolates the client loop from model\n        download dependencies. It proves the loop retains the old task ID\n        trains only the new hot record from the old checkpoint, then uploads a\n        replacement with both task IDs and the cumulative sample count.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            old_checkpoint = root / "old.safetensors"
            new_checkpoint = root / "new.safetensors"
            old_checkpoint.write_bytes(b"old")
            new_checkpoint.write_bytes(b"new")
            plan = EvaluationPlan(output_directory=root / "report", training_mode="gpt")
            runner = EvaluationRunner(plan)
            old_decision = TaskClaimDecision("a" * 512, 1, "TRAIN", "PENDING")
            new_decision = TaskClaimDecision("b" * 512, 2, "TRAIN", "PENDING")
            client = Mock()
            client.config = SimpleNamespace(client_id="client-1")
            client.last_round_instructions = ()
            # The runner must use the response-local instruction snapshot,
            # rather than mutable client state populated by a background
            # heartbeat.
            
            client.send_as_heartbeat_with_instructions.return_value = (
                None,
                (RoundInstruction(new_decision.protected_label, 2, "TRAIN"),),
            )
            client.route_claim_decisions.return_value = LocalTrainingQueues(
                hot_records=(b"new record",),
                cold_records=(),
            )
            claims = {"values": {"client-1": ([old_decision], 0.0)}}
            queues = {"client-1": LocalTrainingQueues((b"old record",), ())}
            training = {
                "wall_seconds": 0.0,
                "accumulated_seconds": 0.0,
                "checkpoints": {"client-1": old_checkpoint},
                "client_metrics": {},
            }
            submissions = {
                "submitted_decisions": {"client-1": (old_decision,)},
                "submitted_sample_counts": {"client-1": 1},
            }
            incremental_training = {
                "wall_seconds": 0.2,
                "accumulated_seconds": 0.2,
                "checkpoints": {"client-1": new_checkpoint},
                "client_metrics": {"client-1": {"cuda": {"available": True}}},
            }
            with patch.object(runner, "_run_training", return_value=incremental_training) as run_training:
                result = runner._apply_pre_aggregate_incremental_training(
                    root,
                    [client],
                    claims,
                    queues,
                    training,
                    submissions,
                    repetition=0,
                    round_id=1,
                )

            self.assertEqual(result["replacement_count"], 1)
            run_training.assert_called_once()
            self.assertEqual(queues["client-1"].hot_records, (b"old record", b"new record"))
            self.assertEqual(submissions["submitted_sample_counts"]["client-1"], 2)
            self.assertEqual(submissions["submitted_decisions"]["client-1"], (old_decision, new_decision))
            client.submit_model_update_at_as.assert_called_once_with(
                new_checkpoint,
                (old_decision, new_decision),
                round_id=1,
                sample_count=2,
            )

    def test_protocol_simulation_allocates_records_for_late_joiners(self) -> None:
        'Keep protocol-only dynamic joining structurally equal to real allocation.'
        case = _Case(
            "dynamic_join",
            "joining_clients",
            2,
            clients=2,
            request_workers=2,
            duplicate_ratio=0.30,
            backend_workers=2,
            records_per_client=4,
            joining_clients=2,
        )
        records = _synthetic_records_for_case(case, repetition=0)
        self.assertEqual(set(records), {"client-0", "client-1", "client-2", "client-3"})
        self.assertTrue(all(len(values) == 4 for values in records.values()))

    def test_recovery_sync_replaces_stale_claims_before_fedavg_roster_selection(self) -> None:
        'Use full post-recovery heartbeat instructions for roster decisions.'
        with tempfile.TemporaryDirectory() as temporary_name:
            runner = EvaluationRunner(EvaluationPlan(output_directory=Path(temporary_name)))
            client = Mock()
            client.config = SimpleNamespace(client_id="recovering-client")
            client.last_round_instructions = (
                SimpleNamespace(
                    protected_label="a" * 512,
                    task_id=11,
                    operation="TRAIN",
                ),
                SimpleNamespace(
                    protected_label="b" * 512,
                    task_id=12,
                    operation="DEDUP",
                ),
            )
            client.route_claim_decisions.side_effect = lambda decisions: LocalTrainingQueues(
                tuple(b"hot" for decision in decisions if decision.operation == "TRAIN"),
                tuple(b"cold" for decision in decisions if decision.operation == "DEDUP"),
            )
            # Deliberately leave the convenience cache stale. The recovery
            # synchronizer must trust the instructions returned by its own
            # heartbeat response, not an overlapping worker's cached view.
            
            
            heartbeat_snapshot = (
                SimpleNamespace(
                    protected_label="d" * 512,
                    task_id=14,
                    operation="DEDUP",
                ),
            )
            response_instructions = client.last_round_instructions
            client.last_round_instructions = heartbeat_snapshot
            client.send_as_heartbeat_with_instructions.return_value = (
                SimpleNamespace(sid=1),
                response_instructions,
            )
            stale = TaskClaimDecision("c" * 512, 13, "TRAIN", "PENDING")
            claims = {"values": {"recovering-client": ([stale], 0.7)}}
            queues = {"recovering-client": LocalTrainingQueues((b"stale",), ())}

            runner._synchronize_recovered_training_ownership([client], claims, queues)
            participants = _current_training_participants([client], claims, queues)

        client.send_as_heartbeat_with_instructions.assert_called_once_with()
        self.assertEqual(client.last_round_instructions, heartbeat_snapshot)
        self.assertEqual(participants, [client])
        self.assertEqual(
            [(decision.protected_label, decision.operation)
             for decision in claims["values"]["recovering-client"][0]],
            [("a" * 512, "TRAIN"), ("b" * 512, "DEDUP")],
        )
        self.assertEqual(claims["values"]["recovering-client"][1], 0.7)
        self.assertEqual(queues["recovering-client"], LocalTrainingQueues((b"hot",), (b"cold",)))

    def test_roster_rejects_stale_hot_queue_without_a_train_decision(self) -> None:
        'Reject a local queue that could otherwise create a missing update.'
        client = Mock()
        client.config = SimpleNamespace(client_id="stale-client")
        claims = {
            "values": {
                "stale-client": ([TaskClaimDecision("d" * 512, 14, "DEDUP", "PENDING")], 0.0),
            },
        }
        queues = {"stale-client": LocalTrainingQueues((b"must-not-train",), ())}

        with self.assertRaisesRegex(RuntimeError, "claims and hot queue disagree"):
            _current_training_participants([client], claims, queues)

    def test_training_fault_defers_offline_client_work_to_the_next_round(self) -> None:
        'Keep a disconnected SID out of the current FedAvg roster.'
        client = SimpleNamespace(config=SimpleNamespace(client_id="offline-client"))
        decision = TaskClaimDecision("a" * 512, 7, "TRAIN", "PENDING")
        claims = {"values": {"offline-client": ([decision], 0.0)}}
        queues = {
            "offline-client": LocalTrainingQueues((b"shared record",), (b"cold",))
        }

        EvaluationRunner._defer_offline_training_clients([client], claims, queues)

        deferred = claims["values"]["offline-client"][0]
        self.assertEqual([item.operation for item in deferred], ["DEDUP"])
        self.assertEqual(queues["offline-client"].hot_records, ())
        self.assertEqual(
            set(queues["offline-client"].cold_records), {b"shared record", b"cold"}
        )
        self.assertEqual(_current_training_participants([client], claims, queues), [])

    def test_ownership_loss_retrains_only_reclaimed_hot_records(self) -> None:
        'Move transferred data cold, then retrain only the remaining TRAIN data.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            plan = EvaluationPlan(output_directory=root, training_mode="simulated")
            runner = EvaluationRunner(plan)
            keep = TaskClaimDecision("a" * 512, 3, "TRAIN", "PENDING")
            transferred = TaskClaimDecision("b" * 512, 4, "TRAIN", "PENDING")
            fresh_keep = TaskClaimDecision("a" * 512, 3, "TRAIN", "PENDING")
            fresh_transferred = TaskClaimDecision("b" * 512, 4, "DEDUP", "PENDING")
            client = Mock()
            client.config = SimpleNamespace(client_id="recovering-client")
            client.submit_model_update_at_as.side_effect = [
                ModelUpdateOwnershipLostError(
                    "taken over / ",
                    dedup_instructions=(fresh_transferred,),
                ),
                None,
            ]
            client.claim_protected_labels_at_as.return_value = [fresh_keep, fresh_transferred]
            client.reconcile_recovery_claims.side_effect = lambda decisions, **_kwargs: list(decisions)
            client.route_claim_decisions.return_value = LocalTrainingQueues(
                (b"keep",),
                (b"transferred",),
            )
            original_checkpoint = root / "old.safetensors"
            original_checkpoint.write_bytes(b"old checkpoint")
            claims = {"values": {"recovering-client": ([keep, transferred], 0.0)}}
            queues = {"recovering-client": LocalTrainingQueues((b"keep", b"transferred"), ())}
            training = {
                "wall_seconds": 0.0,
                "accumulated_seconds": 0.0,
                "checkpoints": {"recovering-client": original_checkpoint},
            }

            result = runner._submit_updates(
                root,
                [client],
                claims,
                queues,
                training,
                repetition=0,
            )

            self.assertEqual(result["submitted_client_count"], 1)
            self.assertEqual(result["ownership_retrain_count"], 1)
            client.claim_protected_labels_at_as.assert_called_once_with(("a" * 512, "b" * 512))
            self.assertEqual(queues["recovering-client"].hot_records, (b"keep",))
            self.assertEqual(queues["recovering-client"].cold_records, (b"transferred",))
            self.assertEqual(
                [(decision.protected_label, decision.operation)
                 for decision in claims["values"]["recovering-client"][0]],
                [("a" * 512, "TRAIN"), ("b" * 512, "DEDUP")],
            )
            retry_arguments = client.submit_model_update_at_as.call_args_list[1].args
            self.assertEqual(retry_arguments[1], [fresh_keep])
            self.assertNotEqual(retry_arguments[0], original_checkpoint)

    def test_runner_skips_a_failed_case_and_persists_its_diagnostics(self) -> None:
        'Continue independent cases and persist one complete failure record.'
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            plan = EvaluationPlan(
                output_directory=root / "report",
                training_mode="simulated",
                repetitions=1,
                prepared_data_path=PurePosixPath("/real-data/prepared-haiku/train.jsonl"),
            )
            # The failure is intentional and asserted below. Suppress progress
            # output so a successful test run is not visually indistinguishable
            # from a real evaluation failure.
            
            runner = EvaluationRunner(plan, progress=lambda _message: None)
            completed_result = {
                "schema_version": "1.0", "suite": "persisted-case", "variable": "clients", "value": 2,
                "repetition": 0, "training_mode": "simulated",
                "oprf_suite": "test-suite", "service_mode": "isolated",
                "configuration": {},
                "total_completion_seconds": 1.0, "dedup_wall_seconds": 0.2,
                "dedup_accumulated_seconds": 0.3, "training_wall_seconds": 0.4,
                "training_accumulated_seconds": 0.5, "recovery_latency_seconds": None,
                "submitted_client_count": 2,
                "metadata_bytes": {"total_metadata_bytes": 16},
            }

            cases = (
                _Case("persisted-case", "clients", 2, 2, 1, 0.5, 1, 2),
                _Case("failed-case", "clients", 4, 4, 1, 0.5, 1, 2),
            )
            with patch.object(runner, "_ensure_prepared_data"), patch(
                "dbtfl.evaluation.runner._build_cases", return_value=cases
            ), patch.object(
                runner,
                "_run_case",
                side_effect=[completed_result, RuntimeError("intentional second-case failure")],
            ):
                output = runner.run()

            raw = json.loads((root / "report" / "results.json").read_text(encoding="utf-8"))
            status = json.loads((root / "report" / "run_status.json").read_text(encoding="utf-8"))
            failures = json.loads((root / "report" / "failed_cases.json").read_text(encoding="utf-8"))
            # The runner returns an absolute canonical path. On Windows, a
            # temporary directory may be presented once with an 8.3 component
            # and once with its long name, so compare normalized paths rather
            # than host-specific spellings.
            
            
            self.assertEqual(output, (root / "report").resolve())
            self.assertEqual(len(raw["results"]), 2)
            self.assertEqual(raw["results"][0]["status"], "completed")
            self.assertEqual(raw["results"][1]["status"], "failed")
            self.assertEqual(
                raw["results"][1]["failure"]["failed_repetitions"][0]["type"],
                "RuntimeError",
            )
            self.assertEqual(raw["plan"]["prepared_data_path"], "/real-data/prepared-haiku/train.jsonl")
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["completed_cases"], 2)
            self.assertEqual(status["succeeded_cases"], 1)
            self.assertEqual(status["failed_cases"], 1)
            self.assertEqual(failures["failed_cases"][0]["suite"], "failed-case")
            self.assertIn("Failed cases", (root / "report" / "REPORT.en.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
