'Integration tests for chunked client model updates and task commits.'

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.entities import (
    AggregationServerConfig,
    AggregationServerEntity,
    ClientConfig,
    ClientEntity,
    KeyServerConfig,
    KeyServerEntity,
    ModelUpdateOwnershipLostError,
    TaskClaimDecision,
)
from dbtfl.communication import (
    AggregationServerPath,
    JsonHttpClient,
    RemoteServiceError,
    TransportError,
    WireMessage,
)
from dbtfl.entities.aggregation_server import (
    AS_CONFIGURE_ROUND_REQUEST,
    AS_MODEL_AGGREGATE_REQUEST,
    AS_MODEL_AGGREGATE_RESPONSE,
    AS_MODEL_CHUNK_REQUEST,
)
from dbtfl.native_index import TaskState


class ModelUpdateOwnershipLostErrorTest(unittest.TestCase):
    'Verify the recovery exception survives all supported Python runtimes.'

    def test_constructor_retains_structured_recovery_instructions(self) -> None:
        'Construct the exception without zero-argument-super binding errors.'
        transferred = TaskClaimDecision("a" * 512, 7, "DEDUP", "PENDING")
        error = ModelUpdateOwnershipLostError(
            "ownership transferred / ",
            dedup_instructions=(transferred,),
            recovered_decisions=(transferred,),
        )

        self.assertEqual(str(error), "ownership transferred / ")
        self.assertEqual(error.dedup_instructions, (transferred,))
        self.assertEqual(error.recovered_decisions, (transferred,))


class ModelUpdateIntegrationTest(unittest.TestCase):
    "Verify an update commits only the submitting SID's PENDING tasks."

    @classmethod
    def setUpClass(cls) -> None:
        'Build an isolated native library for model-update tests.'
        if shutil.which("g++") is None:
            raise unittest.SkipTest("g++ is required for model-update tests /  g++")
        suffix = ".dll" if sys.platform == "win32" else ".so"
        cls.library_path = (
            PROJECT_ROOT
            / "results"
            / "native-test-artifacts"
            / f"atomic_word_model_update_test{suffix}"
        )
        subprocess.run(
            [sys.executable, "scripts/build_native.py", "--output", str(cls.library_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    def setUp(self) -> None:
        'Start temporary HTTP services and a private model-update directory.'
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._root = Path(self._temporary_directory.name)
        key_path = self._root / "ks-key.json"
        self._ks = KeyServerEntity(KeyServerConfig(key_path=key_path, host="127.0.0.1", port=0))
        self._ks.start()
        self.addCleanup(self._ks.close)
        self._as = AggregationServerEntity(
            AggregationServerConfig(
                capacity=32,
                max_clients=4,
                max_edges=64,
                host="127.0.0.1",
                port=0,
                native_library_path=self.library_path,
                model_update_directory=self._root / "model-updates",
                heartbeat_interval_seconds=0.05,
                heartbeat_timeout_seconds=0.4,
            )
        )
        self._as.start()
        self.addCleanup(self._as.close)

    def test_chunked_update_commits_only_trained_pending_tasks(self) -> None:
        'Verify digest-checked upload precedes native PENDING-to-COMMITTED commit.'
        client = ClientEntity(
            ClientConfig(
                client_id="model-update-client",
                ks_base_url=self._ks.base_url,
                as_base_url=self._as.base_url,
                label_store_path=self._root / "labels.json",
                model_chunk_bytes=8,
            )
        )
        client.connect_to_as()
        self.addCleanup(client.close)
        registrations = client.register_records_with_as(["train-me"], created_round=8)
        decisions = client.claim_registered_labels_at_as(registrations)
        self.assertEqual(decisions[0].operation, "TRAIN")
        checkpoint_path = self._root / "local_model.safetensors"
        checkpoint_path.write_bytes(b"test checkpoint bytes for chunked HTTP transfer")

        submission = client.submit_model_update_at_as(
            checkpoint_path,
            decisions,
            round_id=8,
            sample_count=1,
        )

        self.assertEqual(submission.committed_task_ids, (decisions[0].task_id,))
        snapshot = self._as.index.snapshot(decisions[0].task_id)
        self.assertEqual(snapshot.state, TaskState.COMMITTED)
        stored_updates = list((self._root / "model-updates" / "round-8").glob("*.safetensors"))
        self.assertEqual(len(stored_updates), 1)
        self.assertEqual(stored_updates[0].read_bytes(), checkpoint_path.read_bytes())

    def test_chunk_replay_recovers_after_a_lost_success_response(self) -> None:
        'Replay the identical chunk when its successful response is lost.'
        client = self._connected_client("lost-chunk-response")
        decision = self._claim_one(client, "replay model record", 13)
        checkpoint_path = self._write_checkpoint("replay", b"chunk replay must be idempotent")
        assert client._as_transport is not None
        flaky_transport = _LoseFirstModelChunkResponse(client._as_transport)
        client._as_transport = flaky_transport

        submission = client.submit_model_update_at_as(
            checkpoint_path,
            [decision],
            round_id=13,
            sample_count=1,
        )

        self.assertTrue(flaky_transport.did_drop_response)
        self.assertEqual(submission.committed_task_ids, (decision.task_id,))
        stored_updates = list((self._root / "model-updates" / "round-13").glob("*.safetensors"))
        self.assertEqual(len(stored_updates), 1)
        self.assertEqual(stored_updates[0].read_bytes(), checkpoint_path.read_bytes())

    def test_global_model_read_retries_only_after_a_transport_failure(self) -> None:
        'Retry an immutable global-model read without replaying a mutation.'
        client = self._connected_client("global-model-read-retry")
        assert client._as_transport is not None
        request = WireMessage.create(
            "as.global_model.chunk.request",
            {"sid": 1, "round": 1, "offset": 0, "max_bytes": 8},
        )
        expected = WireMessage.create("as.global_model.chunk.response", {})
        with patch.object(
            client._as_transport,
            "send",
            side_effect=[TransportError("lost read response / "), expected],
        ) as send, patch("dbtfl.entities.client.time.sleep") as sleep:
            actual = client._send_global_model_chunk_with_retry(request)

        self.assertIs(actual, expected)
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once()

    def test_upload_refreshes_liveness_after_checkpoint_hashing(self) -> None:
        'Refresh the AS lease between expensive hashing and the first chunk.'
        client = self._connected_client("post-hash-heartbeat")
        decision = self._claim_one(client, "post hash heartbeat record", 14)
        checkpoint_path = self._write_checkpoint("post-hash", b"hashed before upload")
        assert client._as_transport is not None
        recording_transport = _RecordingTransport(client._as_transport)
        client._as_transport = recording_transport

        client.submit_model_update_at_as(
            checkpoint_path,
            [decision],
            round_id=14,
            sample_count=1,
        )

        first_chunk = recording_transport.events.index("model_chunk")
        self.assertEqual(recording_transport.events[first_chunk - 1], "heartbeat")

    def test_upload_reconnects_and_reclaims_before_replaying_offline_chunk(self) -> None:
        'Recover a pre-write offline upload only after repeating separate CAS.'
        client = self._connected_client("offline-upload-reclaim")
        client._stop_heartbeat_worker()
        decision = self._claim_one(client, "recover upload ownership", 15)
        checkpoint_path = self._write_checkpoint("offline-upload", b"reclaim then replay")
        original_heartbeat = client.send_as_heartbeat
        heartbeat_calls = 0

        def heartbeat_then_expire_before_first_chunk():
            'Expire after the final pre-upload heartbeat, before AS writes bytes.'
            nonlocal heartbeat_calls
            session = original_heartbeat()
            heartbeat_calls += 1
            if heartbeat_calls == 2:
                with self._as.service._lock:
                    self._as.service._sessions_by_sid[session.sid].online = False
                self._as.service._release_dropped_trainer_tasks(session.sid)
            return session

        with patch.object(client, "send_as_heartbeat", side_effect=heartbeat_then_expire_before_first_chunk):
            submission = client.submit_model_update_at_as(
                checkpoint_path,
                [decision],
                round_id=15,
                sample_count=1,
            )

        self.assertGreaterEqual(heartbeat_calls, 3)
        self.assertEqual(submission.committed_task_ids, (decision.task_id,))
        self.assertEqual(self._as.index.snapshot(decision.task_id).state, TaskState.COMMITTED)

    def test_finalize_revalidates_ownership_after_pending_task_release(self) -> None:
        'Recover only after CAS confirms ownership lost before final commit.'
        client = self._connected_client("finalize-reclaim")
        decision = self._claim_one(client, "recover finalize ownership", 16)
        checkpoint_path = self._write_checkpoint("finalize-reclaim", b"revalidate before commit")
        session = client.as_session
        assert session is not None and client._as_transport is not None
        releasing_transport = _ReleasePendingTaskBeforeFinalize(
            client._as_transport,
            self._as.service,
            decision.task_id,
            session.sid,
        )
        client._as_transport = releasing_transport

        submission = client.submit_model_update_at_as(
            checkpoint_path,
            [decision],
            round_id=16,
            sample_count=1,
        )

        self.assertTrue(releasing_transport.did_release)
        self.assertEqual(submission.committed_task_ids, (decision.task_id,))
        self.assertEqual(self._as.index.snapshot(decision.task_id).state, TaskState.COMMITTED)

    def test_stale_model_upload_is_rejected_after_recovery_reassignment(self) -> None:
        'Reject a stale checkpoint after a safe owner takes over its task.'
        client = self._connected_client("stale-model-upload")
        successor = self._connected_client("stale-model-successor")
        first_registration = client.register_records_with_as(["expired model record"], created_round=12)
        successor_registration = successor.register_records_with_as(
            ["expired model record"],
            created_round=12,
        )
        decision = client.claim_registered_labels_at_as(first_registration)[0]
        successor.claim_registered_labels_at_as(successor_registration)
        checkpoint_path = self._write_checkpoint("expired", b"stale model update")
        client._stop_heartbeat_worker()
        time.sleep(0.5)
        self._as.service.expire_sessions()
        successor.send_as_heartbeat()

        with self.assertRaises(ModelUpdateOwnershipLostError):
            client.submit_model_update_at_as(
                checkpoint_path,
                [decision],
                round_id=12,
                sample_count=1,
            )

        snapshot = self._as.index.snapshot(decision.task_id)
        successor_session = successor.as_session
        assert successor_session is not None
        self.assertEqual((snapshot.state, snapshot.trainer), (TaskState.PENDING, successor_session.sid))

    def test_finalize_rejection_carries_taken_over_labels_as_dedup_instructions(self) -> None:
        'Return paper-required DEDUP labels when another live SID took over.'
        client = self._connected_client("structured-rejection-client")
        successor = self._connected_client("structured-rejection-successor")
        first_registration = client.register_records_with_as(
            ["structured rejection record"],
            created_round=19,
        )
        successor_registration = successor.register_records_with_as(
            ["structured rejection record"],
            created_round=19,
        )
        decision = client.claim_registered_labels_at_as(first_registration)[0]
        successor.claim_registered_labels_at_as(successor_registration)
        client_session = client.as_session
        assert client_session is not None and client._as_transport is not None

        # Emulate a completed timeout-release/takeover while leaving the original
        # SID online so finalization, rather than connection setup, carries the
        # paper recovery payload.
        
        self.assertTrue(self._as.index.release_if_trainer(decision.task_id, client_session.sid))
        self._as.index.set_recovery_required(decision.task_id, True)
        with self._as.service._lock:
            self._as.service._sessions_by_sid[client_session.sid].recovery_risk = True
            self._as.service._round_dispatch_enabled = True
        successor.send_as_heartbeat()
        successor_session = successor.as_session
        assert successor_session is not None
        self.assertEqual(self._as.index.snapshot(decision.task_id).trainer, successor_session.sid)

        capture = _CaptureOwnershipRejection(client._as_transport)
        client._as_transport = capture
        checkpoint_path = self._write_checkpoint("structured-rejection", b"must retrain")
        with self.assertRaises(ModelUpdateOwnershipLostError) as captured_error:
            client.submit_model_update_at_as(
                checkpoint_path,
                [decision],
                round_id=19,
                sample_count=1,
            )

        error = captured_error.exception
        self.assertEqual([(item.protected_label, item.operation) for item in error.dedup_instructions], [
            (decision.protected_label, "DEDUP")
        ])
        self.assertIsNotNone(capture.payload)
        assert capture.payload is not None
        self.assertEqual(capture.payload["dedup_instructions"], [{
            "protected_label": decision.protected_label,
            "task_id": decision.task_id,
            "operation": "DEDUP",
            "state": "PENDING",
        }])
        self.assertEqual(list((self._root / "model-updates" / "staging").glob("*.part")), [])

    def test_aggregate_route_uses_authoritative_committed_task_counts(self) -> None:
        'Verify AS derives FedAvg weights from COMMITTED task states.'
        first_client = self._connected_client("aggregate-first")
        second_client = self._connected_client("aggregate-second")
        first_decision = self._claim_one(first_client, "first model record", 9)
        second_decision = self._claim_one(second_client, "second model record", 9)
        first_checkpoint = self._write_checkpoint("first", b"first update")
        second_checkpoint = self._write_checkpoint("second", b"second update")
        first_session = first_client.as_session
        second_session = second_client.as_session
        assert first_session is not None and second_session is not None
        participant_sids = (first_session.sid, second_session.sid)
        self.assertEqual(
            first_client.configure_federated_round_at_as(9, participant_sids),
            participant_sids,
        )
        first_submission = first_client.submit_model_update_at_as(
            first_checkpoint,
            [first_decision],
            round_id=9,
            sample_count=2,
        )
        second_submission = second_client.submit_model_update_at_as(
            second_checkpoint,
            [second_decision],
            round_id=9,
            sample_count=5,
        )
        captured: dict[str, object] = {}

        def fake_aggregate(update_paths, sample_counts, output_path):
            'Capture AS-selected inputs without requiring PyTorch in this test.'
            captured["paths"] = tuple(update_paths)
            captured["sample_counts"] = tuple(sample_counts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"global checkpoint")
            return output_path

        with patch("dbtfl.entities.aggregation_server.aggregate_safetensors", fake_aggregate):
            response = first_client.aggregate_federated_round_at_as(
                9, (first_submission.sid, second_submission.sid)
            )

        # The client-provided values 2 and 5 are intentionally inconsistent
        # with the one COMMITTED task owned by each SID. AS must ignore those
        # declarations and count the state table after finalization.
        
        
        self.assertEqual(response["total_samples"], 2)
        self.assertEqual(captured["sample_counts"], (1, 1))
        self.assertEqual(len(captured["paths"]), 2)
        downloaded_path = first_client.download_global_model_from_as(
            9,
            self._root / "first-global.safetensors",
        )
        self.assertEqual(downloaded_path.read_bytes(), b"global checkpoint")
        first_client.send_as_heartbeat()
        self.assertEqual(first_client.active_round, 10)
        self.assertEqual(first_client.last_round_instructions[0].operation, "TRAIN")

    def test_incremental_replacement_update_supersedes_one_uploaded_checkpoint(self) -> None:
        'Use the newest cumulative update when a submitted client gets new TRAIN work.\n        The fixed FedAvg roster contains the same SID throughout this test.  A\n        heartbeat-delivered task is trained after the first upload, and the\n        client submits a cumulative replacement containing both task IDs.  AS\n        must commit only the newly PENDING task and aggregate the replacement\n        checkpoint, rather than rejecting the task or retaining stale bytes.\n        PENDING'
        client = self._connected_client("incremental-replacement")
        first = self._claim_one(client, "first incremental record", 25)
        session = client.as_session
        assert session is not None
        self.assertEqual(
            client.configure_federated_round_at_as(25, (session.sid,)),
            (session.sid,),
        )
        first_checkpoint = self._write_checkpoint("first-incremental", b"first update")
        client.submit_model_update_at_as(
            first_checkpoint,
            [first],
            round_id=25,
            sample_count=1,
        )

        second = self._claim_one(client, "second incremental record", 25)
        self.assertNotEqual(first.task_id, second.task_id)
        replacement_checkpoint = self._write_checkpoint(
            "replacement-incremental",
            b"replacement includes old and new local work",
        )
        replacement = client.submit_model_update_at_as(
            replacement_checkpoint,
            [first, second],
            round_id=25,
            sample_count=2,
        )
        self.assertEqual(
            replacement.committed_task_ids,
            (first.task_id, second.task_id),
        )
        self.assertEqual(self._as.index.snapshot(first.task_id).state, TaskState.COMMITTED)
        self.assertEqual(self._as.index.snapshot(second.task_id).state, TaskState.COMMITTED)

        captured: dict[str, object] = {}

        def fake_aggregate(update_paths, sample_counts, output_path):
            'Capture the replacement descriptor selected by AS.'
            captured["paths"] = tuple(update_paths)
            captured["sample_counts"] = tuple(sample_counts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"replacement global model")
            return output_path

        with patch("dbtfl.entities.aggregation_server.aggregate_safetensors", fake_aggregate):
            response = client.aggregate_federated_round_at_as(25, (session.sid,))

        self.assertEqual(response["total_samples"], 2)
        self.assertEqual(captured["sample_counts"], (2,))
        selected_paths = captured["paths"]
        self.assertEqual(len(selected_paths), 1)
        self.assertEqual(
            selected_paths[0].read_bytes(),
            replacement_checkpoint.read_bytes(),
        )

    def test_global_model_download_recovers_when_the_read_sid_expires(self) -> None:
        'Reconnect and replay the same immutable read after an AS lease expiry.\n        This is the real HTTP regression for the post-FedAvg failure path: AS\n        rejects the first download before returning bytes, the client restores\n        its stable SID, and the downloaded checkpoint still passes SHA-256.'
        first_client = self._connected_client("expired-global-reader-first")
        second_client = self._connected_client("expired-global-reader-second")
        first_decision = self._claim_one(first_client, "first reader record", 21)
        second_decision = self._claim_one(second_client, "second reader record", 21)
        first_checkpoint = self._write_checkpoint("reader-first", b"first reader update")
        second_checkpoint = self._write_checkpoint("reader-second", b"second reader update")
        first_session = first_client.as_session
        second_session = second_client.as_session
        assert first_session is not None and second_session is not None
        participant_sids = (first_session.sid, second_session.sid)
        first_client.configure_federated_round_at_as(21, participant_sids)
        first_submission = first_client.submit_model_update_at_as(
            first_checkpoint, [first_decision], round_id=21, sample_count=1
        )
        second_submission = second_client.submit_model_update_at_as(
            second_checkpoint, [second_decision], round_id=21, sample_count=1
        )

        def fake_aggregate(update_paths, sample_counts, output_path):
            'Create deterministic global bytes without requiring PyTorch.'
            del update_paths, sample_counts
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"global checkpoint recovered after sid expiry")
            return output_path

        with patch("dbtfl.entities.aggregation_server.aggregate_safetensors", fake_aggregate):
            first_client.aggregate_federated_round_at_as(
                21, (first_submission.sid, second_submission.sid)
            )

        assert first_client._as_transport is not None
        expiring_transport = _ExpireFirstGlobalModelRead(
            first_client._as_transport,
            self._as.service,
            first_session.sid,
        )
        first_client._as_transport = expiring_transport
        downloaded_path = first_client.download_global_model_from_as(
            21,
            self._root / "recovered-global.safetensors",
        )

        self.assertTrue(expiring_transport.did_expire)
        self.assertGreaterEqual(expiring_transport.global_read_count, 2)
        self.assertIn("", expiring_transport.last_offline_detail)
        self.assertEqual(
            downloaded_path.read_bytes(),
            b"global checkpoint recovered after sid expiry",
        )

    def test_dropped_trainer_is_released_and_other_owner_receives_takeover(self) -> None:
        'Verify heartbeat timeout safely transfers a shared PENDING task.'
        first_client = self._connected_client("recovery-first")
        second_client = self._connected_client("recovery-second")
        first_registration = first_client.register_records_with_as(["recover me"], created_round=10)
        second_registration = second_client.register_records_with_as(
            ["recover me"],
            created_round=10,
        )
        first_decision = first_client.claim_registered_labels_at_as(first_registration)[0]
        second_decision = second_client.claim_registered_labels_at_as(second_registration)[0]
        self.assertEqual((first_decision.operation, second_decision.operation), ("TRAIN", "DEDUP"))

        first_client._stop_heartbeat_worker()
        time.sleep(0.5)
        self._as.service.expire_sessions()
        second_client.send_as_heartbeat()

        takeover = second_client.last_round_instructions
        self.assertEqual(
            [(item.task_id, item.operation) for item in takeover],
            [(second_decision.task_id, "TRAIN")],
        )
        snapshot = self._as.index.snapshot(second_decision.task_id)
        second_session = second_client.as_session
        assert second_session is not None
        self.assertEqual(
            (snapshot.state, snapshot.trainer),
            (TaskState.PENDING, second_session.sid),
        )

    def test_round_roster_can_expand_after_sequential_registration(self) -> None:
        'Verify a dynamically admitted SID can enter a future fixed FedAvg roster.'
        first_client = self._connected_client("join-first")
        second_client = self._connected_client("join-second")
        first_session = first_client.as_session
        second_session = second_client.as_session
        assert first_session is not None and second_session is not None
        transport = JsonHttpClient(self._as.base_url)
        transport.send(
            AggregationServerPath.CONFIGURE_ROUND.value,
            WireMessage.create(
                AS_CONFIGURE_ROUND_REQUEST,
                {"round": 11, "participant_sids": [first_session.sid]},
            ),
        )
        response = transport.send(
            AggregationServerPath.CONFIGURE_ROUND.value,
            WireMessage.create(
                AS_CONFIGURE_ROUND_REQUEST,
                {"round": 11, "participant_sids": [first_session.sid, second_session.sid]},
            ),
        )

        self.assertEqual(
            response.payload["participant_sids"],
            [first_session.sid, second_session.sid],
        )

    def test_frozen_roster_dispatches_released_task_for_incremental_replacement(self) -> None:
        'Dispatch released work before FedAvg so the participant can replace its update.\n        A frozen roster fixes only participating SIDs. A later heartbeat may\n        move released work to an existing participant; it must incrementally\n        train that work and replace its checkpoint before aggregation.'
        first_client = self._connected_client("sealed-round-first")
        second_client = self._connected_client("sealed-round-second")
        first_decision = self._claim_one(first_client, "sealed shared record", 22)
        second_registration = second_client.register_records_with_as(
            ["sealed shared record"], created_round=22
        )
        second_client.claim_registered_labels_at_as(second_registration)
        first_session = first_client.as_session
        assert first_session is not None
        with self._as.service._lock:
            self._as.service._round_dispatch_enabled = True
        first_client.configure_federated_round_at_as(22, (first_session.sid,))
        self.assertTrue(
            self._as.index.release_if_trainer(first_decision.task_id, first_session.sid)
        )
        self._as.index.set_recovery_required(first_decision.task_id, True)

        first_client.send_as_heartbeat()

        self.assertEqual(
            [(item.task_id, item.operation) for item in first_client.last_round_instructions],
            [(first_decision.task_id, "TRAIN")],
        )
        self.assertEqual(self._as.index.snapshot(first_decision.task_id).state, TaskState.PENDING)

    def _connected_client(self, client_id: str) -> ClientEntity:
        'Create one temporary client connected to the test AS and KS.'
        client = ClientEntity(
            ClientConfig(
                client_id=client_id,
                ks_base_url=self._ks.base_url,
                as_base_url=self._as.base_url,
                label_store_path=self._root / f"{client_id}.json",
                model_chunk_bytes=8,
            )
        )
        client.connect_to_as()
        self.addCleanup(client.close)
        return client

    def _claim_one(self, client: ClientEntity, record: str, round_id: int):
        'Register then claim one locally retained record in paper order.'
        registration = client.register_records_with_as([record], created_round=round_id)
        return client.claim_registered_labels_at_as(registration)[0]

    def _write_checkpoint(self, stem: str, contents: bytes) -> Path:
        'Write one temporary byte checkpoint for transport-route coverage.'
        checkpoint_path = self._root / f"{stem}.safetensors"
        checkpoint_path.write_bytes(contents)
        return checkpoint_path


class _LoseFirstModelChunkResponse:
    'Delegate real HTTP requests but discard one post-write chunk response.'

    def __init__(self, transport: JsonHttpClient) -> None:
        'Wrap one transport for a single controlled response-loss event.'
        self._transport = transport
        self.did_drop_response = False

    def send(self, path: str, message: WireMessage, *, method: str = "POST") -> WireMessage:
        'Persist the first chunk, then simulate a lost response to the client.'
        response = self._transport.send(path, message, method=method)
        if (
            not self.did_drop_response
            and path == AggregationServerPath.SUBMIT_MODEL_UPDATE.value
            and message.message_type == AS_MODEL_CHUNK_REQUEST
        ):
            self.did_drop_response = True
            raise TransportError("simulated response loss / ")
        return response


class _RecordingTransport:
    'Record real HTTP operation order while delegating every request.'

    def __init__(self, transport: JsonHttpClient) -> None:
        'Wrap one working transport and initialize an ordered event log.'
        self._transport = transport
        self.events: list[str] = []

    def send(self, path: str, message: WireMessage, *, method: str = "POST") -> WireMessage:
        'Record only heartbeat and model-chunk request boundaries.'
        if path == AggregationServerPath.HEARTBEAT.value:
            self.events.append("heartbeat")
        elif message.message_type == AS_MODEL_CHUNK_REQUEST:
            self.events.append("model_chunk")
        return self._transport.send(path, message, method=method)


class _ExpireFirstGlobalModelRead:
    'Force one AS global-model read to observe an expired client lease.'

    def __init__(self, transport: JsonHttpClient, service, sid: int) -> None:
        'Retain the real transport and the specific registered reader SID.'
        self._transport = transport
        self._service = service
        self._sid = sid
        self.did_expire = False
        self.global_read_count = 0
        self.last_offline_detail = ""

    def send(self, path: str, message: WireMessage, *, method: str = "POST") -> WireMessage:
        'Expire exactly before the first real immutable-read request.'
        if path == AggregationServerPath.DOWNLOAD_GLOBAL_MODEL.value:
            self.global_read_count += 1
            if not self.did_expire:
                with self._service._lock:
                    self._service._sessions_by_sid[self._sid].online = False
                self.did_expire = True
        try:
            return self._transport.send(path, message, method=method)
        except RemoteServiceError as error:
            if path == AggregationServerPath.DOWNLOAD_GLOBAL_MODEL.value and error.code == "sid_offline":
                self.last_offline_detail = error.detail
            raise


class _ReleasePendingTaskBeforeFinalize:
    'Release one task immediately before the first real finalization request.'

    def __init__(self, transport: JsonHttpClient, service, task_id: int, sid: int) -> None:
        'Keep the real transport and the exact pending-task owner.'
        self._transport = transport
        self._service = service
        self._task_id = task_id
        self._sid = sid
        self.did_release = False

    def send(self, path: str, message: WireMessage, *, method: str = "POST") -> WireMessage:
        'Create the finalization-time ownership race once, then delegate.'
        if (
            not self.did_release
            and path == AggregationServerPath.SUBMIT_MODEL_UPDATE.value
            and message.message_type == "as.model.finalize.request"
        ):
            self.did_release = self._service.index.release_if_trainer(self._task_id, self._sid)
            self._service.index.set_recovery_required(self._task_id, True)
            with self._service._lock:
                self._service._round_dispatch_enabled = True
        return self._transport.send(path, message, method=method)


class _CaptureOwnershipRejection:
    'Capture the real structured 409 payload while preserving client behavior.'

    def __init__(self, transport: JsonHttpClient) -> None:
        'Wrap a real transport.'
        self._transport = transport
        self.payload: dict[str, object] | None = None

    def send(self, path: str, message: WireMessage, *, method: str = "POST") -> WireMessage:
        'Save only the ownership-loss error payload, then re-raise it.'
        try:
            return self._transport.send(path, message, method=method)
        except RemoteServiceError as error:
            if error.code == "task_not_pending_for_sid":
                self.payload = dict(error.payload)
            raise


if __name__ == "__main__":
    unittest.main()
