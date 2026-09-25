'Cross-platform integration tests for the separate AS CAS claim phase.\nAS'

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.communication import RemoteServiceError
from dbtfl.entities import (
    AggregationServerConfig,
    AggregationServerEntity,
    ClientConfig,
    ClientEntity,
    KeyServerConfig,
    KeyServerEntity,
)
from dbtfl.native_index import TaskState


class TaskClaimIntegrationTest(unittest.TestCase):
    'Exercise paper-ordered registration followed by CAS task claiming.'

    @classmethod
    def setUpClass(cls) -> None:
        'Build a native library dedicated to task-claim integration tests.'
        if shutil.which("g++") is None:
            raise unittest.SkipTest("g++ is required for claim tests /  g++")
        suffix = ".dll" if sys.platform == "win32" else ".so"
        cls.library_path = (
            PROJECT_ROOT / "results" / "native-test-artifacts" / f"atomic_word_claim_test{suffix}"
        )
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

    def setUp(self) -> None:
        'Start isolated loopback KS and AS entities for every test.'
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._root = Path(self._temporary_directory.name)
        key_path = self._root / "ks-key.json"
        self._ks = KeyServerEntity(KeyServerConfig(key_path=key_path, host="127.0.0.1", port=0))
        self._ks.start()
        self.addCleanup(self._ks.close)
        self._as = AggregationServerEntity(
            AggregationServerConfig(
                capacity=64,
                max_clients=8,
                max_edges=128,
                host="127.0.0.1",
                port=0,
                native_library_path=self.library_path,
            )
        )
        self._as.start()
        self.addCleanup(self._as.close)

    def _client(self, client_id: str) -> ClientEntity:
        'Create, connect, and clean up one test client.'
        client = ClientEntity(
            ClientConfig(
                client_id=client_id,
                ks_base_url=self._ks.base_url,
                as_base_url=self._as.base_url,
                label_store_path=self._root / f"{client_id}-labels.json",
            )
        )
        client.connect_to_as()
        self.addCleanup(client.close)
        return client

    def test_claim_after_registration_returns_paper_train_dedup_mapping(self) -> None:
        'Verify registered shared data receives TRAIN once and DEDUP thereafter.'
        first_client = self._client("claim-first")
        second_client = self._client("claim-second")
        first_registrations = first_client.register_records_with_as(
            ["shared", "first-only"],
            created_round=5,
        )
        second_registrations = second_client.register_records_with_as(
            ["shared", "second-only"],
            created_round=5,
        )

        first_decisions = first_client.claim_registered_labels_at_as(first_registrations)
        second_decisions = second_client.claim_registered_labels_at_as(second_registrations)

        self.assertEqual([decision.operation for decision in first_decisions], ["TRAIN", "TRAIN"])
        self.assertEqual(second_decisions[0].operation, "DEDUP")
        self.assertEqual(second_decisions[0].state, "PENDING")
        self.assertEqual(second_decisions[1].operation, "TRAIN")
        shared_task_id = first_decisions[0].task_id
        first_session = first_client.as_session
        assert first_session is not None
        snapshot = self._as.index.snapshot(shared_task_id)
        self.assertEqual((snapshot.state, snapshot.trainer), (TaskState.PENDING, first_session.sid))

    def test_concurrent_claims_have_one_train_winner_after_prior_registration(self) -> None:
        'Verify the CAS phase has exactly one TRAIN winner for a shared task.'
        first_client = self._client("claim-concurrent-first")
        second_client = self._client("claim-concurrent-second")
        first_registration = first_client.register_records_with_as(
            ["same claim record"],
            created_round=6,
        )
        second_registration = second_client.register_records_with_as(
            ["same claim record"],
            created_round=6,
        )
        start = threading.Barrier(2)

        def claim(client: ClientEntity, registrations):
            'Claim only after both client workers reach one barrier.'
            start.wait()
            return client.claim_registered_labels_at_as(registrations)[0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(claim, first_client, first_registration)
            second_future = executor.submit(claim, second_client, second_registration)
            decisions = [first_future.result(), second_future.result()]

        self.assertEqual(
            sorted(decision.operation for decision in decisions),
            ["DEDUP", "TRAIN"],
        )
        self.assertEqual({decision.state for decision in decisions}, {"PENDING"})
        self.assertEqual(decisions[0].task_id, decisions[1].task_id)

    def test_non_owner_cannot_claim_registered_label(self) -> None:
        'Verify claim processing preserves the label-owner boundary.'
        owner_client = self._client("claim-owner")
        other_client = self._client("claim-other")
        registration = owner_client.register_records_with_as(["owner-only"], created_round=7)[0]

        with self.assertRaises(RemoteServiceError) as captured_error:
            other_client.claim_protected_labels_at_as([registration.protected_label])

        self.assertEqual(captured_error.exception.status_code, 403)
        self.assertEqual(captured_error.exception.code, "sid_not_label_owner")

    def test_claim_reconnects_once_when_as_reports_an_offline_sid(self) -> None:
        'Recover the separate CAS phase after AS marks the SID offline.'
        client = self._client("claim-offline-reconnect")
        registration = client.register_records_with_as(["recoverable CAS record"], created_round=8)
        session = client.as_session
        assert session is not None

        # The AS rejects before CAS mutation, so the retry must produce exactly
        # one normal decision. AS
        with self._as.service._lock:
            self._as.service._sessions_by_sid[session.sid].online = False

        decision = client.claim_registered_labels_at_as(registration)[0]

        self.assertEqual(decision.operation, "TRAIN")
        self.assertEqual(decision.state, "PENDING")
        recovered_session = client.as_session
        assert recovered_session is not None
        self.assertEqual(recovered_session.sid, session.sid)
        self.assertTrue(self._as.session_snapshot(recovered_session.sid).online)

    def test_claim_reconnects_through_repeated_offline_queue_states(self) -> None:
        'Retry through repeated offline states within the RPC time budget.'
        client = self._client("claim-repeated-offline")
        registration = client.register_records_with_as(["queued CAS record"], created_round=9)
        original_heartbeat = client.send_as_heartbeat
        original_connect = client.connect_to_as
        reconnect_count = 0

        def heartbeat_then_expire() -> object:
            'Refresh once, then model the queue-induced server offline transition.'
            refreshed = original_heartbeat()
            with self._as.service._lock:
                self._as.service._sessions_by_sid[refreshed.sid].online = False
            return refreshed

        def reconnect_with_repeated_expiry() -> object:
            'Force four recovered SIDs to expire before a later retry succeeds.'
            nonlocal reconnect_count
            refreshed = original_connect()
            reconnect_count += 1
            if reconnect_count <= 4:
                with self._as.service._lock:
                    self._as.service._sessions_by_sid[refreshed.sid].online = False
            return refreshed

        with patch.object(client, "send_as_heartbeat", side_effect=heartbeat_then_expire):
            with patch.object(client, "connect_to_as", side_effect=reconnect_with_repeated_expiry):
                decision = client.claim_registered_labels_at_as(registration)[0]

        self.assertEqual(reconnect_count, 5)
        self.assertEqual(decision.operation, "TRAIN")


if __name__ == "__main__":
    unittest.main()
