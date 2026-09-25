'Cross-platform integration tests for client protected-label AS submission.'

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.entities import (
    AggregationServerConfig,
    AggregationServerEntity,
    ClientConfig,
    ClientEntity,
    KeyServerConfig,
    KeyServerEntity,
)
from dbtfl.native_index import TaskState


class ProtectedLabelSubmissionIntegrationTest(unittest.TestCase):
    "Exercise the paper's client--KS--AS label and CAS flow."

    @classmethod
    def setUpClass(cls) -> None:
        'Build an AS-specific native library isolated from other tests.'
        if shutil.which("g++") is None:
            raise unittest.SkipTest("g++ is required for label tests /  g++")
        suffix = ".dll" if sys.platform == "win32" else ".so"
        cls.library_path = (
            PROJECT_ROOT / "results" / "native-test-artifacts" / f"atomic_word_label_test{suffix}"
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
        'Start a shared KS and AS with independent ephemeral HTTP ports.'
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
                heartbeat_interval_seconds=0.03,
                heartbeat_timeout_seconds=0.12,
                native_library_path=self.library_path,
            )
        )
        self._as.start()
        self.addCleanup(self._as.close)

    def _client(self, client_id: str) -> ClientEntity:
        'Create one fully connected client for this test deployment.'
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

    def test_submission_updates_both_indexes_and_keeps_empty_state(self) -> None:
        'Verify shared labels acquire both owners without a trainer claim.'
        first_client = self._client("client-first")
        second_client = self._client("client-second")
        first_registrations = first_client.register_records_with_as(
            ["shared", "first-only"],
            created_round=3,
        )
        second_registrations = second_client.register_records_with_as(
            ["shared", "second-only"],
            created_round=3,
        )

        shared_task_id = first_registrations[0].task_id
        self.assertEqual(shared_task_id, second_registrations[0].task_id)
        first_sid = first_client.as_session
        second_sid = second_client.as_session
        assert first_sid is not None and second_sid is not None
        self.assertEqual(
            self._as.index.owners(shared_task_id),
            (first_sid.sid, second_sid.sid),
        )
        self.assertIn(shared_task_id, self._as.index.client_tasks(first_sid.sid))
        self.assertIn(shared_task_id, self._as.index.client_tasks(second_sid.sid))
        snapshot = self._as.index.snapshot(shared_task_id)
        self.assertEqual(snapshot.state, TaskState.EMPTY)
        self.assertEqual(snapshot.trainer, 0)

    def test_concurrent_shared_submission_updates_both_indexes_without_claiming(self) -> None:
        'Verify concurrent registrations preserve EMPTY for a later CAS phase.'
        first_client = self._client("client-concurrent-first")
        second_client = self._client("client-concurrent-second")
        protected_label = first_client.generate_protected_labels(
            ["same concurrent record"]
        )[0]
        self.assertEqual(
            second_client.generate_protected_labels(["same concurrent record"])[0],
            protected_label,
        )
        start = threading.Barrier(2)

        def submit(client: ClientEntity):
            'Submit after both client threads reach the same start barrier.'
            start.wait()
            return client.register_protected_labels_with_as(
                [protected_label],
                created_round=4,
            )[0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            registrations = list(executor.map(submit, (first_client, second_client)))

        self.assertEqual(registrations[0].task_id, registrations[1].task_id)
        shared_task_id = registrations[0].task_id
        first_sid = first_client.as_session
        second_sid = second_client.as_session
        assert first_sid is not None and second_sid is not None
        self.assertEqual(
            self._as.index.owners(shared_task_id),
            (first_sid.sid, second_sid.sid),
        )
        self.assertEqual(self._as.index.snapshot(shared_task_id).state, TaskState.EMPTY)

    def test_registration_recovers_sid_after_a_long_local_oprf_phase(self) -> None:
        'Refresh an expired SID before AS registration of cached OPRF labels.'
        client = self._client("client-synchronous-refresh")
        client.generate_protected_labels(["CPU-bound local OPRF record"])
        client._stop_heartbeat_worker()
        time.sleep(0.2)

        registrations = client.register_records_with_as(
            ["CPU-bound local OPRF record"],
            created_round=5,
        )

        self.assertEqual(len(registrations), 1)
        session = client.as_session
        assert session is not None
        self.assertTrue(self._as.session_snapshot(session.sid).online)

    def test_registration_reconnects_once_when_as_reports_an_offline_sid(self) -> None:
        'Reconnect and retry when AS observes an offline SID after heartbeat.'
        client = self._client("client-offline-reconnect")
        protected_label = client.generate_protected_labels(["offline registration record"])[0]
        session = client.as_session
        assert session is not None

        # Model the server-side state transition that can race with a client-side
        # heartbeat.
        with self._as.service._lock:
            self._as.service._sessions_by_sid[session.sid].online = False

        registrations = client.register_protected_labels_with_as(
            [protected_label],
            created_round=6,
        )

        self.assertEqual(len(registrations), 1)
        recovered_session = client.as_session
        assert recovered_session is not None
        self.assertEqual(recovered_session.sid, session.sid)
        self.assertTrue(self._as.session_snapshot(recovered_session.sid).online)


if __name__ == "__main__":
    unittest.main()
