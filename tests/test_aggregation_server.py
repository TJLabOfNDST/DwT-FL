'Cross-platform integration tests for AS SID and heartbeat behavior.\nAS SID'

from __future__ import annotations

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

from dbtfl.communication import JsonHttpClient, RemoteServiceError, TransportError, WireMessage
from dbtfl.communication.endpoints import AggregationServerPath
from dbtfl.entities import (
    AggregationServerConfig,
    AggregationServerEntity,
    ClientConfig,
    ClientEntity,
)
from dbtfl.entities.aggregation_server import (
    AS_EVALUATION_LEASE_REQUEST,
    AS_EVALUATION_LEASE_RESPONSE,
    AS_EVALUATION_RESET_REQUEST,
    AS_EVALUATION_RESET_RESPONSE,
    AS_HEARTBEAT_REQUEST,
)


class AggregationServerIntegrationTest(unittest.TestCase):
    'Exercise AS registration and heartbeat routes over loopback HTTP.'

    @classmethod
    def setUpClass(cls) -> None:
        'Build the exact native index required by the AS entity.'
        if shutil.which("g++") is None:
            raise unittest.SkipTest("g++ is required for AS tests / AS  g++")
        suffix = ".dll" if sys.platform == "win32" else ".so"
        cls.library_path = (
            PROJECT_ROOT / "results" / "native-test-artifacts" / f"atomic_word_as_test{suffix}"
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
        'Start a small ephemeral AS and create isolated client files.'
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._root = Path(self._temporary_directory.name)
        self._as = AggregationServerEntity(
            AggregationServerConfig(
                capacity=32,
                max_clients=8,
                max_edges=64,
                host="127.0.0.1",
                port=0,
                heartbeat_interval_seconds=0.03,
                heartbeat_timeout_seconds=0.12,
                native_library_path=self.library_path,
                model_update_directory=self._root / "updates",
                evaluation_reset_token="test-reset-token",
            )
        )
        self._as.start()
        self.addCleanup(self._as.close)

    def _client(self, client_id: str) -> ClientEntity:
        'Create one client configured for this loopback AS.'
        client = ClientEntity(
            ClientConfig(
                client_id=client_id,
                ks_base_url="http://127.0.0.1:1",
                as_base_url=self._as.base_url,
                label_store_path=self._root / f"{client_id}-labels.json",
            )
        )
        self.addCleanup(client.close)
        return client

    def test_registration_issues_stable_unique_sids(self) -> None:
        'Verify each client obtains one unique SID and reconnects reuse it.'
        first_client = self._client("client-a")
        second_client = self._client("client-b")

        first_session = first_client.connect_to_as()
        second_session = second_client.connect_to_as()
        reconnected_session = first_client.connect_to_as()

        self.assertEqual(first_session.sid, reconnected_session.sid)
        self.assertNotEqual(first_session.sid, second_session.sid)
        self.assertEqual((first_session.sid, second_session.sid), (1, 2))

    def test_registration_retries_one_transient_transport_failure(self) -> None:
        'Retry a pre-SID idempotent registration without creating another SID.'
        client = self._client("client-register-retry")
        assert client._as_registration_transport is not None
        original_send = client._as_registration_transport.send
        attempts = 0

        def send_after_one_lost_connection(*args: object, **kwargs: object) -> WireMessage:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TransportError("intentional transient registration failure")
            return original_send(*args, **kwargs)

        with patch.object(
            client._as_registration_transport,
            "send",
            side_effect=send_after_one_lost_connection,
        ):
            session = client.connect_to_as()

        self.assertEqual(attempts, 2)
        self.assertEqual(session.sid, 1)
        self.assertEqual(self._as.session_snapshot(session.sid).client_id, "client-register-retry")

    def test_client_background_heartbeat_keeps_sid_online(self) -> None:
        'Verify the connected client periodically refreshes AS liveness.'
        client = self._client("client-heartbeat")
        session = client.connect_to_as()
        deadline = time.monotonic() + 1.0
        snapshot = self._as.session_snapshot(session.sid)
        while snapshot is not None and snapshot.heartbeat_count < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = self._as.session_snapshot(session.sid)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertTrue(snapshot.online)
        self.assertGreaterEqual(snapshot.heartbeat_count, 2)
        self.assertIsNone(client.last_heartbeat_error)

    def test_as_marks_stopped_client_offline_after_timeout(self) -> None:
        'Verify the independent AS monitor detects a stopped heartbeat worker.'
        client = self._client("client-timeout")
        session = client.connect_to_as()
        client.close()
        deadline = time.monotonic() + 1.0
        snapshot = self._as.session_snapshot(session.sid)
        while snapshot is not None and snapshot.online and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = self._as.session_snapshot(session.sid)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertFalse(snapshot.online)

    def test_unknown_sid_heartbeat_is_rejected(self) -> None:
        'Verify AS does not accept a heartbeat before SID registration.'
        transport = JsonHttpClient(self._as.base_url)
        request = WireMessage.create(AS_HEARTBEAT_REQUEST, {"sid": 999})

        with self.assertRaises(RemoteServiceError) as captured_error:
            transport.send(AggregationServerPath.HEARTBEAT.value, request)

        self.assertEqual(captured_error.exception.status_code, 404)
        self.assertEqual(captured_error.exception.code, "unknown_sid")

    def test_authorized_evaluation_reset_restarts_sid_and_index_state(self) -> None:
        'Ensure every remote-evaluation case starts from an empty AS state.'
        first_client = self._client("client-before-reset")
        first_client.connect_to_as()
        first_client.close()
        transport = JsonHttpClient(self._as.base_url)

        response = transport.send(
            AggregationServerPath.EVALUATION_RESET.value,
            WireMessage.create(
                AS_EVALUATION_RESET_REQUEST,
                {
                    "token": "test-reset-token",
                    "backend_worker_count": 2,
                    "heartbeat_interval_seconds": 0.04,
                    "heartbeat_timeout_seconds": 0.20,
                },
            ),
        )
        second_client = self._client("client-after-reset")
        session = second_client.connect_to_as()

        self.assertEqual(response.message_type, AS_EVALUATION_RESET_RESPONSE)
        self.assertEqual(response.payload, {
            "reset": True,
            "backend_worker_count": 2,
            "heartbeat_interval_seconds": 0.04,
            "heartbeat_timeout_seconds": 0.20,
        })
        self.assertEqual(session.sid, 1)
        self.assertEqual(self._as.service.heartbeat_interval_seconds, 0.04)
        self.assertEqual(self._as.service.heartbeat_timeout_seconds, 0.20)

    def test_authorized_evaluation_lease_switch_preserves_registered_sid(self) -> None:
        'Switch a fault lease without erasing the active experiment state.'
        client = self._client("client-before-lease-switch")
        session = client.connect_to_as()
        # Stop periodic refresh and let the original timer age beyond the new
        # short lease while remaining below the original lease. The atomic
        # control request itself must establish the new lease origin.
        
        
        client._stop_heartbeat_worker()
        time.sleep(0.09)
        response = JsonHttpClient(self._as.base_url).send(
            AggregationServerPath.EVALUATION_LEASE.value,
            WireMessage.create(
                AS_EVALUATION_LEASE_REQUEST,
                # Leave enough wall-clock margin for an HTTP round trip on a
                # loaded cross-platform CI host; the assertion verifies the
                # atomic lease-origin reset rather than a five-millisecond
                # scheduling race.
                
                {"token": "test-reset-token", "heartbeat_timeout_seconds": 0.10},
            ),
        )

        snapshot = self._as.session_snapshot(session.sid)
        self.assertEqual(response.message_type, AS_EVALUATION_LEASE_RESPONSE)
        self.assertEqual(response.payload, {
            "heartbeat_interval_seconds": 0.03,
            "heartbeat_timeout_seconds": 0.10,
        })
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertTrue(snapshot.online)
        self.assertEqual(snapshot.client_id, "client-before-lease-switch")
        self.assertEqual(self._as.service.heartbeat_timeout_seconds, 0.10)


if __name__ == "__main__":
    unittest.main()
