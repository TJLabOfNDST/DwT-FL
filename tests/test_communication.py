'Cross-platform integration tests for the JSON communication package. / JSON'

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.communication import (
    JsonHttpClient,
    JsonRouter,
    ProtocolError,
    RemoteServiceError,
    RequestRejected,
    ThreadedJsonServer,
    TrafficRecorder,
    WireMessage,
)


class CommunicationIntegrationTest(unittest.TestCase):
    'Exercise a real loopback server and client without external dependencies.'

    def _started_server(self, router: JsonRouter) -> ThreadedJsonServer:
        'Create a loopback server on an OS-selected port.'
        server = ThreadedJsonServer("127.0.0.1", 0, router)
        server.start()
        self.addCleanup(server.close)
        return server

    def test_round_trip_preserves_request_id_and_payload(self) -> None:
        'Verify client, HTTP server, router, and envelope compatibility.'
        router = JsonRouter()

        def echo(message: WireMessage) -> WireMessage:
            'Return the submitted payload for this protocol test.'
            return WireMessage.create(
                "test.echo.response",
                {"received": dict(message.payload)},
                request_id=message.request_id,
            )

        router.add("POST", "/v1/test/echo", echo)
        server = self._started_server(router)
        client = JsonHttpClient(server.base_url)
        request = WireMessage.create(
            "test.echo.request",
            {"sample": "local-text", "count": 2},
        )
        response = client.send("/v1/test/echo", request)

        self.assertEqual(response.request_id, request.request_id)
        self.assertEqual(response.message_type, "test.echo.response")
        self.assertEqual(response.payload, {"received": {"sample": "local-text", "count": 2}})

    def test_route_rejection_becomes_a_structured_client_error(self) -> None:
        'Verify a handler rejection remains observable at the client.'
        router = JsonRouter()

        def reject(message: WireMessage) -> WireMessage:
            'Reject the test operation with an explicit conflict.'
            raise RequestRejected(
                409,
                "test_conflict",
                "test route rejects this request / ",
                {"dedup_instructions": [{"protected_label": "opaque-label", "operation": "DEDUP"}]},
            )

        router.add("POST", "/v1/test/reject", reject)
        server = self._started_server(router)
        client = JsonHttpClient(server.base_url)
        request = WireMessage.create("test.reject.request", {})

        with self.assertRaises(RemoteServiceError) as captured_error:
            client.send("/v1/test/reject", request)

        error = captured_error.exception
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.code, "test_conflict")
        self.assertEqual(error.request_id, request.request_id)
        self.assertEqual(error.payload["dedup_instructions"], [
            {"protected_label": "opaque-label", "operation": "DEDUP"}
        ])

    def test_traffic_recorder_reports_body_and_payload_bytes_without_bodies(self) -> None:
        'Record one real exchange without retaining protocol body contents.'
        router = JsonRouter()
        router.add(
            "POST", "/v1/test/traffic",
            lambda message: WireMessage.create(
                "test.traffic.response", {"accepted": True}, request_id=message.request_id
            ),
        )
        server = self._started_server(router)
        recorder = TrafficRecorder()
        request = WireMessage.create("test.traffic.request", {"labels": ["opaque"]})
        JsonHttpClient(server.base_url, traffic_recorder=recorder).send(
            "/v1/test/traffic", request
        )

        records = recorder.snapshot()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.path, "/v1/test/traffic")
        self.assertEqual(record.request_message_type, "test.traffic.request")
        self.assertEqual(record.response_message_type, "test.traffic.response")
        self.assertGreater(record.request_body_bytes, record.request_payload_bytes)
        self.assertGreater(record.response_body_bytes, record.response_payload_bytes)
        self.assertIsNone(record.error_type)

    def test_raised_wire_exceptions_never_fail_during_traceback_attachment(self) -> None:
        'Raise both wire exceptions, including Python 3.10 traceback handling.'
        rejected = RequestRejected(409, "ownership_lost", "ownership changed / ")
        remote = RemoteServiceError(409, "ownership_lost", "ownership changed / ", "trace-id")

        with self.assertRaises(RequestRejected) as rejected_error:
            raise rejected
        with self.assertRaises(RemoteServiceError) as remote_error:
            raise remote

        self.assertEqual(rejected_error.exception.code, "ownership_lost")
        self.assertEqual(remote_error.exception.request_id, "trace-id")

    def test_protocol_rejects_unknown_fields_and_invalid_message_types(self) -> None:
        'Verify schema strictness before messages reach a service.'
        with self.assertRaises(ProtocolError):
            WireMessage.create("Invalid Type", {})
        with self.assertRaises(ProtocolError):
            WireMessage.from_json_bytes(
                b'{"schema_version":"1.0","message_type":"test.request",'
                b'"request_id":"abc","payload":{},"unknown":true}'
            )

    def test_control_path_bypasses_saturated_data_request_slots(self) -> None:
        'Keep a heartbeat-like control RPC available behind a busy data plane.'
        router = JsonRouter()
        data_started = threading.Event()
        release_data = threading.Event()

        def blocked_data(message: WireMessage) -> WireMessage:
            'Hold the only data slot until the test releases it.'
            data_started.set()
            release_data.wait(timeout=2.0)
            return WireMessage.create("test.data.response", {}, request_id=message.request_id)

        def control(message: WireMessage) -> WireMessage:
            'Return a minimal control acknowledgement.'
            return WireMessage.create("test.control.response", {"alive": True}, request_id=message.request_id)

        router.add("POST", "/v1/test/data", blocked_data)
        router.add("POST", "/v1/test/control", control)
        server = ThreadedJsonServer(
            "127.0.0.1",
            0,
            router,
            max_concurrent_requests=1,
            control_paths=("/v1/test/control",),
        )
        server.start()
        self.addCleanup(server.close)
        data_client = JsonHttpClient(server.base_url, timeout_seconds=2.0)
        data_thread = threading.Thread(
            target=lambda: data_client.send("/v1/test/data", WireMessage.create("test.data.request", {})),
            daemon=True,
        )
        data_thread.start()
        self.assertTrue(data_started.wait(timeout=1.0))

        started = time.perf_counter()
        response = JsonHttpClient(server.base_url, timeout_seconds=0.5).send(
            "/v1/test/control",
            WireMessage.create("test.control.request", {}),
        )
        elapsed = time.perf_counter() - started
        release_data.set()
        data_thread.join(timeout=1.0)

        self.assertEqual(response.payload, {"alive": True})
        self.assertLess(elapsed, 0.5)

    def test_ten_parallel_control_requests_complete_without_data_slot_contention(self) -> None:
        'Serve one concurrent control request for each live experimental client.'
        router = JsonRouter()
        request_count = 10
        entered = threading.Event()
        lock = threading.Lock()
        state = {"count": 0}

        def control(message: WireMessage) -> WireMessage:
            'Wait until the concurrent heartbeat cohort has reached AS.'
            with lock:
                state["count"] += 1
                if state["count"] == request_count:
                    entered.set()
            if not entered.wait(timeout=1.0):
                raise RuntimeError("control requests were serialized")
            return WireMessage.create(
                "test.control.parallel.response",
                {"alive": True},
                request_id=message.request_id,
            )

        router.add("POST", "/v1/test/control-parallel", control)
        server = ThreadedJsonServer(
            "127.0.0.1",
            0,
            router,
            max_concurrent_requests=1,
            control_paths=("/v1/test/control-parallel",),
        )
        server.start()
        self.addCleanup(server.close)

        def invoke(position: int) -> dict[str, object]:
            'Issue one independent client control request.'
            response = JsonHttpClient(server.base_url, timeout_seconds=2.0).send(
                "/v1/test/control-parallel",
                WireMessage.create("test.control.parallel.request", {"position": position}),
            )
            return dict(response.payload)

        with ThreadPoolExecutor(max_workers=request_count) as executor:
            responses = list(executor.map(invoke, range(request_count)))

        self.assertEqual(state["count"], request_count)
        self.assertEqual(responses, [{"alive": True}] * request_count)


if __name__ == "__main__":
    unittest.main()
