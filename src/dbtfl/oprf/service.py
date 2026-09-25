'Native Ristretto255 KS OPRF endpoint for the DwT-FL protocol.\nDwT-FL'

from __future__ import annotations

import ssl
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from dbtfl.communication import JsonRouter, RequestRejected, ThreadedJsonServer, WireMessage
from dbtfl.communication.endpoints import KeyServerPath

from .group14 import OprfValidationError
from .keystore import OprfKeyMaterial, OprfKeyStore
from .ristretto255 import decode_wire_point, encode_wire_point, scalar_multiply


OPRF_EVALUATE_REQUEST: Final[str] = "oprf.evaluate.request"
OPRF_EVALUATE_RESPONSE: Final[str] = "oprf.evaluate.response"
MAX_BATCH_ELEMENTS: Final[int] = 4_096
"""Hard per-request point bound to control JSON and server memory use. /  JSON """


class KeyServerOprfService:
    'Evaluate blinded Ristretto points with the KS secret scalar.\n    All private-scalar multiplication is delegated to the required libsodium\n    backend.'

    def __init__(self, material: OprfKeyMaterial) -> None:
        'Keep a validated KS scalar only in the KS service process.'
        self._scalar = material.scalar
        self._metrics_lock = threading.Lock()
        self._evaluated_element_count = 0
        self._evaluation_compute_seconds = 0.0
        self.router = JsonRouter()
        self.router.add("POST", KeyServerPath.EVALUATE_OPRF.value, self.evaluate)

    def evaluate(self, message: WireMessage) -> WireMessage:
        'Validate and evaluate a bounded vector of blinded Ristretto points.'
        if message.message_type != OPRF_EVALUATE_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected oprf.evaluate.request /  oprf.evaluate.request",
            )
        blinded_elements = self._decode_payload(message.payload)
        started = time.perf_counter()
        evaluated_elements = [
            encode_wire_point(scalar_multiply(self._scalar, element))
            for element in blinded_elements
        ]
        elapsed = time.perf_counter() - started
        with self._metrics_lock:
            self._evaluated_element_count += len(blinded_elements)
            self._evaluation_compute_seconds += elapsed
        return WireMessage.create(
            OPRF_EVALUATE_RESPONSE,
            {"evaluated_elements": evaluated_elements},
            request_id=message.request_id,
        )

    def metrics_snapshot(self) -> dict[str, int | float]:
        'Return length-only KS cryptographic-work counters for benchmarks.'
        with self._metrics_lock:
            return {
                "evaluated_element_count": self._evaluated_element_count,
                "evaluation_compute_seconds": self._evaluation_compute_seconds,
            }

    @staticmethod
    def _decode_payload(payload: Mapping[str, Any]) -> list[bytes]:
        'Extract only canonical blinded Ristretto wire points.'
        if set(payload) != {"blinded_elements"}:
            raise RequestRejected(
                400,
                "invalid_oprf_payload",
                "payload must contain only blinded_elements /  blinded_elements",
            )
        encoded_elements = payload["blinded_elements"]
        if (
            not isinstance(encoded_elements, Sequence)
            or isinstance(encoded_elements, (str, bytes))
            or not 1 <= len(encoded_elements) <= MAX_BATCH_ELEMENTS
        ):
            raise RequestRejected(
                400,
                "invalid_oprf_batch",
                "blinded_elements must be a non-empty bounded array / blinded_elements ",
            )
        try:
            return [decode_wire_point(element) for element in encoded_elements]
        except OprfValidationError as error:
            raise RequestRejected(400, "invalid_blinded_element", str(error)) from error


def build_ks_oprf_server(
    host: str,
    *,
    key_path: str | Path,
    port: int = 18081,
    ssl_context: ssl.SSLContext | None = None,
) -> ThreadedJsonServer:
    'Build the deployable KS listener; callers own its lifecycle.'
    material = OprfKeyStore(key_path).load_or_create()
    service = KeyServerOprfService(material)
    return ThreadedJsonServer(host, port, service.router, ssl_context=ssl_context)
