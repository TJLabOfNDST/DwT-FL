"""Native Ristretto255 KS OPRF endpoint for the DwT-FL protocol.

DwT-FL 协议的原生 Ristretto255 KS OPRF 端点。
"""

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
"""Hard per-request point bound to control JSON and server memory use. / 控制 JSON 与服务器内存的单请求点上限。"""


class KeyServerOprfService:
    """Evaluate blinded Ristretto points with the KS secret scalar.

    使用 KS 私钥标量对盲化 Ristretto 点进行求值。

    All private-scalar multiplication is delegated to the required libsodium
    backend. 所有涉及私钥的标量乘法均委托给必需的 libsodium 后端。
    """

    def __init__(self, material: OprfKeyMaterial) -> None:
        """Keep a validated KS scalar only in the KS service process.

        仅在 KS 服务进程中保留已验证的 KS 私钥标量。
        """
        self._scalar = material.scalar
        self._metrics_lock = threading.Lock()
        self._evaluated_element_count = 0
        self._evaluation_compute_seconds = 0.0
        self.router = JsonRouter()
        self.router.add("POST", KeyServerPath.EVALUATE_OPRF.value, self.evaluate)

    def evaluate(self, message: WireMessage) -> WireMessage:
        """Validate and evaluate a bounded vector of blinded Ristretto points.

        验证并求值一个有界的盲化 Ristretto 点向量。
        """
        if message.message_type != OPRF_EVALUATE_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected oprf.evaluate.request / 应为 oprf.evaluate.request",
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
        """Return length-only KS cryptographic-work counters for benchmarks.

        为基准测试返回仅含长度的 KS 密码工作计数器。
        """
        with self._metrics_lock:
            return {
                "evaluated_element_count": self._evaluated_element_count,
                "evaluation_compute_seconds": self._evaluation_compute_seconds,
            }

    @staticmethod
    def _decode_payload(payload: Mapping[str, Any]) -> list[bytes]:
        """Extract only canonical blinded Ristretto wire points.

        仅提取规范的盲化 Ristretto 线协议点。
        """
        if set(payload) != {"blinded_elements"}:
            raise RequestRejected(
                400,
                "invalid_oprf_payload",
                "payload must contain only blinded_elements / 载荷只能包含 blinded_elements",
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
                "blinded_elements must be a non-empty bounded array / blinded_elements 必须为非空且有上限的数组",
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
    """Build the deployable KS listener; callers own its lifecycle.

    构建可部署的 KS 监听器；其生命周期由调用者管理。传入 TLS 上下文可启用 HTTPS。
    """
    material = OprfKeyStore(key_path).load_or_create()
    service = KeyServerOprfService(material)
    return ThreadedJsonServer(host, port, service.router, ssl_context=ssl_context)
