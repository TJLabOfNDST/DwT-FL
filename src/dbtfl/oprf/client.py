"""Client-side Ristretto255 blind OPRF flow for DwT-FL.

DwT-FL 的客户端 Ristretto255 盲 OPRF 流程。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dbtfl.communication import JsonHttpClient, TransportError, WireMessage
from dbtfl.communication.endpoints import KeyServerPath

from .group14 import OprfValidationError
from .ristretto255 import (
    decode_wire_point,
    encode_wire_point,
    hash_to_group,
    protected_label,
    random_scalar,
    scalar_inverse,
    scalar_multiply,
)
from .service import MAX_BATCH_ELEMENTS, OPRF_EVALUATE_REQUEST, OPRF_EVALUATE_RESPONSE


@dataclass(frozen=True, slots=True)
class _PendingBlind:
    """Private inverse blinding scalar for one input record.

    一条输入记录对应的私有去盲标量逆元。
    """

    inverse: bytes


class OprfClient:
    """Run ``H(m)^r -> H(m)^(rk) -> H(m)^k`` over a KS transport.

    通过 KS 传输执行 ``H(m)^r -> H(m)^(rk) -> H(m)^k``。

    Ristretto's additive group representation implements the same blind-evaluate-
    unblind algebra as scalar multiplication.  The server receives only ``rH(m)``;
    it never receives the record nor the blinding scalar. Ristretto 的加法群表示
    以标量乘法实现相同的盲化、求值和去盲代数。服务器只收到 ``rH(m)``，不会收到
    记录或盲化标量。
    """

    def __init__(self, transport: JsonHttpClient, *, batch_size: int = 1024) -> None:
        """Bind the client to a KS JSON transport and a bounded wire batch size.

        将客户端绑定至 KS JSON 传输和有界线协议批大小。
        """
        if not 1 <= batch_size <= MAX_BATCH_ELEMENTS:
            raise ValueError(
                f"batch_size must be in 1..{MAX_BATCH_ELEMENTS} / "
                f"batch_size 必须位于 1..{MAX_BATCH_ELEMENTS}"
            )
        self._transport = transport
        self._batch_size = batch_size

    def evaluate(self, records: Sequence[str | bytes]) -> list[str]:
        """Return 512-hex native-index-compatible protected labels in record order.

        按记录顺序返回兼容原生索引的 512 位十六进制保护标签。
        """
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of str or bytes / records 必须是 str 或 bytes 序列")
        if not 1 <= len(records) <= MAX_BATCH_ELEMENTS:
            raise OprfValidationError(
                f"batch must contain 1..{MAX_BATCH_ELEMENTS} records / "
                f"批次必须包含 1..{MAX_BATCH_ELEMENTS} 条记录"
            )
        labels: list[str] = []
        for offset in range(0, len(records), self._batch_size):
            labels.extend(self._evaluate_batch(records[offset : offset + self._batch_size]))
        return labels

    def _evaluate_batch(self, records: Sequence[str | bytes]) -> list[str]:
        """Evaluate one network batch while retaining the caller's record order.

        对一个网络批次求值，同时保持调用者的记录顺序。
        """
        blinded_elements: list[str] = []
        pending_blinds: list[_PendingBlind] = []
        for record in records:
            base = hash_to_group(record)
            blinding_scalar = random_scalar()
            blinded_elements.append(encode_wire_point(scalar_multiply(blinding_scalar, base)))
            pending_blinds.append(_PendingBlind(scalar_inverse(blinding_scalar)))
        request = WireMessage.create(OPRF_EVALUATE_REQUEST, {"blinded_elements": blinded_elements})
        response = self._transport.send(KeyServerPath.EVALUATE_OPRF.value, request)
        evaluated_elements = self._validated_response(response, len(pending_blinds))
        return [
            protected_label(scalar_multiply(pending.inverse, element))
            for element, pending in zip(evaluated_elements, pending_blinds, strict=True)
        ]

    @staticmethod
    def _validated_response(response: WireMessage, expected_count: int) -> list[bytes]:
        """Validate response envelope and every returned Ristretto point.

        验证响应信封和每一个返回的 Ristretto 点。
        """
        if response.message_type != OPRF_EVALUATE_RESPONSE:
            raise TransportError("KS returned an unexpected OPRF response type / KS 返回了意外的 OPRF 响应类型")
        if set(response.payload) != {"evaluated_elements"}:
            raise TransportError("KS returned an invalid OPRF payload / KS 返回了无效 OPRF 载荷")
        encoded_elements = response.payload["evaluated_elements"]
        if (
            not isinstance(encoded_elements, Sequence)
            or isinstance(encoded_elements, (str, bytes))
            or len(encoded_elements) != expected_count
        ):
            raise TransportError("KS response count does not match request / KS 响应数量与请求不一致")
        try:
            return [decode_wire_point(element) for element in encoded_elements]
        except OprfValidationError as error:
            raise TransportError("KS response has an invalid Ristretto point / KS 响应含无效 Ristretto 点") from error
