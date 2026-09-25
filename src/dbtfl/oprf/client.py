'Client-side Ristretto255 blind OPRF flow for DwT-FL.\nDwT-FL'

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
    'Private inverse blinding scalar for one input record.'

    inverse: bytes


class OprfClient:
    "Run ``H(m)^r -> H(m)^(rk) -> H(m)^k`` over a KS transport.\n    Ristretto's additive group representation implements the same blind-evaluate\n    unblind algebra as scalar multiplication.  The server receives only ``rH(m)``\n    it never receives the record nor the blinding scalar. Ristretto"

    def __init__(self, transport: JsonHttpClient, *, batch_size: int = 1024) -> None:
        'Bind the client to a KS JSON transport and a bounded wire batch size.'
        if not 1 <= batch_size <= MAX_BATCH_ELEMENTS:
            raise ValueError(
                f"batch_size must be in 1..{MAX_BATCH_ELEMENTS} / "
                f"batch_size  1..{MAX_BATCH_ELEMENTS}"
            )
        self._transport = transport
        self._batch_size = batch_size

    def evaluate(self, records: Sequence[str | bytes]) -> list[str]:
        'Return 512-hex native-index-compatible protected labels in record order.'
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of str or bytes / records  str  bytes ")
        if not 1 <= len(records) <= MAX_BATCH_ELEMENTS:
            raise OprfValidationError(
                f"batch must contain 1..{MAX_BATCH_ELEMENTS} records / "
                f" 1..{MAX_BATCH_ELEMENTS} "
            )
        labels: list[str] = []
        for offset in range(0, len(records), self._batch_size):
            labels.extend(self._evaluate_batch(records[offset : offset + self._batch_size]))
        return labels

    def _evaluate_batch(self, records: Sequence[str | bytes]) -> list[str]:
        "Evaluate one network batch while retaining the caller's record order."
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
        'Validate response envelope and every returned Ristretto point.'
        if response.message_type != OPRF_EVALUATE_RESPONSE:
            raise TransportError("KS returned an unexpected OPRF response type / KS  OPRF ")
        if set(response.payload) != {"evaluated_elements"}:
            raise TransportError("KS returned an invalid OPRF payload / KS  OPRF ")
        encoded_elements = response.payload["evaluated_elements"]
        if (
            not isinstance(encoded_elements, Sequence)
            or isinstance(encoded_elements, (str, bytes))
            or len(encoded_elements) != expected_count
        ):
            raise TransportError("KS response count does not match request / KS ")
        try:
            return [decode_wire_point(element) for element in encoded_elements]
        except OprfValidationError as error:
            raise TransportError("KS response has an invalid Ristretto point / KS  Ristretto ") from error
