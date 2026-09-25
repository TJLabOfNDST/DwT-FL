'Versioned, JSON-only messages shared by AS, KS, and clients. / ASKS'

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "1.0"
"""Current wire-schema version. / """

_MESSAGE_TYPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}$"
)
_REQUEST_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_-]{1,128}$"
)


class ProtocolError(ValueError):
    'Raised when a wire message violates the DwT-FL protocol.'


def _validated_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    'Copy a JSON-serializable object payload.'
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ProtocolError("payload must be an object with string keys / ")
    copied_payload = dict(payload)
    try:
        json.dumps(copied_payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProtocolError("payload must be JSON serializable /  JSON ") from error
    return copied_payload


@dataclass(frozen=True, slots=True)
class WireMessage:
    'A request or response envelope independent from service internals.'

    message_type: str
    payload: Mapping[str, Any]
    request_id: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        'Validate all protocol fields when constructing an envelope.'
        if self.schema_version != SCHEMA_VERSION:
            raise ProtocolError(
                f"unsupported schema version: {self.schema_version} / "
                f"{self.schema_version}"
            )
        if not _MESSAGE_TYPE_PATTERN.fullmatch(self.message_type):
            raise ProtocolError("message_type is invalid / message_type ")
        if not _REQUEST_ID_PATTERN.fullmatch(self.request_id):
            raise ProtocolError("request_id is invalid / request_id ")
        object.__setattr__(self, "payload", _validated_payload(self.payload))

    @classmethod
    def create(
        cls,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> "WireMessage":
        'Create a message with a fresh traceable request identifier.'
        return cls(
            message_type=message_type,
            payload=payload,
            request_id=request_id or uuid.uuid4().hex,
        )

    def to_dict(self) -> dict[str, Any]:
        'Return the canonical JSON-object representation.'
        return {
            "schema_version": self.schema_version,
            "message_type": self.message_type,
            "request_id": self.request_id,
            "payload": dict(self.payload),
        }

    def to_json_bytes(self) -> bytes:
        'Encode one deterministic UTF-8 message body.'
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw_message: bytes) -> "WireMessage":
        'Decode and validate one UTF-8 JSON object.'
        try:
            decoded_message = json.loads(raw_message.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError(
                "message body must be valid UTF-8 JSON / "
                " UTF-8 JSON"
            ) from error
        if not isinstance(decoded_message, Mapping):
            raise ProtocolError("message body must be a JSON object /  JSON ")
        expected_fields = {"schema_version", "message_type", "request_id", "payload"}
        received_fields = set(decoded_message)
        if received_fields != expected_fields:
            raise ProtocolError(
                "message fields must equal "
                f"{sorted(expected_fields)} /  {sorted(expected_fields)}"
            )
        return cls(
            schema_version=decoded_message["schema_version"],
            message_type=decoded_message["message_type"],
            request_id=decoded_message["request_id"],
            payload=decoded_message["payload"],
        )


def error_message(
    code: str,
    detail: str,
    *,
    request_id: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> WireMessage:
    'Create a structured protocol error response with optional public context.\n    Error context is restricted to route-approved, JSON-safe protocol data. It\n    lets a recoverable 4xx response carry paper-defined instructions without\n    exposing service internals.'
    extra_context = {} if context is None else dict(context)
    if {"code", "detail"}.intersection(extra_context):
        raise ProtocolError(
            "error context must not override code or detail / "
            " code  detail"
        )
    return WireMessage.create(
        "protocol.error",
        {"code": code, "detail": detail, **extra_context},
        request_id=request_id,
    )
