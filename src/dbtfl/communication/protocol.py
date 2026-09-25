"""Versioned, JSON-only messages shared by AS, KS, and clients. / AS、KS 与客户端共享的版本化纯 JSON 消息。"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "1.0"
"""Current wire-schema version. / 当前线协议版本。"""

_MESSAGE_TYPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}$"
)
_REQUEST_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_-]{1,128}$"
)


class ProtocolError(ValueError):
    """Raised when a wire message violates the DwT-FL protocol. / 线协议消息违反 DwT-FL 协议时引发。"""


def _validated_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a JSON-serializable object payload. / 复制一个可 JSON 序列化的对象载荷。"""
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise ProtocolError("payload must be an object with string keys / 载荷必须是键为字符串的对象")
    copied_payload = dict(payload)
    try:
        json.dumps(copied_payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ProtocolError("payload must be JSON serializable / 载荷必须可被 JSON 序列化") from error
    return copied_payload


@dataclass(frozen=True, slots=True)
class WireMessage:
    """A request or response envelope independent from service internals. / 独立于服务内部实现的请求或响应信封。"""

    message_type: str
    payload: Mapping[str, Any]
    request_id: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate all protocol fields when constructing an envelope. / 构造信封时验证全部协议字段。"""
        if self.schema_version != SCHEMA_VERSION:
            raise ProtocolError(
                f"unsupported schema version: {self.schema_version} / "
                f"不支持的协议版本：{self.schema_version}"
            )
        if not _MESSAGE_TYPE_PATTERN.fullmatch(self.message_type):
            raise ProtocolError("message_type is invalid / message_type 无效")
        if not _REQUEST_ID_PATTERN.fullmatch(self.request_id):
            raise ProtocolError("request_id is invalid / request_id 无效")
        object.__setattr__(self, "payload", _validated_payload(self.payload))

    @classmethod
    def create(
        cls,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> "WireMessage":
        """Create a message with a fresh traceable request identifier. / 使用新的可追踪请求标识创建消息。"""
        return cls(
            message_type=message_type,
            payload=payload,
            request_id=request_id or uuid.uuid4().hex,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-object representation. / 返回规范的 JSON 对象表示。"""
        return {
            "schema_version": self.schema_version,
            "message_type": self.message_type,
            "request_id": self.request_id,
            "payload": dict(self.payload),
        }

    def to_json_bytes(self) -> bytes:
        """Encode one deterministic UTF-8 message body. / 编码一个确定性的 UTF-8 消息体。"""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw_message: bytes) -> "WireMessage":
        """Decode and validate one UTF-8 JSON object. / 解码并验证一个 UTF-8 JSON 对象。"""
        try:
            decoded_message = json.loads(raw_message.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError(
                "message body must be valid UTF-8 JSON / "
                "消息体必须是有效的 UTF-8 JSON"
            ) from error
        if not isinstance(decoded_message, Mapping):
            raise ProtocolError("message body must be a JSON object / 消息体必须是 JSON 对象")
        expected_fields = {"schema_version", "message_type", "request_id", "payload"}
        received_fields = set(decoded_message)
        if received_fields != expected_fields:
            raise ProtocolError(
                "message fields must equal "
                f"{sorted(expected_fields)} / 消息字段必须等于 {sorted(expected_fields)}"
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
    """Create a structured protocol error response with optional public context.

    创建带可选公开上下文的结构化协议错误响应。

    Error context is restricted to route-approved, JSON-safe protocol data. It
    lets a recoverable 4xx response carry paper-defined instructions without
    exposing service internals. 错误上下文仅限路由批准且可 JSON 序列化的协议数据，
    因而可让可恢复的 4xx 响应携带论文规定的指令，而不暴露服务内部状态。
    """
    extra_context = {} if context is None else dict(context)
    if {"code", "detail"}.intersection(extra_context):
        raise ProtocolError(
            "error context must not override code or detail / "
            "错误上下文不得覆盖 code 或 detail"
        )
    return WireMessage.create(
        "protocol.error",
        {"code": code, "detail": detail, **extra_context},
        request_id=request_id,
    )
