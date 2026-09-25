"""Cross-platform HTTP JSON transport for DwT-FL services. / DwT-FL 服务的跨平台 HTTP JSON 传输层。"""

from __future__ import annotations

import ssl
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .protocol import ProtocolError, WireMessage, error_message


DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0
DEFAULT_MAX_MESSAGE_BYTES: Final[int] = 4 * 1024 * 1024


class CommunicationError(RuntimeError):
    """Base class for transport and remote-service failures. / 传输与远端服务失败的基类。"""


class TransportError(CommunicationError):
    """Raised when a request cannot reach or decode a service. / 请求无法到达或无法解码服务响应时引发。"""


@dataclass(frozen=True, slots=True)
class TrafficRecord:
    """One length-only observation for a JSON HTTP exchange.

    一次 JSON HTTP 交互的仅长度观测。

    The record deliberately excludes request bodies, protected labels, and
    plaintext records. It supports reproducible communication-overhead
    accounting without collecting protocol secrets. 该记录刻意不保存请求体、
    受保护标签或明文记录；它可支持可复现的通信开销核算，而不收集协议秘密。
    """

    base_url: str
    path: str
    method: str
    request_message_type: str
    response_message_type: str | None
    request_body_bytes: int
    response_body_bytes: int
    request_payload_bytes: int
    response_payload_bytes: int
    status_code: int | None
    elapsed_seconds: float
    error_type: str | None


class TrafficRecorder:
    """Thread-safe, length-only recorder for real client HTTP exchanges.

    面向真实客户端 HTTP 交互的线程安全仅长度记录器。
    """

    def __init__(self) -> None:
        """Create an empty recorder. / 创建空记录器。"""
        self._lock = threading.Lock()
        self._records: list[TrafficRecord] = []

    def record(self, observation: TrafficRecord) -> None:
        """Append one immutable observation. / 追加一条不可变观测。"""
        with self._lock:
            self._records.append(observation)

    def snapshot(self) -> tuple[TrafficRecord, ...]:
        """Return a stable copy of all recorded exchanges. / 返回全部交互的稳定副本。"""
        with self._lock:
            return tuple(self._records)


def _payload_byte_length(message: WireMessage) -> int:
    """Return canonical UTF-8 bytes of a message payload without its envelope.

    返回消息载荷的规范 UTF-8 字节数，不包含信封字段。
    """
    # The canonical full body is already produced by ``WireMessage``. Removing
    # the fixed envelope fields here isolates application payload growth from
    # request ID and schema framing growth. ``repr`` is intentionally avoided
    # because it is not a portable wire representation. 规范完整报文体已由
    # ``WireMessage`` 生成；此处移除固定信封字段，以区分应用载荷增长与请求 ID、
    # 模式封装增长；刻意不使用不具可移植线协议语义的 ``repr``。
    import json

    return len(json.dumps(
        dict(message.payload), allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8"))


@dataclass
class RemoteServiceError(CommunicationError):
    """A structured non-success response returned by a service. / 服务返回的结构化非成功响应。"""

    status_code: int
    code: str
    detail: str
    request_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Render an actionable error without discarding the request identifier.

        呈现不丢失请求标识的可操作错误。
        """
        return (
            f"remote service returned HTTP {self.status_code}: {self.code}: {self.detail} "
            f"(request_id={self.request_id})"
        )


@dataclass
class RequestRejected(CommunicationError):
    """An intentional server-side rejection exposed to a route handler.

    暴露给路由处理函数的预期服务端拒绝。

    Exception instances must remain mutable because CPython attaches traceback,
    context, and cause attributes while raising them. In particular, Python
    3.10 cannot safely raise a ``frozen=True, slots=True`` dataclass exception:
    its generated ``__setattr__`` may fail with ``super(type, obj)``. These
    wire-error records therefore intentionally do not use frozen slots. 异常
    实例在抛出时必须保持可变，因为 CPython 会附加 traceback、context 与 cause
    属性。尤其 Python 3.10 无法安全抛出 ``frozen=True, slots=True`` 数据类
    异常：其生成的 ``__setattr__`` 可能产生 ``super(type, obj)``。因此这些
    线协议错误记录刻意不使用冻结槽位。
    """

    status_code: int
    code: str
    detail: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid HTTP status codes before they reach the wire.

        在写入网络前拒绝无效 HTTP 状态码。
        """
        if not 400 <= self.status_code <= 599:
            raise ValueError(
                "rejection status must be 4xx or 5xx / 拒绝状态必须是 4xx 或 5xx"
            )
        try:
            # Constructing a temporary wire error gives all callers one strict
            # JSON validation boundary for optional error context. 构造临时线协议
            # 错误可为可选错误上下文提供统一且严格的 JSON 校验边界。
            error_message(self.code, self.detail, context=self.context)
        except ProtocolError as error:
            raise ValueError(
                "rejection context must be JSON serializable / 拒绝上下文必须可 JSON 序列化"
            ) from error


RouteHandler = Callable[[WireMessage], WireMessage]
"""A service route consumes and returns validated envelopes. / 服务路由消费并返回已验证的信封。"""


def _validated_path(path: str) -> str:
    """Validate one absolute API path without query or fragment data.

    验证不含查询串和片段的绝对 API 路径。
    """
    if not isinstance(path, str):
        raise ValueError("path must be a string / 路径必须是字符串")
    parsed_path = urlsplit(path)
    if (
        not path.startswith("/")
        or parsed_path.scheme
        or parsed_path.netloc
        or parsed_path.query
        or parsed_path.fragment
    ):
        raise ValueError(
            "path must be an absolute path without query data / "
            "路径必须是不含查询数据的绝对路径"
        )
    return path


class JsonRouter:
    """Thread-safe registry of versioned JSON message routes. / 版本化 JSON 消息路由的线程安全注册表。"""

    def __init__(self) -> None:
        """Create an empty route registry. / 创建空路由注册表。"""
        self._routes: dict[tuple[str, str], RouteHandler] = {}
        self._lock = threading.Lock()

    def add(self, method: str, path: str, handler: RouteHandler) -> None:
        """Register one unique HTTP-method and API-path pair.

        注册一个唯一的 HTTP 方法与 API 路径对。
        """
        normalized_method = method.upper()
        normalized_path = _validated_path(path)
        if not callable(handler):
            raise TypeError("handler must be callable / 处理函数必须可调用")
        route_key = (normalized_method, normalized_path)
        with self._lock:
            if route_key in self._routes:
                raise ValueError(
                    f"route already exists: {normalized_method} {normalized_path} / 路由已存在"
                )
            self._routes[route_key] = handler

    def dispatch(self, method: str, path: str, message: WireMessage) -> WireMessage:
        """Dispatch one validated message or produce a structured 404. / 分发一个已验证消息，或产生结构化 404。"""
        route_key = (method.upper(), _validated_path(path))
        with self._lock:
            handler = self._routes.get(route_key)
        if handler is None:
            raise RequestRejected(
                404,
                "route_not_found",
                f"no route for {route_key[0]} {route_key[1]}",
            )
        response = handler(message)
        if not isinstance(response, WireMessage):
            raise TypeError("route handlers must return WireMessage / 路由处理函数必须返回 WireMessage")
        if response.request_id != message.request_id:
            raise RequestRejected(
                500,
                "request_id_mismatch",
                "response request_id must equal request request_id / "
                "响应 request_id 必须等于请求 request_id",
            )
        return response


class JsonHttpClient:
    """Synchronous standard-library client for AS, KS, or test clients.

    面向 AS、KS 或测试客户端的同步标准库客户端。
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        ssl_context: ssl.SSLContext | None = None,
        traffic_recorder: TrafficRecorder | None = None,
    ) -> None:
        """Configure a service endpoint and bounded response reader. / 配置服务端点与有界响应读取器。"""
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(
                "base_url must include http(s) scheme and host / "
                "base_url 必须包含 http(s) 协议和主机"
            )
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("base_url must not include query or fragment / base_url 不得包含查询串或片段")
        if timeout_seconds <= 0 or max_message_bytes <= 0:
            raise ValueError(
                "timeout_seconds and max_message_bytes must be positive / "
                "timeout_seconds 和 max_message_bytes 必须为正数"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_message_bytes = max_message_bytes
        self.ssl_context = ssl_context
        self.traffic_recorder = traffic_recorder

    def send(
        self,
        path: str,
        message: WireMessage,
        *,
        method: str = "POST",
    ) -> WireMessage:
        """Send one envelope and return a validated matching response.
        """
        normalized_path = _validated_path(path)
        request_url = f"{self.base_url}{normalized_path}"
        request_body = message.to_json_bytes()
        request = Request(
            request_url,
            data=request_body,
            method=method.upper(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "X-DwT-Request-Id": message.request_id,
            },
        )
        started = time.perf_counter()
        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self.ssl_context,
            ) as response:
                status_code = response.status
                response_body = response.read(self.max_message_bytes + 1)
        except HTTPError as error:
            status_code = error.code
            response_body = error.read(self.max_message_bytes + 1)
        except (OSError, TimeoutError, URLError) as error:
            self._record_traffic(
                normalized_path, method, message, request_body, None, b"",
                time.perf_counter() - started, type(error).__name__,
            )
            raise TransportError(
                f"request to {request_url} failed for request_id={message.request_id} / "
                f"向 {request_url} 发起的请求失败，request_id={message.request_id}"
            ) from error
        self._record_traffic(
            normalized_path, method, message, request_body, status_code, response_body,
            time.perf_counter() - started, None,
        )
        return self._decode_response(status_code, response_body, message.request_id)

    def _record_traffic(
        self,
        path: str,
        method: str,
        request: WireMessage,
        request_body: bytes,
        status_code: int | None,
        response_body: bytes,
        elapsed_seconds: float,
        error_type: str | None,
    ) -> None:
        """Record one completed or failed exchange without affecting transport.

        记录一次完成或失败的交互，且绝不影响传输逻辑。
        """
        recorder = self.traffic_recorder
        if recorder is None:
            return
        response: WireMessage | None = None
        try:
            if response_body:
                response = WireMessage.from_json_bytes(response_body)
        except ProtocolError:
            # The normal decoder will surface the protocol failure. The recorder
            # must remain observational and never mask it. 常规解码器会报告协议
            # 错误；记录器必须保持纯观测性，绝不能掩盖该错误。
            pass
        recorder.record(TrafficRecord(
            base_url=self.base_url,
            path=path,
            method=method.upper(),
            request_message_type=request.message_type,
            response_message_type=None if response is None else response.message_type,
            request_body_bytes=len(request_body),
            response_body_bytes=len(response_body),
            request_payload_bytes=_payload_byte_length(request),
            response_payload_bytes=0 if response is None else _payload_byte_length(response),
            status_code=status_code,
            elapsed_seconds=elapsed_seconds,
            error_type=error_type,
        ))

    def _decode_response(
        self,
        status_code: int,
        body: bytes,
        request_id: str,
    ) -> WireMessage:
        """Decode success and error responses under one bounded contract.

        依据同一有界契约解码成功与错误响应。
        """
        if len(body) > self.max_message_bytes:
            raise TransportError("response exceeds max_message_bytes / 响应超过 max_message_bytes")
        try:
            response = WireMessage.from_json_bytes(body)
        except ProtocolError as error:
            raise TransportError(
                "service response violates the wire protocol / 服务响应违反线协议"
            ) from error
        if response.request_id != request_id:
            raise TransportError(
                "response request_id does not match request / "
                "响应 request_id 与请求不匹配"
            )
        if response.message_type == "protocol.error" or not 200 <= status_code < 300:
            code = str(response.payload.get("code", "unexpected_http_status"))
            detail = str(
                response.payload.get("detail", "service returned a non-success response")
            )
            raise RemoteServiceError(
                status_code,
                code,
                detail,
                response.request_id,
                dict(response.payload),
            )
        return response


class _ReusableThreadingHttpServer(ThreadingHTTPServer):
    """Threading server configured for repeatable local test binds. / 配置为可重复本地绑定的线程服务。"""

    allow_reuse_address = True
    daemon_threads = True
    # The standard-library default backlog is only five connections.  A local
    # FedAvg distribution can legitimately have several clients opening their
    # next bounded checkpoint-read connection at the same instant.  Keep a
    # bounded but sufficiently large kernel accept queue; this changes neither
    # route ordering nor the configured application-level request slots.
    # 标准库默认监听队列仅有五个连接。本地 FedAvg 分发时，多个客户端可以在同一
    # 时刻为下一次有界检查点读取建立连接。这里保留有界但足够大的内核接收队列；
    # 它不改变路由顺序，也不改变已配置的应用层请求槽位。
    request_queue_size = 128


class _ResizableRequestSlots:
    """Adjust a server's request limit without replacing its listening socket.

    在不替换监听套接字的情况下调整服务端请求上限。
    """

    def __init__(self, limit: int) -> None:
        """Create a positive initial concurrency limit. / 创建正的初始并发上限。"""
        self._limit = limit
        self._active = 0
        self._condition = threading.Condition()

    def acquire(self) -> None:
        """Wait for one request slot. / 等待一个请求槽位。"""
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1

    def release(self) -> None:
        """Release one request slot. / 释放一个请求槽位。"""
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def set_limit(self, limit: int) -> None:
        """Apply a new positive limit to future and waiting requests.

        将新的正上限应用于未来和正在等待的请求。
        """
        if limit < 1:
            raise ValueError("request limit must be positive / 请求上限必须为正数")
        with self._condition:
            self._limit = limit
            self._condition.notify_all()


def _handler_type(
    router: JsonRouter,
    max_message_bytes: int,
    request_slots: _ResizableRequestSlots | None,
    control_paths: frozenset[str],
) -> type[BaseHTTPRequestHandler]:
    """Create one request handler bound to a router instance.

    创建绑定到一个路由实例的请求处理器。
    """

    class JsonRequestHandler(BaseHTTPRequestHandler):
        """Handle one POST envelope without service-specific logic.

        在没有服务特定逻辑的情况下处理一个 POST 信封。
        """

        protocol_version = "HTTP/1.1"
        server_version = "DwTFLJson/1.0"

        def do_POST(self) -> None:
            """Decode, route, and encode one JSON request. / 解码、路由并编码一个 JSON 请求。"""
            # Liveness traffic must not wait behind a saturated data plane.
            # A heartbeat only refreshes a lease and retrieves instructions;
            # it does not execute the benchmarked label-index operation.  Keep
            # it on a bounded HTTP control path so a 0.1-second fault lease is
            # not invalidated by unrelated OPRF, registration, or upload work.
            # 保活流量不能在饱和的数据平面之后排队。心跳仅刷新租约并获取指令，
            # 不执行被测的标签索引操作；因此它使用有界 HTTP 控制路径，避免 0.1 秒
            # 故障租约被无关的 OPRF、登记或上传工作错误地耗尽。
            holds_data_slot = request_slots is not None and self.path not in control_paths
            if holds_data_slot:
                request_slots.acquire()
            try:
                self._dispatch_post()
            finally:
                if holds_data_slot:
                    request_slots.release()

        def _dispatch_post(self) -> None:
            """Process one request after acquiring an optional backend slot.

            在获取可选后端槽位后处理一个请求。
            """
            request_id: str | None = None
            try:
                message = self._read_message()
                request_id = message.request_id
                response = router.dispatch("POST", self.path, message)
                self._write_message(200, response)
            except (BrokenPipeError, ConnectionResetError):
                # The peer can time out or close during a long, valid service
                # operation. Its response is no longer deliverable, so do not
                # attempt a second error response or emit a misleading server
                # traceback. 对端可能在长但有效的服务操作期间超时或关闭连接；
                # 此时响应已无法送达，不能再次写入错误响应，也不应输出误导性的
                # 服务端回溯。
                return
            except RequestRejected as error:
                self._write_message(
                    error.status_code,
                    error_message(
                        error.code,
                        error.detail,
                        request_id=request_id,
                        context=error.context,
                    ),
                )
            except ProtocolError as error:
                self._write_message(
                    400,
                    error_message("invalid_message", str(error), request_id=request_id),
                )
            except Exception:
                # Keep the full server-side cause in the service log without
                # exposing implementation details through the HTTP envelope.
                # 在服务日志中保留完整的服务端原因，但绝不通过 HTTP 信封暴露实现细节。
                traceback.print_exc()
                self._write_message(
                    500,
                    error_message(
                        "internal_error",
                        "service failed while handling the request / 服务处理请求时失败",
                        request_id=request_id,
                    ),
                )

        def _read_message(self) -> WireMessage:
            """Read one bounded request body using Content-Length. / 使用 Content-Length 读取一个有界请求体。"""
            content_length_header = self.headers.get("Content-Length")
            if content_length_header is None:
                raise RequestRejected(
                    411,
                    "content_length_required",
                    "Content-Length is required / 必须提供 Content-Length",
                )
            try:
                content_length = int(content_length_header)
            except ValueError as error:
                raise RequestRejected(
                    400,
                    "invalid_content_length",
                    "Content-Length is invalid / Content-Length 无效",
                ) from error
            if not 0 <= content_length <= max_message_bytes:
                raise RequestRejected(
                    413,
                    "message_too_large",
                    "request body exceeds configured limit / 请求体超过配置限制",
                )
            return WireMessage.from_json_bytes(self.rfile.read(content_length))

        def _write_message(self, status_code: int, message: WireMessage) -> None:
            """Write one deterministic JSON response with a byte length. / 写入带字节长度的确定性 JSON 响应。"""
            body = message.to_json_bytes()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-DwT-Request-Id", message.request_id)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *arguments: object) -> None:
            """Suppress default stderr logging; callers own structured logs.

            抑制默认 stderr 日志；调用方负责结构化日志。
            """

    return JsonRequestHandler


class ThreadedJsonServer:
    """Lifecycle wrapper for a local AS or KS JSON server. / 本地 AS 或 KS JSON 服务的生命周期封装。"""

    def __init__(
        self,
        host: str,
        port: int,
        router: JsonRouter,
        *,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        max_concurrent_requests: int | None = None,
        control_paths: Iterable[str] = (),
        ssl_context: ssl.SSLContext | None = None,
        advertised_host: str | None = None,
    ) -> None:
        """Bind a service without starting its serving thread.

        绑定服务但不启动其服务线程。
        """
        if not 0 <= port <= 65535 or max_message_bytes <= 0:
            raise ValueError(
                "port and max_message_bytes are invalid / port 和 max_message_bytes 无效"
            )
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError(
                "max_concurrent_requests must be positive when set / "
                "设置时 max_concurrent_requests 必须为正数"
            )
        self._control_paths = frozenset(control_paths)
        if any(not path.startswith("/") for path in self._control_paths):
            raise ValueError(
                "control paths must be absolute HTTP paths / 控制路径必须是绝对 HTTP 路径"
            )
        self._ssl_context = ssl_context
        self._request_slots = (
            _ResizableRequestSlots(max_concurrent_requests)
            if max_concurrent_requests is not None
            else None
        )
        handler = _handler_type(
            router,
            max_message_bytes,
            self._request_slots,
            self._control_paths,
        )
        self._server = _ReusableThreadingHttpServer((host, port), handler)
        if ssl_context is not None:
            self._server.socket = ssl_context.wrap_socket(
                self._server.socket,
                server_side=True,
            )
        self._thread: threading.Thread | None = None
        self._closed = False
        self._advertised_host = advertised_host or (
            "127.0.0.1" if host == "0.0.0.0" else host
        )

    @property
    def port(self) -> int:
        """Return the actual bound TCP port. / 返回实际绑定的 TCP 端口。"""
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """Return a loopback-safe endpoint for the bound server. / 返回可用于绑定服务的安全回环端点。"""
        scheme = "https" if self._ssl_context is not None else "http"
        return f"{scheme}://{self._advertised_host}:{self.port}"

    def start(self) -> None:
        """Start serving on one daemon thread. / 在一个守护线程上开始提供服务。"""
        if self._closed:
            raise RuntimeError("server is closed / 服务已关闭")
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()

    def set_max_concurrent_requests(self, limit: int) -> None:
        """Adjust an existing bounded request handler pool at an idle boundary.

        在空闲边界调整现有的有界请求处理池。
        """
        if self._request_slots is None:
            raise RuntimeError(
                "server has no configurable request limit / 服务没有可配置的请求上限"
            )
        self._request_slots.set_limit(limit)

    def close(self) -> None:
        """Stop serving and release the bound socket. / 停止服务并释放绑定套接字。"""
        if self._closed:
            return
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()
        self._closed = True

    def __enter__(self) -> "ThreadedJsonServer":
        """Start the service when entering a context. / 进入上下文时启动服务。"""
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Close the service when leaving a context.

        离开上下文时关闭服务。
        """
        self.close()
