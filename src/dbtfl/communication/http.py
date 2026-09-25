'Cross-platform HTTP JSON transport for DwT-FL services. / DwT-FL'

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
    'Base class for transport and remote-service failures.'


class TransportError(CommunicationError):
    'Raised when a request cannot reach or decode a service.'


@dataclass(frozen=True, slots=True)
class TrafficRecord:
    'One length-only observation for a JSON HTTP exchange.\n    The record deliberately excludes request bodies, protected labels, and\n    plaintext records. It supports reproducible communication-overhead\n    accounting without collecting protocol secrets.'

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
    'Thread-safe, length-only recorder for real client HTTP exchanges.'

    def __init__(self) -> None:
        'Create an empty recorder.'
        self._lock = threading.Lock()
        self._records: list[TrafficRecord] = []

    def record(self, observation: TrafficRecord) -> None:
        'Append one immutable observation.'
        with self._lock:
            self._records.append(observation)

    def snapshot(self) -> tuple[TrafficRecord, ...]:
        'Return a stable copy of all recorded exchanges.'
        with self._lock:
            return tuple(self._records)


def _payload_byte_length(message: WireMessage) -> int:
    'Return canonical UTF-8 bytes of a message payload without its envelope.'
    # The canonical full body is already produced by ``WireMessage``. Removing
    # the fixed envelope fields here isolates application payload growth from
    # request ID and schema framing growth. ``repr`` is intentionally avoided
    # because it is not a portable wire representation.
    # ``WireMessage``
    
    import json

    return len(json.dumps(
        dict(message.payload), allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8"))


@dataclass
class RemoteServiceError(CommunicationError):
    'A structured non-success response returned by a service.'

    status_code: int
    code: str
    detail: str
    request_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        'Render an actionable error without discarding the request identifier.'
        return (
            f"remote service returned HTTP {self.status_code}: {self.code}: {self.detail} "
            f"(request_id={self.request_id})"
        )


@dataclass
class RequestRejected(CommunicationError):
    'An intentional server-side rejection exposed to a route handler.\n    Exception instances must remain mutable because CPython attaches traceback\n    context, and cause attributes while raising them. In particular, Python\n    3.10 cannot safely raise a ``frozen=True, slots=True`` dataclass exception\n    its generated ``__setattr__`` may fail with ``super(type, obj)``. These\n    wire-error records therefore intentionally do not use frozen slots.'

    status_code: int
    code: str
    detail: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        'Reject invalid HTTP status codes before they reach the wire.'
        if not 400 <= self.status_code <= 599:
            raise ValueError(
                "rejection status must be 4xx or 5xx /  4xx  5xx"
            )
        try:
            # Constructing a temporary wire error gives all callers one strict
            # JSON validation boundary for optional error context.
            
            error_message(self.code, self.detail, context=self.context)
        except ProtocolError as error:
            raise ValueError(
                "rejection context must be JSON serializable /  JSON "
            ) from error


RouteHandler = Callable[[WireMessage], WireMessage]
"""A service route consumes and returns validated envelopes. / """


def _validated_path(path: str) -> str:
    'Validate one absolute API path without query or fragment data.'
    if not isinstance(path, str):
        raise ValueError("path must be a string / ")
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
            ""
        )
    return path


class JsonRouter:
    'Thread-safe registry of versioned JSON message routes.'

    def __init__(self) -> None:
        'Create an empty route registry.'
        self._routes: dict[tuple[str, str], RouteHandler] = {}
        self._lock = threading.Lock()

    def add(self, method: str, path: str, handler: RouteHandler) -> None:
        'Register one unique HTTP-method and API-path pair.'
        normalized_method = method.upper()
        normalized_path = _validated_path(path)
        if not callable(handler):
            raise TypeError("handler must be callable / ")
        route_key = (normalized_method, normalized_path)
        with self._lock:
            if route_key in self._routes:
                raise ValueError(
                    f"route already exists: {normalized_method} {normalized_path} / "
                )
            self._routes[route_key] = handler

    def dispatch(self, method: str, path: str, message: WireMessage) -> WireMessage:
        'Dispatch one validated message or produce a structured 404.'
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
            raise TypeError("route handlers must return WireMessage /  WireMessage")
        if response.request_id != message.request_id:
            raise RequestRejected(
                500,
                "request_id_mismatch",
                "response request_id must equal request request_id / "
                " request_id  request_id",
            )
        return response


class JsonHttpClient:
    'Synchronous standard-library client for AS, KS, or test clients.'

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        ssl_context: ssl.SSLContext | None = None,
        traffic_recorder: TrafficRecorder | None = None,
    ) -> None:
        'Configure a service endpoint and bounded response reader.'
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(
                "base_url must include http(s) scheme and host / "
                "base_url  http(s) "
            )
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("base_url must not include query or fragment / base_url ")
        if timeout_seconds <= 0 or max_message_bytes <= 0:
            raise ValueError(
                "timeout_seconds and max_message_bytes must be positive / "
                "timeout_seconds  max_message_bytes "
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
                f" {request_url} request_id={message.request_id}"
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
        'Record one completed or failed exchange without affecting transport.'
        recorder = self.traffic_recorder
        if recorder is None:
            return
        response: WireMessage | None = None
        try:
            if response_body:
                response = WireMessage.from_json_bytes(response_body)
        except ProtocolError:
            # The normal decoder will surface the protocol failure. The recorder
            # must remain observational and never mask it.
            
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
        'Decode success and error responses under one bounded contract.'
        if len(body) > self.max_message_bytes:
            raise TransportError("response exceeds max_message_bytes /  max_message_bytes")
        try:
            response = WireMessage.from_json_bytes(body)
        except ProtocolError as error:
            raise TransportError(
                "service response violates the wire protocol / "
            ) from error
        if response.request_id != request_id:
            raise TransportError(
                "response request_id does not match request / "
                " request_id "
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
    'Threading server configured for repeatable local test binds.'

    allow_reuse_address = True
    daemon_threads = True
    # The standard-library default backlog is only five connections.  A local
    # FedAvg distribution can legitimately have several clients opening their
    # next bounded checkpoint-read connection at the same instant.  Keep a
    # bounded but sufficiently large kernel accept queue; this changes neither
    # route ordering nor the configured application-level request slots.
    
    
    
    request_queue_size = 128


class _ResizableRequestSlots:
    "Adjust a server's request limit without replacing its listening socket."

    def __init__(self, limit: int) -> None:
        'Create a positive initial concurrency limit.'
        self._limit = limit
        self._active = 0
        self._condition = threading.Condition()

    def acquire(self) -> None:
        'Wait for one request slot.'
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1

    def release(self) -> None:
        'Release one request slot.'
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def set_limit(self, limit: int) -> None:
        'Apply a new positive limit to future and waiting requests.'
        if limit < 1:
            raise ValueError("request limit must be positive / ")
        with self._condition:
            self._limit = limit
            self._condition.notify_all()


def _handler_type(
    router: JsonRouter,
    max_message_bytes: int,
    request_slots: _ResizableRequestSlots | None,
    control_paths: frozenset[str],
) -> type[BaseHTTPRequestHandler]:
    'Create one request handler bound to a router instance.'

    class JsonRequestHandler(BaseHTTPRequestHandler):
        'Handle one POST envelope without service-specific logic.'

        protocol_version = "HTTP/1.1"
        server_version = "DwTFLJson/1.0"

        def do_POST(self) -> None:
            'Decode, route, and encode one JSON request.'
            # Liveness traffic must not wait behind a saturated data plane.
            # A heartbeat only refreshes a lease and retrieves instructions;
            # it does not execute the benchmarked label-index operation.  Keep
            # it on a bounded HTTP control path so a 0.1-second fault lease is
            # not invalidated by unrelated OPRF, registration, or upload work.
            
            
            
            holds_data_slot = request_slots is not None and self.path not in control_paths
            if holds_data_slot:
                request_slots.acquire()
            try:
                self._dispatch_post()
            finally:
                if holds_data_slot:
                    request_slots.release()

        def _dispatch_post(self) -> None:
            'Process one request after acquiring an optional backend slot.'
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
                # traceback.
                
                
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
                
                traceback.print_exc()
                self._write_message(
                    500,
                    error_message(
                        "internal_error",
                        "service failed while handling the request / ",
                        request_id=request_id,
                    ),
                )

        def _read_message(self) -> WireMessage:
            'Read one bounded request body using Content-Length.'
            content_length_header = self.headers.get("Content-Length")
            if content_length_header is None:
                raise RequestRejected(
                    411,
                    "content_length_required",
                    "Content-Length is required /  Content-Length",
                )
            try:
                content_length = int(content_length_header)
            except ValueError as error:
                raise RequestRejected(
                    400,
                    "invalid_content_length",
                    "Content-Length is invalid / Content-Length ",
                ) from error
            if not 0 <= content_length <= max_message_bytes:
                raise RequestRejected(
                    413,
                    "message_too_large",
                    "request body exceeds configured limit / ",
                )
            return WireMessage.from_json_bytes(self.rfile.read(content_length))

        def _write_message(self, status_code: int, message: WireMessage) -> None:
            'Write one deterministic JSON response with a byte length.'
            body = message.to_json_bytes()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-DwT-Request-Id", message.request_id)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *arguments: object) -> None:
            'Suppress default stderr logging; callers own structured logs.'

    return JsonRequestHandler


class ThreadedJsonServer:
    'Lifecycle wrapper for a local AS or KS JSON server.'

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
        'Bind a service without starting its serving thread.'
        if not 0 <= port <= 65535 or max_message_bytes <= 0:
            raise ValueError(
                "port and max_message_bytes are invalid / port  max_message_bytes "
            )
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError(
                "max_concurrent_requests must be positive when set / "
                " max_concurrent_requests "
            )
        self._control_paths = frozenset(control_paths)
        if any(not path.startswith("/") for path in self._control_paths):
            raise ValueError(
                "control paths must be absolute HTTP paths /  HTTP "
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
        'Return the actual bound TCP port.'
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        'Return a loopback-safe endpoint for the bound server.'
        scheme = "https" if self._ssl_context is not None else "http"
        return f"{scheme}://{self._advertised_host}:{self.port}"

    def start(self) -> None:
        'Start serving on one daemon thread.'
        if self._closed:
            raise RuntimeError("server is closed / ")
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()

    def set_max_concurrent_requests(self, limit: int) -> None:
        'Adjust an existing bounded request handler pool at an idle boundary.'
        if self._request_slots is None:
            raise RuntimeError(
                "server has no configurable request limit / "
            )
        self._request_slots.set_limit(limit)

    def close(self) -> None:
        'Stop serving and release the bound socket.'
        if self._closed:
            return
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
        self._server.server_close()
        self._closed = True

    def __enter__(self) -> "ThreadedJsonServer":
        'Start the service when entering a context.'
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        'Close the service when leaving a context.'
        self.close()
