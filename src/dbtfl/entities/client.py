"""Client entity that persists local record--protected-label correspondences.

持久化本地数据—受保护标签对应关系的客户端实体。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import ssl
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dbtfl.communication import (
    CommunicationError,
    JsonHttpClient,
    RemoteServiceError,
    TrafficRecorder,
    TransportError,
    WireMessage,
)
from dbtfl.communication.endpoints import AggregationServerPath
from dbtfl.federation import chunk_file, sha256_file
from dbtfl.oprf import (
    OPRF_SUITE_IDENTIFIER,
    OprfClient,
    OprfValidationError,
    normalize_record,
    validate_protected_label,
)
from dbtfl.oprf.service import MAX_BATCH_ELEMENTS

from .aggregation_server import (
    AS_CLAIM_TASKS_REQUEST,
    AS_CLAIM_TASKS_RESPONSE,
    AS_HEARTBEAT_REQUEST,
    AS_HEARTBEAT_RESPONSE,
    AS_REGISTER_LABELS_REQUEST,
    AS_REGISTER_LABELS_RESPONSE,
    AS_REGISTER_REQUEST,
    AS_REGISTER_RESPONSE,
    AS_MODEL_CHUNK_REQUEST,
    AS_MODEL_CHUNK_RESPONSE,
    AS_MODEL_FINALIZE_REQUEST,
    AS_MODEL_FINALIZE_RESPONSE,
    AS_GLOBAL_MODEL_CHUNK_REQUEST,
    AS_GLOBAL_MODEL_CHUNK_RESPONSE,
    AS_MODEL_AGGREGATE_REQUEST,
    AS_MODEL_AGGREGATE_RESPONSE,
    AS_CONFIGURE_ROUND_REQUEST,
    AS_CONFIGURE_ROUND_RESPONSE,
)


LABEL_STORE_SCHEMA_VERSION: Final[str] = "2.0"
"""Version for the client-private label store. / 客户端私有标签存储的版本。"""

MODEL_CHUNK_TRANSPORT_ATTEMPTS: Final[int] = 3
"""Bounded attempts for one idempotent chunk request. / 单个幂等分块请求的有界尝试次数。"""

MODEL_CHUNK_RETRY_DELAY_SECONDS: Final[float] = 0.5
"""Initial backoff for one lost model-chunk response. / 单次模型分块响应丢失的初始退避时间。"""

GLOBAL_MODEL_CHUNK_TRANSPORT_ATTEMPTS: Final[int] = 4
"""Bounded attempts for one idempotent global-model read. / 单次幂等全局模型读取的有界尝试次数。"""

GLOBAL_MODEL_CHUNK_RETRY_DELAY_SECONDS: Final[float] = 0.25
"""Initial backoff for a lost global-model chunk response. / 全局模型分块响应丢失的初始退避时间。"""

MODEL_HASH_READ_BYTES: Final[int] = 1024 * 1024
"""Streaming read size for checkpoint hashing. / 检查点摘要的流式读取大小。"""

AS_OFFLINE_RECOVERY_INITIAL_DELAY_SECONDS: Final[float] = 0.05
"""Initial reconnect backoff after an AS offline response. / AS 返回离线响应后的初始重连退避时间。"""

AS_OFFLINE_RECOVERY_MAX_DELAY_SECONDS: Final[float] = 1.0
"""Maximum reconnect backoff that prevents a hot retry loop. / 防止热重试循环的最大重连退避时间。"""

AS_REGISTRATION_TRANSPORT_ATTEMPTS: Final[int] = 4
"""Attempts for a pre-SID idempotent registration request. / SID 分配前幂等注册请求的尝试次数。"""

AS_REGISTRATION_RETRY_DELAY_SECONDS: Final[float] = 0.05
"""Initial backoff for a transient pre-SID registration failure. / SID 分配前短暂注册失败的初始退避时间。"""

@dataclass(frozen=True, slots=True)
class ClientAsSession:
    """Client-side AS session assigned during registration.

    注册时分配的客户端侧 AS 会话。
    """

    sid: int
    heartbeat_interval_seconds: float


@dataclass(frozen=True, slots=True)
class RoundInstruction:
    """One AS heartbeat instruction for the active or recovered round.

    一条由 AS 心跳下发的当前轮或恢复轮指令。
    """

    protected_label: str
    task_id: int
    operation: str


@dataclass(frozen=True, slots=True)
class LabelRegistration:
    """One AS index-registration acknowledgement for a protected label.

    AS 对一个受保护标签的索引登记确认。
    """

    protected_label: str
    task_id: int


@dataclass(frozen=True, slots=True)
class TaskClaimDecision:
    """One separate CAS-phase decision returned by the AS.

    AS 返回的一个独立 CAS 阶段决策。
    """

    protected_label: str
    task_id: int
    operation: str
    state: str


@dataclass(frozen=True, slots=True)
class LocalTrainingQueues:
    """Private plaintext records routed by paper TRAIN and DEDUP decisions.

    由论文 TRAIN 与 DEDUP 决策路由的私有明文记录。
    """

    hot_records: tuple[bytes, ...]
    cold_records: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class ModelUpdateSubmission:
    """AS acknowledgement for one completed client model-update submission.

    AS 对一个完成客户端模型更新提交的确认。
    """

    round_id: int
    sid: int
    update_id: str
    committed_task_ids: tuple[int, ...]
    checkpoint_sha256: str


class ModelUpdateOwnershipLostError(RuntimeError):
    """A recovered upload no longer owns every task used for local training.

    恢复后的上传已不再拥有本地训练使用的全部任务。
    """

    def __init__(
        self,
        message: str,
        *,
        dedup_instructions: Sequence[TaskClaimDecision] = (),
        recovered_decisions: Sequence[TaskClaimDecision] = (),
    ) -> None:
        """Retain paper-level recovery instructions with the rejection.

        保留随拒绝返回的论文级恢复指令。

        ``dedup_instructions`` originates in the AS 409 payload and identifies
        labels safely taken by another trainer. ``recovered_decisions`` records
        a later client-side CAS observation when the loss was found while
        reconnecting. 两者分别记录 AS 409 载荷中已安全接管的标签，以及重连 CAS
        过程中发现所有权丢失时的最新观测。
        """
        # Do not use zero-argument ``super()`` in this protocol exception.
        # The exception is deliberately raised from recovery callbacks that may
        # be wrapped by test doubles or subprocess boundaries; explicitly
        # initializing RuntimeError keeps the wire-recovery error constructible
        # in every supported Python runtime. 不在此协议异常中使用无参数的
        # ``super()``。该异常会从可能被测试替身或子进程边界包装的恢复回调中
        # 抛出；显式初始化 RuntimeError 可确保所有支持的 Python 运行时都能
        # 构造该线协议恢复异常。
        RuntimeError.__init__(self, message)
        self.dedup_instructions = tuple(dedup_instructions)
        self.recovered_decisions = tuple(recovered_decisions)


class ClientLabelStoreError(RuntimeError):
    """Raised when the client-private record-label store is invalid or unsafe.

    客户端私有的数据—标签存储无效或不安全时引发。
    """


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Client identity, KS endpoint, and private label-store configuration.

    客户端身份、KS 端点及私有标签存储配置。
    """

    client_id: str
    ks_base_url: str
    label_store_path: Path
    as_base_url: str | None = None
    timeout_seconds: float = 10.0
    heartbeat_rpc_timeout_seconds: float = 2.0
    registration_rpc_timeout_seconds: float = 2.0
    oprf_batch_size: int = 1024
    oprf_suite: str = OPRF_SUITE_IDENTIFIER
    ssl_context: ssl.SSLContext | None = None
    traffic_recorder: TrafficRecorder | None = None
    model_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        """Reject incomplete client configuration before any RPC is attempted.

        在发起任何 RPC 前拒绝不完整的客户端配置。
        """
        if not self.client_id:
            raise ValueError("client_id must not be empty / client_id 不得为空")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive / timeout_seconds 必须为正数")
        if self.heartbeat_rpc_timeout_seconds <= 0:
            raise ValueError(
                "heartbeat_rpc_timeout_seconds must be positive / "
                "heartbeat_rpc_timeout_seconds 必须为正数"
            )
        if self.registration_rpc_timeout_seconds <= 0:
            raise ValueError(
                "registration_rpc_timeout_seconds must be positive / "
                "registration_rpc_timeout_seconds 必须为正数"
            )
        if not 1 <= self.oprf_batch_size <= MAX_BATCH_ELEMENTS:
            raise ValueError(
                f"oprf_batch_size must be in 1..{MAX_BATCH_ELEMENTS} / "
                f"oprf_batch_size 必须位于 1..{MAX_BATCH_ELEMENTS}"
            )
        if self.oprf_suite != OPRF_SUITE_IDENTIFIER:
            raise ValueError(
                "client must use the active Ristretto255 OPRF suite / "
                "客户端必须使用当前 Ristretto255 OPRF 套件"
            )
        if self.model_chunk_bytes < 1 or self.model_chunk_bytes > 2 * 1024 * 1024:
            raise ValueError("model_chunk_bytes must be in 1..2097152 / 模型分块字节数必须位于 1..2097152")


class ClientEntity:
    """Generate and remember protected labels while retaining raw records locally.

    生成并记住受保护标签，同时将原始数据保留在本地。

    The store encodes records with Base64 for lossless JSON storage.  Base64 is
    not encryption; deploy it on a private client filesystem with OS access
    controls.  存储使用 Base64 实现无损 JSON 编码；Base64 不是加密，部署时必须
    放在受操作系统访问控制保护的客户端私有文件系统中。
    """

    def __init__(self, config: ClientConfig) -> None:
        """Load existing local correspondences without contacting the KS.

        加载既有本地对应关系，且不联系 KS。
        """
        self.config = config
        self._lock = threading.RLock()
        self._labels_by_record = self._load_store()
        transport = JsonHttpClient(
            config.ks_base_url,
            timeout_seconds=config.timeout_seconds,
            ssl_context=config.ssl_context,
            traffic_recorder=config.traffic_recorder,
        )
        self._oprf_client = OprfClient(transport, batch_size=config.oprf_batch_size)
        self._as_transport = (
            JsonHttpClient(
                config.as_base_url,
                timeout_seconds=config.timeout_seconds,
                ssl_context=config.ssl_context,
                traffic_recorder=config.traffic_recorder,
            )
            if config.as_base_url is not None
            else None
        )
        self._as_heartbeat_transport = (
            JsonHttpClient(
                config.as_base_url,
                timeout_seconds=min(
                    config.timeout_seconds,
                    config.heartbeat_rpc_timeout_seconds,
                ),
                ssl_context=config.ssl_context,
                traffic_recorder=config.traffic_recorder,
            )
            if config.as_base_url is not None
            else None
        )
        # Registration is idempotent before SID assignment: repeating the
        # client identity returns its current session instead of creating a
        # second participant. A short timeout keeps a transient local AS
        # startup race from consuming the general model-RPC timeout.
        # 注册在 SID 分配前具有幂等性：重复提交同一客户端身份会返回当前会话，
        # 而不会创建第二个参与者。较短超时可避免本地 AS 短暂启动竞争耗尽通用
        # 模型 RPC 超时预算。
        self._as_registration_transport = (
            JsonHttpClient(
                config.as_base_url,
                timeout_seconds=min(
                    config.timeout_seconds,
                    config.registration_rpc_timeout_seconds,
                ),
                ssl_context=config.ssl_context,
                traffic_recorder=config.traffic_recorder,
            )
            if config.as_base_url is not None
            else None
        )
        self._as_session: ClientAsSession | None = None
        # A client has both a periodic worker and foreground protocol calls.
        # Serialize heartbeat RPCs per SID so they cannot create duplicate
        # concurrent requests or overwrite one another's instruction snapshot.
        # 一个客户端同时具有周期工作线程与前台协议调用。按 SID 串行化心跳 RPC，
        # 防止同一客户端制造重复并发请求或互相覆盖指令快照。
        self._heartbeat_request_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_heartbeat_error: CommunicationError | None = None
        self._active_round: int | None = None
        self._last_round_instructions: tuple[RoundInstruction, ...] = ()

    @property
    def record_count(self) -> int:
        """Return the number of locally stored record-label associations.

        返回本地存储的数据—标签关联数量。
        """
        with self._lock:
            return len(self._labels_by_record)

    @property
    def as_session(self) -> ClientAsSession | None:
        """Return the current in-memory AS session, if connected.

        返回当前内存中的 AS 会话；未连接时返回空值。
        """
        with self._lock:
            return self._as_session

    @property
    def last_heartbeat_error(self) -> CommunicationError | None:
        """Return the latest background heartbeat error without raising it.

        返回最新后台心跳错误，但不在前台操作中抛出它。
        """
        with self._lock:
            return self._last_heartbeat_error

    @property
    def active_round(self) -> int | None:
        """Return the latest active round announced by an AS heartbeat.

        返回 AS 心跳最近下发的当前轮次。
        """
        with self._lock:
            return self._active_round

    @property
    def last_round_instructions(self) -> tuple[RoundInstruction, ...]:
        """Return the latest AS instructions without exposing mutable state.

        返回最新 AS 指令，且不暴露可变状态。
        """
        with self._lock:
            return self._last_round_instructions

    def connect_to_as(self) -> ClientAsSession:
        """Register this client, receive its SID, and start periodic heartbeats.

        注册此客户端、获取其 SID，并启动周期性心跳。
        """
        if self._as_registration_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        self._stop_heartbeat_worker()
        request = WireMessage.create(AS_REGISTER_REQUEST, {"client_id": self.config.client_id})
        response: WireMessage | None = None
        retry_delay_seconds = AS_REGISTRATION_RETRY_DELAY_SECONDS
        for attempt in range(AS_REGISTRATION_TRANSPORT_ATTEMPTS):
            try:
                response = self._as_registration_transport.send(
                    AggregationServerPath.REGISTER_CLIENT.value,
                    request,
                )
                break
            except TransportError:
                if attempt + 1 >= AS_REGISTRATION_TRANSPORT_ATTEMPTS:
                    raise
                time.sleep(retry_delay_seconds)
                retry_delay_seconds = min(retry_delay_seconds * 2.0, 0.5)
        assert response is not None
        session = self._validated_registration_response(response)
        with self._lock:
            self._as_session = session
            self._last_heartbeat_error = None
        self._start_heartbeat_worker(session.heartbeat_interval_seconds)
        return session

    def send_as_heartbeat(self) -> ClientAsSession:
        """Synchronously refresh the AS timer for the registered SID.

        为已注册 SID 同步刷新 AS 计时器。
        """
        session, _instructions = self.send_as_heartbeat_with_instructions()
        return session

    def send_as_heartbeat_with_instructions(
        self,
    ) -> tuple[ClientAsSession, tuple[RoundInstruction, ...]]:
        """Refresh an SID and return the immutable instructions from this response.

        刷新 SID，并返回本次响应中的不可变指令快照。

        The periodic heartbeat worker may finish another request immediately
        before or after this call. Recovery code must therefore use the
        returned snapshot rather than rereading ``last_round_instructions``;
        the latter is deliberately only a latest-state convenience view.
        周期性心跳线程可能恰好在本调用前后完成另一请求。因此恢复逻辑必须使用
        本函数返回的快照，而不能重新读取 ``last_round_instructions``；后者仅是
        为方便读取而提供的“最新状态”视图。
        """
        if self._as_heartbeat_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        with self._heartbeat_request_lock:
            with self._lock:
                session = self._as_session
            if session is None:
                raise RuntimeError("client is not registered with AS / 客户端尚未注册到 AS")
            request = WireMessage.create(AS_HEARTBEAT_REQUEST, {"sid": session.sid})
            response = self._as_heartbeat_transport.send(
                AggregationServerPath.HEARTBEAT.value,
                request,
            )
            refreshed_session, active_round, instructions = self._validated_heartbeat_response(
                response,
                session.sid,
            )
            with self._lock:
                self._as_session = refreshed_session
                self._last_heartbeat_error = None
                self._active_round = active_round
                self._last_round_instructions = instructions
            return refreshed_session, instructions

    def route_round_instructions(self) -> LocalTrainingQueues:
        """Route the latest heartbeat work without treating DEDUP data as input.

        路由最新心跳工作，且不将 DEDUP 数据作为训练输入。
        """
        with self._lock:
            instructions = self._last_round_instructions
        decisions = tuple(
            TaskClaimDecision(
                protected_label=instruction.protected_label,
                task_id=instruction.task_id,
                operation=instruction.operation,
                state="PENDING" if instruction.operation == "TRAIN" else "EMPTY",
            )
            for instruction in instructions
        )
        return self.route_claim_decisions(decisions)

    def download_global_model_from_as(self, round_id: int, destination: Path) -> Path:
        """Download and SHA-256-verify one AS global checkpoint in chunks.

        分块下载并以 SHA-256 验证一个 AS 全局检查点。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / round_id 必须非负")
        # A real GPT round can take substantially longer than the short lease
        # deliberately used by the training-fault evaluation. Refresh liveness
        # at the read boundary, after all training and aggregation work has
        # completed. 全局模型分发发生在训练和聚合之后；真实 GPT 训练可能远超训练
        # 故障评估中的短租约，因此必须在只读分发边界同步刷新在线状态。
        recovery_deadline = time.monotonic() + self.config.timeout_seconds
        session = self._refresh_global_model_read_session(recovery_deadline)
        next_liveness_refresh = (
            time.monotonic() + self._global_model_liveness_refresh_period(session)
        )
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor: tuple[int, str] | None = None
        offset = 0
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                while True:
                    # Keep a long, chunked model read inside its advertised AS
                    # lease. Unlike model upload, this is an immutable read
                    # guarded by the final SHA-256, so reconnecting and reading
                    # the same offset again cannot commit stale training work.
                    # 长时间分块读取期间保持 AS 租约。与模型上传不同，此路径是受最终
                    # SHA-256 保护的不可变读取；重连后重读同一偏移不会提交陈旧训练结果。
                    if time.monotonic() >= next_liveness_refresh:
                        session = self._refresh_global_model_read_session(recovery_deadline)
                        next_liveness_refresh = (
                            time.monotonic() + self._global_model_liveness_refresh_period(session)
                        )
                    request = WireMessage.create(
                        AS_GLOBAL_MODEL_CHUNK_REQUEST,
                        {
                            "sid": session.sid,
                            "round": round_id,
                            "offset": offset,
                            "max_bytes": self.config.model_chunk_bytes,
                        },
                    )
                    try:
                        response = self._send_global_model_chunk_with_retry(request)
                    except RemoteServiceError as error:
                        if not self._is_offline_sid_error(error):
                            raise
                        # The AS rejected before returning any bytes. Reconnect
                        # with the stable client identity and rebuild precisely
                        # the same immutable read request on the next loop.
                        # AS 在返回任何字节前拒绝了请求；使用稳定客户端身份重连，并在
                        # 下一轮循环中重建完全相同的不可变读取请求。
                        session = self._reconnect_global_model_reader(error, recovery_deadline)
                        next_liveness_refresh = (
                            time.monotonic() + self._global_model_liveness_refresh_period(session)
                        )
                        continue
                    total_bytes, sha256, chunk, complete = self._validated_global_model_chunk(
                        response,
                        round_id=round_id,
                        offset=offset,
                    )
                    if descriptor is None:
                        descriptor = (total_bytes, sha256)
                    elif descriptor != (total_bytes, sha256):
                        raise TransportError("global model changed during download / 下载期间全局模型发生变化")
                    stream.write(chunk)
                    offset += len(chunk)
                    if complete:
                        break
                    if not chunk:
                        raise TransportError("global model download made no progress / 全局模型下载没有进展")
            if (
                descriptor is None
                or offset != descriptor[0]
                or sha256_file(Path(temporary_name)) != descriptor[1]
            ):
                raise TransportError("global model digest verification failed / 全局模型摘要校验失败")
            Path(temporary_name).replace(destination)
            return destination
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def _refresh_global_model_read_session(self, recovery_deadline: float) -> ClientAsSession:
        """Synchronously refresh or safely restore a read-only model session.

        为只读模型会话同步刷新在线状态，或安全恢复该会话。

        This helper is intentionally restricted to immutable global-model
        distribution. It must never be reused by the stateful checkpoint-upload
        state machine, whose recovery has to revalidate task ownership first.
        本辅助方法刻意仅限于不可变全局模型分发；不得复用于有状态检查点上传状态机，
        后者的恢复必须先重新验证任务所有权。
        """
        try:
            return self.send_as_heartbeat()
        except RemoteServiceError as error:
            if not self._is_offline_sid_error(error):
                raise
            return self._reconnect_global_model_reader(error, recovery_deadline)

    def _reconnect_global_model_reader(
        self,
        offline_error: RemoteServiceError,
        recovery_deadline: float,
    ) -> ClientAsSession:
        """Reconnect an expired SID while a SHA-256-verified model read is active.

        在 SHA-256 校验的模型读取期间重连已过期 SID。
        """
        if time.monotonic() >= recovery_deadline:
            raise offline_error
        return self.connect_to_as()

    @staticmethod
    def _is_offline_sid_error(error: RemoteServiceError) -> bool:
        """Identify the explicit AS precondition used for reconnect recovery.

        识别 AS 用于重连恢复的显式前置条件错误。
        """
        return error.status_code == 409 and error.code == "sid_offline"

    @staticmethod
    def _global_model_liveness_refresh_period(session: ClientAsSession) -> float:
        """Use half the advertised heartbeat period, with a small safe floor.

        使用公布心跳周期的一半，并设置很小的安全下限。
        """
        return max(0.01, session.heartbeat_interval_seconds / 2.0)

    def configure_federated_round_at_as(
        self,
        round_id: int,
        participant_sids: Sequence[int],
    ) -> tuple[int, ...]:
        """Freeze the explicit FedAvg roster before any client uploads.

        在任一客户端上传前固定显式 FedAvg 名册。

        The evaluator uses this public method instead of reaching into the
        transport implementation, so a real multi-round experiment exercises
        the same client--AS wire contract as deployment code. 评估器通过此公开
        方法而非访问内部传输对象，使真实多轮实验复用部署代码的客户端--AS 报文契约。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / 轮次必须非负")
        normalized_sids = _validated_sid_sequence(participant_sids)
        response = self._as_transport.send(
            AggregationServerPath.CONFIGURE_ROUND.value,
            WireMessage.create(
                AS_CONFIGURE_ROUND_REQUEST,
                {"round": round_id, "participant_sids": list(normalized_sids)},
            ),
        )
        if response.message_type != AS_CONFIGURE_ROUND_RESPONSE:
            raise TransportError("AS returned an unexpected round configuration response / "
                                 "AS 返回了意外的轮次配置响应")
        payload = dict(response.payload)
        if payload != {"round": round_id, "participant_sids": list(normalized_sids)}:
            raise TransportError("AS round configuration acknowledgement does not match / "
                                 "AS 轮次配置确认不匹配")
        return normalized_sids

    def aggregate_federated_round_at_as(
        self,
        round_id: int,
        participant_sids: Sequence[int],
    ) -> dict[str, object]:
        """Request FedAvg for the previously frozen complete roster.

        请求对已固定且完整的名册执行 FedAvg。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / 轮次必须非负")
        normalized_sids = _validated_sid_sequence(participant_sids)
        response = self._as_transport.send(
            AggregationServerPath.AGGREGATE_MODEL_UPDATES.value,
            WireMessage.create(
                AS_MODEL_AGGREGATE_REQUEST,
                {"round": round_id, "expected_sids": list(normalized_sids)},
            ),
        )
        if response.message_type != AS_MODEL_AGGREGATE_RESPONSE:
            raise TransportError("AS returned an unexpected FedAvg response / "
                                 "AS 返回了意外的 FedAvg 响应")
        payload = dict(response.payload)
        expected_fields = {"round", "participant_sids", "total_samples", "global_sha256", "global_bytes"}
        if (
            set(payload) != expected_fields
            or payload["round"] != round_id
            or payload["participant_sids"] != list(normalized_sids)
            or isinstance(payload["total_samples"], bool)
            or not isinstance(payload["total_samples"], int)
            or payload["total_samples"] < 1
            or isinstance(payload["global_bytes"], bool)
            or not isinstance(payload["global_bytes"], int)
            or payload["global_bytes"] < 1
            or not isinstance(payload["global_sha256"], str)
            or len(payload["global_sha256"]) != 64
        ):
            raise TransportError("AS FedAvg acknowledgement is invalid / AS FedAvg 确认无效")
        return payload

    def register_records_with_as(
        self,
        records: Sequence[str | bytes],
        *,
        created_round: int = 0,
        refresh_lease: bool = True,
    ) -> list[LabelRegistration]:
        """Generate labels, then register the complete set at AS.

        生成标签，然后在 AS 登记完整集合。
        """
        protected_labels = self.generate_protected_labels(records)
        if refresh_lease:
            # Real OPRF batches can take longer than an AS heartbeat interval
            # on a CPU-bound client. Refresh synchronously before the AS-only
            # registration phase so a delayed background thread cannot leave
            # the SID offline. 真实 OPRF 分批在 CPU 密集型客户端上可能超过 AS
            # 心跳周期；在仅属于 AS 的登记阶段前同步刷新，避免后台线程延迟而使
            # SID 被标记为离线。
            self.send_as_heartbeat()
        unique_labels = tuple(dict.fromkeys(protected_labels))
        registrations = self.register_protected_labels_with_as(
            unique_labels,
            created_round=created_round,
        )
        registrations_by_label = {
            registration.protected_label: registration
            for registration in registrations
        }
        return [registrations_by_label[label] for label in protected_labels]

    def register_protected_labels_with_as(
        self,
        protected_labels: Sequence[str],
        *,
        created_round: int = 0,
    ) -> list[LabelRegistration]:
        """Register one canonical OPRF label set without claiming training.

        登记一个规范 OPRF 标签集合，但不抢占训练权。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS / 客户端尚未注册到 AS")
        labels = self._validated_submission_labels(protected_labels)
        if isinstance(created_round, bool) or not isinstance(created_round, int):
            raise ValueError("created_round must fit uint32 / created_round 必须适配 uint32")
        if not 0 <= created_round <= 0xFFFFFFFF:
            raise ValueError("created_round must fit uint32 / created_round 必须适配 uint32")
        response, response_sid = self._send_as_request_with_offline_reconnect(
            AggregationServerPath.REGISTER_LABELS.value,
            lambda active_sid: WireMessage.create(
                AS_REGISTER_LABELS_REQUEST,
                {"sid": active_sid, "round": created_round,
                 "protected_labels": list(labels)},
            ),
        )
        return self._validated_label_registration_response(
            response, expected_sid=response_sid, expected_round=created_round,
            expected_labels=labels,
        )

    def _send_as_request_with_offline_reconnect(
        self,
        path: str,
        request_for_sid: Callable[[int], WireMessage],
    ) -> tuple[WireMessage, int]:
        """Send an AS request, reconnecting only after AS reports an offline SID.

        发送一项 AS 请求；只有 AS 明确报告 SID 离线时才重连。调用方仅可将此方法
        用于 AS 在任何状态变更前拒绝的幂等请求，例如标签登记和 CAS 抢占。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS / 客户端尚未注册到 AS")

        # Use the caller-configured RPC window instead of an arbitrary retry
        # count. A finite deadline prevents a permanently unavailable AS from
        # blocking an evaluation and its GPU workers forever. 使用调用方配置的
        # RPC 时间窗口而非任意重试次数；有限截止时间可避免永久不可用的 AS 无限占用
        # 评估进程和 GPU 工作线程。
        recovery_deadline = time.monotonic() + self.config.timeout_seconds
        active_session = session
        retry_index = 0
        last_offline_error: RemoteServiceError | None = None
        while True:
            try:
                response = self._as_transport.send(
                    path,
                    request_for_sid(active_session.sid),
                )
                return response, active_session.sid
            except RemoteServiceError as error:
                if error.status_code != 409 or error.code != "sid_offline":
                    raise
                last_offline_error = error
                if time.monotonic() >= recovery_deadline:
                    raise

            # AS has explicitly required reconnecting. Registering the same client
            # ID restores its stable SID and starts a fresh heartbeat worker. This
            # helper is used only where AS rejects before mutation, so each retry
            # cannot duplicate an index or CAS transition. AS 已明确要求重连；
            # 使用相同 client ID 注册会恢复稳定 SID 并启动新的心跳线程。该辅助方法仅用于
            # AS 在状态修改前拒绝的路径，因此每次重试都不会重复索引或 CAS 状态转换。
            retry_index += 1
            remaining_seconds = recovery_deadline - time.monotonic()
            if remaining_seconds <= 0:
                assert last_offline_error is not None
                raise last_offline_error
            delay_seconds = min(
                AS_OFFLINE_RECOVERY_INITIAL_DELAY_SECONDS * (2 ** min(retry_index - 1, 4)),
                AS_OFFLINE_RECOVERY_MAX_DELAY_SECONDS,
                remaining_seconds,
            )
            time.sleep(delay_seconds)
            active_session = self.connect_to_as()

    def claim_registered_labels_at_as(
        self,
        registrations: Sequence[LabelRegistration],
        *,
        refresh_lease: bool = True,
    ) -> list[TaskClaimDecision]:
        """Run the later CAS phase for labels acknowledged by registration.

        为已完成登记确认的标签运行后续 CAS 阶段。
        """
        if isinstance(registrations, (str, bytes)) or not isinstance(
            registrations,
            Sequence,
        ):
            raise TypeError("registrations must be a sequence / registrations 必须是一个序列")
        registrations_tuple = tuple(registrations)
        if not registrations_tuple:
            return []
        if any(
            not isinstance(registration, LabelRegistration)
            for registration in registrations_tuple
        ):
            raise TypeError(
                "registrations must contain LabelRegistration values / "
                "registrations 必须包含 LabelRegistration 值"
            )
        unique_labels = tuple(
            dict.fromkeys(registration.protected_label for registration in registrations_tuple)
        )
        decisions = self.claim_protected_labels_at_as(
            unique_labels,
            refresh_lease=refresh_lease,
        )
        decisions_by_label = {decision.protected_label: decision for decision in decisions}
        return [
            decisions_by_label[registration.protected_label]
            for registration in registrations_tuple
        ]

    def claim_protected_labels_at_as(
        self,
        protected_labels: Sequence[str],
        *,
        refresh_lease: bool = True,
    ) -> list[TaskClaimDecision]:
        """Claim already registered labels without modifying index ownership.

        抢占已登记标签，但不修改索引所有权。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS / 客户端尚未注册到 AS")
        labels = self._validated_submission_labels(protected_labels)
        if refresh_lease:
            # Registration can involve many native-index writes. Refresh
            # immediately before the later independent CAS phase so
            # server-queue delay is measured from a current lease. 标签登记可能
            # 包含大量原生索引写入；在后续独立 CAS 阶段前立即刷新，使服务端排队延迟从
            # 当前租约开始计算。
            self.send_as_heartbeat()
        response, response_sid = self._send_as_request_with_offline_reconnect(
            AggregationServerPath.CLAIM_TASK.value,
            lambda active_sid: WireMessage.create(
                AS_CLAIM_TASKS_REQUEST,
                {"sid": active_sid, "protected_labels": list(labels)},
            ),
        )
        return self._validated_task_claim_response(
            response, expected_sid=response_sid, expected_labels=labels,
        )

    def route_claim_decisions(
        self,
        decisions: Sequence[TaskClaimDecision],
    ) -> LocalTrainingQueues:
        """Route locally retained records into paper hot and cold queues.

        将本地保留记录路由到论文中的热队列和冷队列。
        """
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("decisions must be a sequence / decisions 必须是一个序列")
        with self._lock:
            records_by_label = {
                label: base64.b64decode(record_key.encode("ascii"), validate=True)
                for record_key, label in self._labels_by_record.items()
            }
        hot_records: list[bytes] = []
        cold_records: list[bytes] = []
        for decision in decisions:
            if not isinstance(decision, TaskClaimDecision):
                raise TypeError(
                    "decisions must contain TaskClaimDecision / "
                    "decisions 必须包含 TaskClaimDecision"
                )
            record = records_by_label.get(decision.protected_label)
            if record is None:
                raise ClientLabelStoreError(
                    "AS decision has no local record correspondence / AS 决策没有本地数据对应关系"
                )
            if decision.operation == "TRAIN":
                hot_records.append(record)
            elif decision.operation == "DEDUP":
                cold_records.append(record)
            else:
                raise ValueError("decision operation is invalid / 决策操作标识无效")
        return LocalTrainingQueues(tuple(hot_records), tuple(cold_records))

    def reconcile_recovery_claims(
        self,
        decisions: Sequence[TaskClaimDecision],
        *,
        forced_dedup_labels: Sequence[str] = (),
    ) -> list[TaskClaimDecision]:
        """Merge a fresh CAS result with this SID's heartbeat ownership proof.

        将最新 CAS 结果与该 SID 的心跳所有权证明合并。

        CAS correctly returns DEDUP for an already PENDING task, including one
        still owned by this SID. During recovery, a same-SID TRAIN heartbeat
        instruction is therefore authoritative for retaining that hot item.
        Labels explicitly supplied by AS as transferred always remain DEDUP.
        对已经 PENDING 的任务，CAS 会正确返回 DEDUP，即使它仍由当前 SID 拥有。
        因此恢复期间，同 SID 的 TRAIN 心跳指令是保留热队列项的权威依据；AS 明确
        标记为已转移的标签始终保持 DEDUP。
        """
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("decisions must be a sequence / decisions 必须是一个序列")
        forced_labels = set(self._validated_submission_labels(forced_dedup_labels)) \
            if forced_dedup_labels else set()
        with self._lock:
            instructions_by_label = {
                instruction.protected_label: instruction
                for instruction in self._last_round_instructions
            }
        reconciled: list[TaskClaimDecision] = []
        for decision in decisions:
            if not isinstance(decision, TaskClaimDecision):
                raise TypeError(
                    "decisions must contain TaskClaimDecision / "
                    "decisions 必须包含 TaskClaimDecision"
                )
            instruction = instructions_by_label.get(decision.protected_label)
            owns_pending_task = (
                instruction is not None
                and instruction.task_id == decision.task_id
                and instruction.operation == "TRAIN"
            )
            if decision.protected_label not in forced_labels and (
                decision.operation == "TRAIN" or owns_pending_task
            ):
                reconciled.append(
                    TaskClaimDecision(
                        decision.protected_label,
                        decision.task_id,
                        "TRAIN",
                        "PENDING",
                    )
                )
            else:
                reconciled.append(decision)
        return reconciled

    def submit_model_update_at_as(
        self,
        checkpoint_path: Path,
        decisions: Sequence[TaskClaimDecision],
        *,
        round_id: int,
        sample_count: int,
    ) -> ModelUpdateSubmission:
        """Upload a trained checkpoint and commit this SID's TRAIN tasks.

        上传训练后的检查点，并提交此 SID 的 TRAIN 任务。

        The file is sent in bounded base64 chunks because the existing DwT-FL
        protocol is JSON-only.  AS verifies a SHA-256 digest before it marks
        the associated PENDING tasks COMMITTED.  由于既有 DwT-FL 协议仅使用 JSON，
        文件以有界 Base64 分块发送；AS 在将对应 PENDING 任务标记为 COMMITTED 前会
        验证 SHA-256 摘要。 ``sample_count`` remains in the client message for
        compatibility and diagnostics, but AS derives FedAvg weights only from
        actual COMMITTED state-table tasks. ``sample_count`` 保留在客户端报文中
        用于兼容和诊断，但 AS 仅以状态表中的实际 COMMITTED 任务数作为 FedAvg 权重。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / round_id 必须非负")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("sample_count must be positive / sample_count 必须为正数")
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 1:
            raise FileNotFoundError("checkpoint must be a non-empty file / 检查点必须为非空文件")
        train_decisions = tuple(decision for decision in decisions if decision.operation == "TRAIN")
        task_ids = self._train_task_ids(train_decisions)
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS / 客户端尚未注册到 AS")
        update_id = uuid.uuid4().hex
        total_bytes = checkpoint_path.stat().st_size
        checkpoint_sha256 = self._hash_checkpoint_with_heartbeats(checkpoint_path)
        # Refresh again immediately before the stateful upload phase. 再次在有状态的
        # 上传阶段前立即刷新。
        session = self.send_as_heartbeat()
        recovery_deadline: float | None = None
        recovery_index = 0
        while True:
            try:
                for offset, chunk in chunk_file(checkpoint_path, self.config.model_chunk_bytes):
                    request = WireMessage.create(
                        AS_MODEL_CHUNK_REQUEST,
                        {
                            "sid": session.sid,
                            "round": round_id,
                            "update_id": update_id,
                            "total_bytes": total_bytes,
                            "sha256": checkpoint_sha256,
                            "offset": offset,
                            "chunk_base64": base64.b64encode(chunk).decode("ascii"),
                        },
                    )
                    response = self._send_model_chunk_with_retry(request)
                    self._validated_model_chunk_response(
                        response,
                        sid=session.sid,
                        round_id=round_id,
                        update_id=update_id,
                        expected_received_bytes=offset + len(chunk),
                        total_bytes=total_bytes,
                    )
                request = WireMessage.create(
                    AS_MODEL_FINALIZE_REQUEST,
                    {
                        "sid": session.sid,
                        "round": round_id,
                        "update_id": update_id,
                        "total_bytes": total_bytes,
                        "sha256": checkpoint_sha256,
                        "sample_count": sample_count,
                        "task_ids": list(task_ids),
                    },
                )
                response = self._as_transport.send(
                    AggregationServerPath.SUBMIT_MODEL_UPDATE.value,
                    request,
                )
                return self._validated_model_finalize_response(
                    response,
                    sid=session.sid,
                    round_id=round_id,
                    update_id=update_id,
                    task_ids=task_ids,
                    checkpoint_sha256=checkpoint_sha256,
                )
            except RemoteServiceError as error:
                # Both codes are returned before AS commits this model update. The
                # latter means a lease release was observed at finalization, so the
                # same ownership proof is required before any replay. 两种状态码都在
                # AS 提交该模型更新前返回；后者表示完成阶段观察到租约释放，因此在任何
                # 重放前同样必须证明当前所有权。
                if error.status_code != 409 or error.code not in {
                    "sid_offline",
                    "task_not_pending_for_sid",
                }:
                    raise
                if error.code == "task_not_pending_for_sid":
                    dedup_instructions = self._ownership_loss_instructions_from_error(error)
                    if dedup_instructions:
                        # A structured AS rejection proves another online
                        # trainer already owns at least one plaintext-equivalent
                        # record. Do not replay the stale checkpoint. 一个结构化
                        # AS 拒绝证明另一在线训练者已拥有至少一条明文等价记录；不得重放
                        # 陈旧检查点。
                        raise ModelUpdateOwnershipLostError(
                            "AS rejected the stale model update after task takeover / "
                            "AS 在任务接管后拒绝陈旧模型更新",
                            dedup_instructions=dedup_instructions,
                        ) from error
                if recovery_deadline is None:
                    recovery_deadline = time.monotonic() + self.config.timeout_seconds
                remaining_seconds = recovery_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise

            # AS may reject before a chunk write or before final task commit, and
            # its timeout monitor can already have released PENDING tasks. Reconnect
            # and repeat the paper's separate CAS phase before replaying exact chunks.
            # AS 可能在写入分块前或最终提交任务前拒绝请求，其超时监控可能已释放 PENDING
            # 任务；因此必须先重连并再次执行论文中独立的 CAS 阶段，随后才重放完全相同的分块。
            recovery_index += 1
            delay_seconds = min(
                AS_OFFLINE_RECOVERY_INITIAL_DELAY_SECONDS * (2 ** min(recovery_index - 1, 4)),
                AS_OFFLINE_RECOVERY_MAX_DELAY_SECONDS,
                remaining_seconds,
            )
            time.sleep(delay_seconds)
            session = self._recover_model_upload_ownership(train_decisions)

    def _recover_model_upload_ownership(
        self,
        train_decisions: Sequence[TaskClaimDecision],
    ) -> ClientAsSession:
        """Reconnect and re-claim every task before replaying a rejected upload.

        在重放被拒绝的上传前重连并重新抢占每一个任务。
        """
        expected_tasks = {decision.protected_label: decision.task_id for decision in train_decisions}
        if not expected_tasks or len(expected_tasks) != len(train_decisions):
            raise ValueError(
                "TRAIN decisions must have unique protected labels / "
                "TRAIN 决策必须具有唯一受保护标签"
            )
        session = self.connect_to_as()
        recovered = self.claim_protected_labels_at_as(tuple(expected_tasks))
        recovered_by_label = {decision.protected_label: decision for decision in recovered}
        # The AS can atomically claim a released task for this same SID while it
        # answers the synchronous heartbeat that precedes CAS. In that case the
        # following explicit CAS correctly reports DEDUP/PENDING, while the
        # heartbeat instruction is the proof that this SID owns the task. AS 可能在
        # CAS 前的同步心跳响应中为同一 SID 原子抢占已释放任务；此时随后的显式 CAS 会
        # 正确返回 DEDUP/PENDING，而心跳指令证明该 SID 拥有该任务。
        with self._lock:
            instructions_by_label = {
                instruction.protected_label: instruction
                for instruction in self._last_round_instructions
            }
        retained = (
            len(recovered_by_label) == len(expected_tasks)
            and all(
                recovered_by_label.get(label) is not None
                and recovered_by_label[label].task_id == task_id
                and recovered_by_label[label].state == "PENDING"
                and (
                    recovered_by_label[label].operation == "TRAIN"
                    or (
                        instructions_by_label.get(label) is not None
                        and instructions_by_label[label].task_id == task_id
                        and instructions_by_label[label].operation == "TRAIN"
                    )
                )
                for label, task_id in expected_tasks.items()
            )
        )
        if not retained:
            # A different live client can validly win recovery CAS. Its model must
            # be trained from its own current hot queue, not reused from this stale
            # checkpoint. 另一个在线客户端可以合法赢得恢复 CAS；它必须从自己的当前热
            # 队列训练模型，而不得复用这个陈旧检查点。
            raise ModelUpdateOwnershipLostError(
                "model upload ownership was transferred during recovery / "
                "模型上传所有权已在恢复期间转移",
                recovered_decisions=tuple(recovered),
            )
        return session

    @staticmethod
    def _ownership_loss_instructions_from_error(
        error: RemoteServiceError,
    ) -> tuple[TaskClaimDecision, ...]:
        """Parse AS's 409 DEDUP instructions before any local retraining.

        在任何本地重训前解析 AS 的 409 DEDUP 指令。
        """
        raw_instructions = error.payload.get("dedup_instructions")
        if raw_instructions is None:
            return ()
        if not isinstance(raw_instructions, list):
            raise TransportError(
                "AS ownership-loss payload is invalid / AS 所有权丢失载荷无效"
            )
        parsed: list[TaskClaimDecision] = []
        for instruction in raw_instructions:
            if not isinstance(instruction, Mapping) or set(instruction) != {
                "protected_label", "task_id", "operation", "state"
            }:
                raise TransportError(
                    "AS ownership-loss instruction is invalid / "
                    "AS 所有权丢失指令无效"
                )
            protected_label = instruction["protected_label"]
            task_id = instruction["task_id"]
            operation = instruction["operation"]
            state = instruction["state"]
            if (
                not isinstance(protected_label, str)
                or isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id < 0
                or operation != "DEDUP"
                or state not in {"PENDING", "COMMITTED"}
            ):
                raise TransportError(
                    "AS ownership-loss instruction fields are invalid / "
                    "AS 所有权丢失指令字段无效"
                )
            try:
                validate_protected_label(protected_label)
            except OprfValidationError as validation_error:
                raise TransportError(
                    "AS ownership-loss label is invalid / AS 所有权丢失标签无效"
                ) from validation_error
            parsed.append(TaskClaimDecision(protected_label, task_id, operation, state))
        if len({item.protected_label for item in parsed}) != len(parsed):
            raise TransportError(
                "AS ownership-loss labels must be unique / "
                "AS 所有权丢失标签必须唯一"
            )
        return tuple(parsed)

    def _hash_checkpoint_with_heartbeats(self, checkpoint_path: Path) -> str:
        """Hash one checkpoint while retaining the AS liveness lease.

        在保留 AS 在线租约的同时计算检查点摘要。完整 Safetensors 文件的本地 I/O
        可能远长于一个心跳周期；在摘要循环中刷新心跳不会改变输入字节或 SHA-256 结果。
        """
        session = self.send_as_heartbeat()
        heartbeat_period = max(0.1, session.heartbeat_interval_seconds / 2)
        next_heartbeat = time.monotonic() + heartbeat_period
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as stream:
            while chunk := stream.read(MODEL_HASH_READ_BYTES):
                digest.update(chunk)
                now = time.monotonic()
                if now >= next_heartbeat:
                    session = self.send_as_heartbeat()
                    heartbeat_period = max(0.1, session.heartbeat_interval_seconds / 2)
                    next_heartbeat = now + heartbeat_period
        return digest.hexdigest()

    def _send_model_chunk_with_retry(self, request: WireMessage) -> WireMessage:
        """Send one idempotent upload chunk with bounded transport recovery.

        对一个幂等上传分块执行有界传输恢复。仅重试没有收到 HTTP 响应的传输错误；
        同一请求保留原始请求标识，AS 会校验并确认完全相同的已写入分块。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        for attempt in range(MODEL_CHUNK_TRANSPORT_ATTEMPTS):
            try:
                return self._as_transport.send(
                    AggregationServerPath.SUBMIT_MODEL_UPDATE.value,
                    request,
                )
            except TransportError:
                if attempt + 1 == MODEL_CHUNK_TRANSPORT_ATTEMPTS:
                    raise
                time.sleep(MODEL_CHUNK_RETRY_DELAY_SECONDS * (attempt + 1))
        raise AssertionError("bounded model-chunk retry loop must return or raise")

    def _send_global_model_chunk_with_retry(self, request: WireMessage) -> WireMessage:
        """Read one immutable global-model chunk with bounded transport recovery.

        The AS does not mutate state for this endpoint: the request identifies
        one round, offset, and maximum byte count, and the client subsequently
        verifies the final SHA-256.  Retrying only a transport failure is thus
        safe and avoids turning a temporary local accept-backlog spike into a
        failed experiment case. AS 对该端点不改变状态：请求只标识轮次、偏移和最大
        字节数，客户端随后会验证最终 SHA-256。因此仅重试传输失败是安全的，并可避免
        将临时本地接收队列峰值变成失败的实验用例。
        """
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured / 尚未配置 AS 端点")
        for attempt in range(GLOBAL_MODEL_CHUNK_TRANSPORT_ATTEMPTS):
            try:
                return self._as_transport.send(
                    AggregationServerPath.DOWNLOAD_GLOBAL_MODEL.value,
                    request,
                )
            except TransportError:
                if attempt + 1 == GLOBAL_MODEL_CHUNK_TRANSPORT_ATTEMPTS:
                    raise
                time.sleep(GLOBAL_MODEL_CHUNK_RETRY_DELAY_SECONDS * (attempt + 1))
        raise AssertionError("bounded global-model retry loop must return or raise")

    def protected_label_for(self, record: str | bytes) -> str | None:
        """Return a locally known protected label without making a network call.

        返回本地已知受保护标签，且不发起网络调用。
        """
        record_key = self._record_key(normalize_record(record))
        with self._lock:
            return self._labels_by_record.get(record_key)

    def generate_protected_labels(self, records: Sequence[str | bytes]) -> list[str]:
        """Generate labels through KS and atomically persist new correspondences.

        经由 KS 生成标签，并原子持久化新的对应关系。
        """
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError(
                "records must be a sequence of str or bytes / "
                "records 必须为 str 或 bytes 序列"
            )
        normalized_records = [normalize_record(record) for record in records]
        if not normalized_records:
            return []

        missing_records: dict[str, bytes] = {}
        with self._lock:
            for record in normalized_records:
                record_key = self._record_key(record)
                if record_key not in self._labels_by_record:
                    missing_records.setdefault(record_key, record)

        generated_labels: dict[str, str] = {}
        if missing_records:
            generated = self._oprf_client.evaluate(list(missing_records.values()))
            generated_labels = dict(zip(missing_records, generated, strict=True))

        with self._lock:
            changed = False
            for record_key, protected_label in generated_labels.items():
                existing_label = self._labels_by_record.get(record_key)
                if existing_label is None:
                    self._labels_by_record[record_key] = protected_label
                    changed = True
                elif existing_label != protected_label:
                    raise ClientLabelStoreError(
                        "KS returned a label inconsistent with local OPRF state / "
                        "KS 返回的标签与本地 OPRF 状态不一致"
                    )
            if changed:
                self._persist_store()
            return [
                self._labels_by_record[self._record_key(record)]
                for record in normalized_records
            ]

    def close(self) -> None:
        """Stop the background heartbeat worker without altering local mappings.

        停止后台心跳线程，不修改本地映射。
        """
        self._stop_heartbeat_worker()

    def __enter__(self) -> "ClientEntity":
        """Enter a managed client lifetime. / 进入受管理的客户端生命周期。"""
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Stop heartbeats at managed-lifetime exit. / 在受管理生命周期退出时停止心跳。"""
        self.close()

    def _load_store(self) -> dict[str, str]:
        """Load a strict, client-bound mapping without creating an empty file.

        加载严格且绑定客户端的映射，不创建空文件。
        """
        path = self.config.label_store_path
        if not path.exists():
            return {}
        if path.is_symlink():
            raise ClientLabelStoreError(
                "label-store path must not be a symlink / 标签存储路径不得为符号链接"
            )
        try:
            serialized = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClientLabelStoreError("cannot read label store / 无法读取标签存储") from error
        if not isinstance(serialized, dict) or set(serialized) != {
            "schema_version",
            "oprf_suite",
            "client_id",
            "entries",
        }:
            raise ClientLabelStoreError("invalid label-store schema / 标签存储模式无效")
        if serialized["schema_version"] != LABEL_STORE_SCHEMA_VERSION:
            raise ClientLabelStoreError(
                "unsupported label-store version; create a fresh Ristretto255 cache instead of mixing legacy labels / "
                "不支持的标签存储版本；请创建新的 Ristretto255 缓存，不得混用旧标签"
            )
        if serialized["oprf_suite"] != self.config.oprf_suite:
            raise ClientLabelStoreError(
                "label store uses another OPRF suite / 标签存储使用另一 OPRF 套件"
            )
        if serialized["client_id"] != self.config.client_id:
            raise ClientLabelStoreError(
                "label store belongs to another client / 标签存储属于另一客户端"
            )
        entries = serialized["entries"]
        if not isinstance(entries, list):
            raise ClientLabelStoreError("label-store entries must be an array / 标签存储条目必须为数组")
        loaded: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"data_b64", "protected_label"}:
                raise ClientLabelStoreError("invalid label-store entry / 标签存储条目无效")
            encoded_record = entry["data_b64"]
            protected_label = entry["protected_label"]
            if not isinstance(encoded_record, str) or not isinstance(protected_label, str):
                raise ClientLabelStoreError("label-store value has invalid type / 标签存储值类型无效")
            try:
                record = base64.b64decode(encoded_record.encode("ascii"), validate=True)
                normalized_record = normalize_record(record)
                validate_protected_label(protected_label)
            except (UnicodeEncodeError, ValueError, OprfValidationError) as error:
                raise ClientLabelStoreError(
                    "invalid protected-label mapping / 无效的受保护标签映射"
                ) from error
            canonical_key = self._record_key(normalized_record)
            if encoded_record != canonical_key or canonical_key in loaded:
                raise ClientLabelStoreError(
                    "duplicate or non-canonical record mapping / 重复或不规范的数据映射"
                )
            loaded[canonical_key] = protected_label
        return loaded

    def _persist_store(self) -> None:
        """Atomically replace the private mapping after a complete OPRF response.

        在收到完整 OPRF 响应后原子替换私有映射。
        """
        path = self.config.label_store_path
        if path.exists() and path.is_symlink():
            raise ClientLabelStoreError(
                "label-store path must not be a symlink / 标签存储路径不得为符号链接"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            {
                "schema_version": LABEL_STORE_SCHEMA_VERSION,
                "oprf_suite": self.config.oprf_suite,
                "client_id": self.config.client_id,
                "entries": [
                    {"data_b64": record_key, "protected_label": protected_label}
                    for record_key, protected_label in sorted(self._labels_by_record.items())
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        except OSError as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ClientLabelStoreError("cannot persist label store / 无法持久化标签存储") from error

    @staticmethod
    def _validated_submission_labels(protected_labels: Sequence[str]) -> tuple[str, ...]:
        """Validate the client-side set before it reaches the AS boundary.

        在其抵达 AS 边界前验证客户端侧集合。
        """
        if isinstance(protected_labels, (str, bytes)) or not isinstance(
            protected_labels,
            Sequence,
        ):
            raise TypeError(
                "protected_labels must be a sequence of strings / "
                "protected_labels 必须是字符串序列"
            )
        labels = tuple(protected_labels)
        if not labels:
            raise ValueError("protected_labels must not be empty / protected_labels 不得为空")
        if any(not isinstance(label, str) for label in labels):
            raise TypeError(
                "protected_labels must contain only strings / "
                "protected_labels 必须只包含字符串"
            )
        if len(set(labels)) != len(labels):
            raise ValueError("protected_labels must be unique / protected_labels 必须唯一")
        try:
            for label in labels:
                validate_protected_label(label)
        except OprfValidationError as error:
            raise ValueError(
                "protected_labels contain an invalid OPRF label / "
                "受保护标签含无效 OPRF 标签"
            ) from error
        return labels

    def _start_heartbeat_worker(self, interval_seconds: float) -> None:
        """Start the client timer after registration has completed successfully.

        注册成功后启动客户端计时器。
        """
        self._heartbeat_stop = threading.Event()
        # Clients that connect in one protocol cohort must not subsequently
        # create a perfectly phase-aligned 0.1-second control burst. Derive a
        # deterministic per-client phase in one interval: this preserves the
        # configured heartbeat period and reproducibility, while spreading only
        # control-plane arrivals. 同一协议组连接的客户端不能在后续形成完全同相的
        # 0.1 秒控制请求突发。按客户端 ID 在一个周期内确定性分相：既保持配置的
        # 心跳周期与可复现性，又仅分散控制面到达时刻。
        digest = hashlib.sha256(self.config.client_id.encode("utf-8")).digest()
        phase_fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
        initial_delay_seconds = phase_fraction * interval_seconds
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval_seconds, initial_delay_seconds, self._heartbeat_stop),
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat_worker(self) -> None:
        """Stop any prior heartbeat worker before a new connection is created.

        创建新连接前停止已有心跳线程。
        """
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._heartbeat_thread = None

    def _heartbeat_loop(
        self,
        interval_seconds: float,
        initial_delay_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        """Send periodic heartbeats and retain background failures for inspection.

        发送周期性心跳，并保留后台失败供检查。
        """
        if stop_event.wait(initial_delay_seconds):
            return
        while not stop_event.is_set():
            try:
                self.send_as_heartbeat()
            except CommunicationError as error:
                with self._lock:
                    self._last_heartbeat_error = error
            except Exception as error:
                with self._lock:
                    self._last_heartbeat_error = TransportError(
                        "background heartbeat failed / 后台心跳失败"
                    )
            if stop_event.wait(interval_seconds):
                return

    @staticmethod
    def _validated_registration_response(response: WireMessage) -> ClientAsSession:
        """Validate an AS registration response before retaining its SID.

        在保存 SID 前验证 AS 注册响应。
        """
        if response.message_type != AS_REGISTER_RESPONSE:
            raise TransportError("AS returned an invalid registration response / AS 返回了无效注册响应")
        payload = response.payload
        if set(payload) != {"sid", "reused", "heartbeat_interval_seconds"}:
            raise TransportError("AS returned an invalid registration payload / AS 返回了无效注册载荷")
        return ClientEntity._session_from_payload(payload, expected_sid=None)

    @staticmethod
    def _validated_heartbeat_response(
        response: WireMessage,
        sid: int,
    ) -> tuple[ClientAsSession, int, tuple[RoundInstruction, ...]]:
        """Validate a matching successful heartbeat response.

        验证匹配且成功的心跳响应。
        """
        if response.message_type != AS_HEARTBEAT_RESPONSE:
            raise TransportError("AS returned an invalid heartbeat response / AS 返回了无效心跳响应")
        payload = response.payload
        if set(payload) != {
            "sid",
            "online",
            "heartbeat_interval_seconds",
            "round",
            "instructions",
        }:
            raise TransportError("AS returned an invalid heartbeat payload / AS 返回了无效心跳载荷")
        if payload["online"] is not True:
            raise TransportError("AS did not confirm client liveness / AS 未确认客户端存活")
        session = ClientEntity._session_from_payload(payload, expected_sid=sid)
        active_round = payload["round"]
        raw_instructions = payload["instructions"]
        if isinstance(active_round, bool) or not isinstance(active_round, int) or active_round < 0:
            raise TransportError("AS returned an invalid active round / AS 返回了无效当前轮次")
        if not isinstance(raw_instructions, list):
            raise TransportError("AS returned invalid round instructions / AS 返回了无效轮次指令")
        instructions: list[RoundInstruction] = []
        for raw_instruction in raw_instructions:
            if not isinstance(raw_instruction, Mapping) or set(raw_instruction) != {
                "protected_label",
                "task_id",
                "operation",
            }:
                raise TransportError("AS returned an invalid round instruction / AS 返回了无效轮次指令")
            protected_label = raw_instruction["protected_label"]
            task_id = raw_instruction["task_id"]
            operation = raw_instruction["operation"]
            if not isinstance(protected_label, str) or operation not in {"TRAIN", "DEDUP"}:
                raise TransportError("AS round instruction fields are invalid / AS 轮次指令字段无效")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise TransportError("AS round task ID is invalid / AS 轮次任务 ID 无效")
            instructions.append(RoundInstruction(protected_label, task_id, operation))
        return session, active_round, tuple(instructions)

    @staticmethod
    def _session_from_payload(
        payload: Mapping[str, object],
        *,
        expected_sid: int | None,
    ) -> ClientAsSession:
        """Construct one typed session only from a canonical AS payload.

        仅从规范 AS 载荷构造一个类型化会话。
        """
        sid = payload["sid"]
        interval = payload["heartbeat_interval_seconds"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise TransportError("AS returned an invalid SID / AS 返回了无效 SID")
        if expected_sid is not None and sid != expected_sid:
            raise TransportError("AS heartbeat SID does not match / AS 心跳 SID 不匹配")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
            raise TransportError("AS returned an invalid heartbeat interval / AS 返回了无效心跳周期")
        return ClientAsSession(sid=sid, heartbeat_interval_seconds=float(interval))

    @staticmethod
    def _validated_label_registration_response(
        response: WireMessage,
        *,
        expected_sid: int,
        expected_round: int,
        expected_labels: tuple[str, ...],
    ) -> list[LabelRegistration]:
        """Validate every AS index-registration acknowledgement.

        验证每个 AS 索引登记确认。
        """
        if response.message_type != AS_REGISTER_LABELS_RESPONSE:
            raise TransportError("AS returned an invalid label response / AS 返回了无效标签响应")
        payload = response.payload
        if set(payload) != {"sid", "round", "registrations"}:
            raise TransportError("AS returned an invalid label payload / AS 返回了无效标签载荷")
        if payload["sid"] != expected_sid or payload["round"] != expected_round:
            raise TransportError("AS response does not match submission / AS 响应与提交不匹配")
        raw_registrations = payload["registrations"]
        if not isinstance(raw_registrations, list) or len(raw_registrations) != len(
            expected_labels
        ):
            raise TransportError(
                "AS returned an invalid registration count / AS 返回了无效登记数量"
            )
        registrations: list[LabelRegistration] = []
        for expected_label, raw_registration in zip(
            expected_labels,
            raw_registrations,
            strict=True,
        ):
            if not isinstance(raw_registration, Mapping) or set(raw_registration) != {
                "protected_label",
                "task_id",
            }:
                raise TransportError("AS returned an invalid registration / AS 返回了无效登记")
            label = raw_registration["protected_label"]
            task_id = raw_registration["task_id"]
            if label != expected_label:
                raise TransportError("AS registration label is invalid / AS 登记标签无效")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise TransportError("AS registration task_id is invalid / AS 登记 task_id 无效")
            registrations.append(LabelRegistration(label, task_id))
        return registrations

    @staticmethod
    def _validated_task_claim_response(
        response: WireMessage,
        *,
        expected_sid: int,
        expected_labels: tuple[str, ...],
    ) -> list[TaskClaimDecision]:
        """Validate the paper's TRAIN or DEDUP state mapping from the AS.

        验证 AS 返回的论文中 TRAIN 或 DEDUP 状态映射。
        """
        if response.message_type != AS_CLAIM_TASKS_RESPONSE:
            raise TransportError("AS returned an invalid task-claim response / AS 返回了无效任务抢占响应")
        payload = response.payload
        if set(payload) != {"sid", "decisions"} or payload["sid"] != expected_sid:
            raise TransportError("AS returned an invalid task-claim payload / AS 返回了无效任务抢占载荷")
        raw_decisions = payload["decisions"]
        if not isinstance(raw_decisions, list) or len(raw_decisions) != len(expected_labels):
            raise TransportError("AS returned an invalid task-decision count / AS 返回了无效任务决策数量")
        decisions: list[TaskClaimDecision] = []
        for expected_label, raw_decision in zip(expected_labels, raw_decisions, strict=True):
            if not isinstance(raw_decision, Mapping) or set(raw_decision) != {
                "protected_label",
                "task_id",
                "operation",
                "state",
            }:
                raise TransportError("AS returned an invalid task decision / AS 返回了无效任务决策")
            label = raw_decision["protected_label"]
            task_id = raw_decision["task_id"]
            operation = raw_decision["operation"]
            state = raw_decision["state"]
            if label != expected_label or operation not in {"TRAIN", "DEDUP"}:
                raise TransportError("AS task decision is invalid / AS 任务决策无效")
            if state not in {"EMPTY", "PENDING", "COMMITTED"}:
                raise TransportError("AS task state is invalid / AS 任务状态无效")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise TransportError("AS task_id is invalid / AS task_id 无效")
            if operation == "TRAIN" and state != "PENDING":
                raise TransportError("TRAIN decision must be PENDING / TRAIN 决策必须是 PENDING")
            decisions.append(TaskClaimDecision(label, task_id, operation, state))
        return decisions

    @staticmethod
    def _train_task_ids(decisions: Sequence[TaskClaimDecision]) -> tuple[int, ...]:
        """Extract the unique PENDING task identifiers owned by this client.

        提取本客户端拥有的唯一 PENDING 任务标识。
        """
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("decisions must be a sequence / decisions 必须是一个序列")
        task_ids: list[int] = []
        for decision in decisions:
            if not isinstance(decision, TaskClaimDecision):
                raise TypeError(
                    "decisions must contain TaskClaimDecision / "
                    "decisions 必须包含 TaskClaimDecision"
                )
            if decision.operation == "TRAIN":
                if decision.state != "PENDING":
                    raise ValueError("TRAIN task must be PENDING / TRAIN 任务必须为 PENDING")
                task_ids.append(decision.task_id)
            elif decision.operation != "DEDUP":
                raise ValueError("decision operation is invalid / 决策操作标识无效")
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("at least one unique TRAIN task is required / 至少需要一个唯一 TRAIN 任务")
        return tuple(task_ids)

    @staticmethod
    def _validated_model_chunk_response(
        response: WireMessage,
        *,
        sid: int,
        round_id: int,
        update_id: str,
        expected_received_bytes: int,
        total_bytes: int,
    ) -> None:
        """Validate one ordered model-upload chunk acknowledgement.

        验证一个有序模型上传分块确认。
        """
        if response.message_type != AS_MODEL_CHUNK_RESPONSE:
            raise TransportError("AS returned an invalid model-chunk response / AS 返回了无效模型分块响应")
        payload = response.payload
        if set(payload) != {"sid", "round", "update_id", "received_bytes"}:
            raise TransportError("AS returned an invalid model-chunk payload / AS 返回了无效模型分块载荷")
        received_bytes = payload["received_bytes"]
        if (
            payload["sid"] != sid
            or payload["round"] != round_id
            or payload["update_id"] != update_id
            or isinstance(received_bytes, bool)
            or not isinstance(received_bytes, int)
            or not expected_received_bytes <= received_bytes <= total_bytes
        ):
            raise TransportError("AS model-chunk acknowledgement does not match / AS 模型分块确认不匹配")

    @staticmethod
    def _validated_model_finalize_response(
        response: WireMessage,
        *,
        sid: int,
        round_id: int,
        update_id: str,
        task_ids: tuple[int, ...],
        checkpoint_sha256: str,
    ) -> ModelUpdateSubmission:
        """Validate final task commits bound to one checkpoint digest.

        验证绑定到一个检查点摘要的最终任务提交。
        """
        if response.message_type != AS_MODEL_FINALIZE_RESPONSE:
            raise TransportError("AS returned an invalid model-finalize response / AS 返回了无效模型完成响应")
        payload = response.payload
        expected_fields = {
            "sid",
            "round",
            "update_id",
            "committed_task_ids",
            "checkpoint_sha256",
        }
        if set(payload) != expected_fields:
            raise TransportError("AS returned an invalid model-finalize payload / AS 返回了无效模型完成载荷")
        if (
            payload["sid"] != sid
            or payload["round"] != round_id
            or payload["update_id"] != update_id
            or payload["checkpoint_sha256"] != checkpoint_sha256
            or payload["committed_task_ids"] != list(task_ids)
        ):
            raise TransportError("AS model-finalize acknowledgement does not match / AS 模型完成确认不匹配")
        return ModelUpdateSubmission(
            round_id=round_id,
            sid=sid,
            update_id=update_id,
            committed_task_ids=task_ids,
            checkpoint_sha256=checkpoint_sha256,
        )

    def _validated_global_model_chunk(
        self,
        response: WireMessage,
        *,
        round_id: int,
        offset: int,
    ) -> tuple[int, str, bytes, bool]:
        """Validate and decode one bounded AS global-model chunk response.

        验证并解码一个有界 AS 全局模型分块响应。
        """
        if response.message_type != AS_GLOBAL_MODEL_CHUNK_RESPONSE:
            raise TransportError("AS returned an invalid global-model response / AS 返回了无效全局模型响应")
        payload = response.payload
        expected_fields = {
            "round",
            "offset",
            "global_bytes",
            "global_sha256",
            "chunk_base64",
            "complete",
        }
        if (
            set(payload) != expected_fields
            or payload["round"] != round_id
            or payload["offset"] != offset
        ):
            raise TransportError("AS global-model response does not match / AS 全局模型响应不匹配")
        total_bytes = payload["global_bytes"]
        sha256 = payload["global_sha256"]
        encoded_chunk = payload["chunk_base64"]
        complete = payload["complete"]
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
            raise TransportError("AS global model size is invalid / AS 全局模型大小无效")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise TransportError("AS global model digest is invalid / AS 全局模型摘要无效")
        if not isinstance(encoded_chunk, str) or not isinstance(complete, bool):
            raise TransportError("AS global model chunk is invalid / AS 全局模型分块无效")
        try:
            chunk = base64.b64decode(encoded_chunk.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise TransportError(
                "AS global model chunk is not base64 / AS 全局模型分块不是 Base64"
            ) from error
        if len(chunk) > self.config.model_chunk_bytes:
            raise TransportError("AS global model chunk exceeds bound / AS 全局模型分块超过上限")
        if offset + len(chunk) > total_bytes or (complete and offset + len(chunk) != total_bytes):
            raise TransportError("AS global model chunk bounds are invalid / AS 全局模型分块边界无效")
        return total_bytes, sha256, chunk, complete

    @staticmethod
    def _record_key(record: bytes) -> str:
        """Encode raw local data losslessly as one canonical JSON-safe key.

        将原始本地数据无损编码为规范且 JSON 安全的键。
        """
        return base64.b64encode(record).decode("ascii")


def _validated_sid_sequence(participant_sids: Sequence[int]) -> tuple[int, ...]:
    """Validate a non-empty, ordered, duplicate-free FedAvg roster.

    验证非空、有序且不重复的 FedAvg 名册。
    """
    if isinstance(participant_sids, (str, bytes)) or not isinstance(participant_sids, Sequence):
        raise TypeError("participant_sids must be a sequence / participant_sids 必须是一个序列")
    normalized = tuple(participant_sids)
    if (
        not normalized
        or any(isinstance(sid, bool) or not isinstance(sid, int) or sid < 1 for sid in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("participant_sids must be unique positive integers / "
                         "participant_sids 必须为唯一正整数")
    return normalized
