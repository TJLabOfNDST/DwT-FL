"""Aggregation Server session management compatible with the DwT-FL index.

与 DwT-FL 双向索引兼容的聚合服务器会话管理。
"""

from __future__ import annotations

import math
import threading
import time
from hmac import compare_digest
from base64 import b64decode, b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dbtfl.communication import JsonRouter, RequestRejected, ThreadedJsonServer, WireMessage
from dbtfl.communication.endpoints import AggregationServerPath
from dbtfl.federation import (
    FedAvgError,
    ModelUpdateDescriptor,
    ModelUpdateStore,
    aggregate_safetensors,
    sha256_file,
)
from dbtfl.native_index import NativeIndex, NativeIndexError, TaskState
from dbtfl.oprf import OprfValidationError, validate_protected_label


AS_REGISTER_REQUEST: Final[str] = "as.client.register.request"
AS_REGISTER_RESPONSE: Final[str] = "as.client.register.response"
AS_HEARTBEAT_REQUEST: Final[str] = "as.client.heartbeat.request"
AS_HEARTBEAT_RESPONSE: Final[str] = "as.client.heartbeat.response"
AS_REGISTER_LABELS_REQUEST: Final[str] = "as.labels.register.request"
AS_REGISTER_LABELS_RESPONSE: Final[str] = "as.labels.register.response"
AS_CLAIM_TASKS_REQUEST: Final[str] = "as.tasks.claim.request"
AS_CLAIM_TASKS_RESPONSE: Final[str] = "as.tasks.claim.response"
AS_MODEL_CHUNK_REQUEST: Final[str] = "as.model.chunk.request"
AS_MODEL_CHUNK_RESPONSE: Final[str] = "as.model.chunk.response"
AS_MODEL_FINALIZE_REQUEST: Final[str] = "as.model.finalize.request"
AS_MODEL_FINALIZE_RESPONSE: Final[str] = "as.model.finalize.response"
AS_MODEL_AGGREGATE_REQUEST: Final[str] = "as.model.aggregate.request"
AS_MODEL_AGGREGATE_RESPONSE: Final[str] = "as.model.aggregate.response"
AS_GLOBAL_MODEL_CHUNK_REQUEST: Final[str] = "as.global_model.chunk.request"
AS_GLOBAL_MODEL_CHUNK_RESPONSE: Final[str] = "as.global_model.chunk.response"
AS_CONFIGURE_ROUND_REQUEST: Final[str] = "as.round.configure.request"
AS_CONFIGURE_ROUND_RESPONSE: Final[str] = "as.round.configure.response"
AS_METRICS_REQUEST: Final[str] = "as.evaluation.metrics.request"
AS_METRICS_RESPONSE: Final[str] = "as.evaluation.metrics.response"
AS_EVALUATION_RESET_REQUEST: Final[str] = "as.evaluation.reset.request"
AS_EVALUATION_RESET_RESPONSE: Final[str] = "as.evaluation.reset.response"
AS_EVALUATION_LEASE_REQUEST: Final[str] = "as.evaluation.lease.request"
AS_EVALUATION_LEASE_RESPONSE: Final[str] = "as.evaluation.lease.response"
MAX_LABELS_PER_SUBMISSION: Final[int] = 4_096
MAX_MODEL_CHUNK_BYTES: Final[int] = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ClientSessionSnapshot:
    """Current AS-side liveness state for one globally unique SID.

    一个全局唯一 SID 的当前 AS 侧存活状态。
    """

    sid: int
    client_id: str
    online: bool
    heartbeat_count: int
    recovery_risk: bool


@dataclass(frozen=True, slots=True)
class GlobalModelDescriptor:
    """AS-local global checkpoint metadata available for a completed round.

    为一个完成轮次提供的 AS 本地全局检查点元数据。
    """

    round_id: int
    checkpoint_path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class RoundConfiguration:
    """Fixed FedAvg roster selected before client updates arrive.

    在客户端更新到达前选择的固定 FedAvg 名册。
    """

    round_id: int
    participant_sids: tuple[int, ...]


@dataclass(slots=True)
class _ClientSession:
    """Mutable session timer state private to the AS service.

    AS 服务私有的可变会话计时状态。
    """

    sid: int
    client_id: str
    last_heartbeat_seconds: float
    online: bool = True
    heartbeat_count: int = 0
    recovery_risk: bool = False

    def snapshot(self) -> ClientSessionSnapshot:
        """Copy public liveness information without exposing timer internals.

        复制公开存活信息，不暴露计时器内部细节。
        """
        return ClientSessionSnapshot(
            sid=self.sid,
            client_id=self.client_id,
            online=self.online,
            heartbeat_count=self.heartbeat_count,
            recovery_risk=self.recovery_risk,
        )


class AggregationServerService:
    """Issue SIDs and maintain heartbeat timers alongside the native index.

    在原生索引旁发放 SID 并维护心跳计时器。
    """

    def __init__(
        self,
        index: NativeIndex,
        *,
        heartbeat_interval_seconds: float,
        heartbeat_timeout_seconds: float,
        model_update_directory: Path,
        max_model_update_bytes: int,
        claim_mode: str = "cas",
        recovery_index_mode: str = "inverse",
        history_scheduling_enabled: bool = True,
        evaluation_reset_token: str | None = None,
        evaluation_reset_callback: Callable[[int, float, float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create session routes without exposing index internals over HTTP.

        创建会话路由，不通过 HTTP 暴露索引内部状态。
        """
        if heartbeat_interval_seconds <= 0 or heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat values must be positive / 心跳参数必须为正数")
        if heartbeat_timeout_seconds <= heartbeat_interval_seconds:
            raise ValueError(
                "heartbeat timeout must exceed interval / 心跳超时必须大于发送周期"
            )
        if claim_mode not in {"cas", "mutex"}:
            raise ValueError("claim_mode must be cas or mutex / 抢占模式必须为 cas 或 mutex")
        if recovery_index_mode not in {"inverse", "scan"}:
            raise ValueError("recovery_index_mode must be inverse or scan / 恢复索引模式必须为 inverse 或 scan")
        self.index = index
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.claim_mode = claim_mode
        self.recovery_index_mode = recovery_index_mode
        self.history_scheduling_enabled = history_scheduling_enabled
        self._clock = clock
        self._sessions_by_sid: dict[int, _ClientSession] = {}
        self._sid_by_client_id: dict[str, int] = {}
        self._offline_detected_at: dict[int, float] = {}
        self._recovery_takeover_at: dict[int, float] = {}
        self._next_sid = 1
        self._lock = threading.RLock()
        self._mutex_claim_lock = threading.Lock()
        self.model_update_store = ModelUpdateStore(
            model_update_directory,
            max_update_bytes=max_model_update_bytes,
        )
        self._updates_by_round: dict[int, dict[int, ModelUpdateDescriptor]] = {}
        self._global_models: dict[int, GlobalModelDescriptor] = {}
        self._round_configurations: dict[int, RoundConfiguration] = {}
        self._active_round = 0
        self._round_dispatch_enabled = False
        self._round_dispatch_sealed = False
        self._evaluation_reset_token = evaluation_reset_token
        self._evaluation_reset_callback = evaluation_reset_callback
        self.router = JsonRouter()
        self.router.add("POST", AggregationServerPath.REGISTER_CLIENT.value, self.register)
        self.router.add("POST", AggregationServerPath.HEARTBEAT.value, self.heartbeat)
        self.router.add("POST", AggregationServerPath.REGISTER_LABELS.value, self.register_labels)
        self.router.add("POST", AggregationServerPath.CLAIM_TASK.value, self.claim_tasks)
        self.router.add("POST", AggregationServerPath.SUBMIT_MODEL_UPDATE.value, self.model_update)
        self.router.add(
            "POST",
            AggregationServerPath.AGGREGATE_MODEL_UPDATES.value,
            self.aggregate_model_updates,
        )
        self.router.add(
            "POST",
            AggregationServerPath.DOWNLOAD_GLOBAL_MODEL.value,
            self.download_global_model,
        )
        self.router.add(
            "POST",
            AggregationServerPath.CONFIGURE_ROUND.value,
            self.configure_round,
        )
        self.router.add(
            "POST",
            AggregationServerPath.METRICS.value,
            self.evaluation_metrics,
        )
        self.router.add(
            "POST",
            AggregationServerPath.EVALUATION_RESET.value,
            self.reset_evaluation,
        )
        self.router.add(
            "POST",
            AggregationServerPath.EVALUATION_LEASE.value,
            self.configure_evaluation_lease,
        )

    def register(self, message: WireMessage) -> WireMessage:
        """Issue an unused SID or reconnect a known client with its same SID.

        发放未使用 SID，或让已知客户端使用原 SID 重连。
        """
        if message.message_type != AS_REGISTER_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.client.register.request / 应为 as.client.register.request",
            )
        client_id = self._client_id_from_payload(message.payload)
        now = self._clock()
        with self._lock:
            existing_sid = self._sid_by_client_id.get(client_id)
            if existing_sid is None:
                if self._next_sid > self.index.max_clients:
                    raise RequestRejected(
                        503,
                        "sid_capacity_exhausted",
                        "AS has no remaining SID capacity / AS 没有剩余 SID 容量",
                    )
                sid = self._next_sid
                self._next_sid += 1
                self._sid_by_client_id[client_id] = sid
                self._sessions_by_sid[sid] = _ClientSession(sid, client_id, now)
                reused = False
            else:
                sid = existing_sid
                session = self._sessions_by_sid[sid]
                session.last_heartbeat_seconds = now
                session.online = True
                reused = True
        return WireMessage.create(
            AS_REGISTER_RESPONSE,
            {
                "sid": sid,
                "reused": reused,
                "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            },
            request_id=message.request_id,
        )

    def heartbeat(self, message: WireMessage) -> WireMessage:
        """Refresh one known SID timer and confirm its active connection.

        刷新一个已知 SID 的计时器，并确认其活动连接。
        """
        if message.message_type != AS_HEARTBEAT_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.client.heartbeat.request / 应为 as.client.heartbeat.request",
            )
        sid = self._sid_from_payload(message.payload)
        now = self._clock()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(
                    404,
                    "unknown_sid",
                    "SID has not been registered / SID 尚未注册",
                )
            session.last_heartbeat_seconds = now
            session.online = True
            session.heartbeat_count += 1
        instructions = self._round_instructions_for_sid(sid)
        return WireMessage.create(
            AS_HEARTBEAT_RESPONSE,
            {
                "sid": sid,
                "online": True,
                "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
                "round": self._active_round,
                "instructions": instructions,
            },
            request_id=message.request_id,
        )

    def register_labels(self, message: WireMessage) -> WireMessage:
        """Append both directions while leaving every new task in EMPTY state.

        追加双向索引，并让每个新任务保持在 EMPTY 状态。
        """
        if message.message_type != AS_REGISTER_LABELS_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.labels.register.request / 应为 as.labels.register.request",
            )
        sid, created_round, labels = self._label_submission_from_payload(message.payload)
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(
                    404,
                    "unknown_sid",
                    "SID has not been registered / SID 尚未注册",
                )
            if not session.online:
                raise RequestRejected(
                    409,
                    "sid_offline",
                    "offline SID must reconnect before submitting labels / "
                    "离线 SID 必须重连后才能提交标签",
                )

        registrations: list[dict[str, object]] = []
        for protected_label in labels:
            try:
                task_id = self.index.register_label(protected_label, sid, created_round)
            except (NativeIndexError, ValueError) as error:
                raise RequestRejected(
                    503,
                    "index_registration_failed",
                    "AS native index could not register the label / "
                    "AS 原生索引无法登记该标签",
                ) from error
            registrations.append(
                {
                    "protected_label": protected_label,
                    "task_id": task_id,
                }
            )
        return WireMessage.create(
            AS_REGISTER_LABELS_RESPONSE,
            {
                "sid": sid,
                "round": created_round,
                "registrations": registrations,
            },
            request_id=message.request_id,
        )

    def claim_tasks(self, message: WireMessage) -> WireMessage:
        """Perform the separate paper CAS phase for already registered labels.

        为已登记标签执行论文中独立的 CAS 阶段。
        """
        if message.message_type != AS_CLAIM_TASKS_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.tasks.claim.request / 应为 as.tasks.claim.request",
            )
        sid, labels = self._claim_payload_from_message(message.payload)
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(404, "unknown_sid", "SID has not been registered / SID 尚未注册")
            if not session.online:
                raise RequestRejected(
                    409,
                    "sid_offline",
                    "offline SID must reconnect before claiming tasks / "
                    "离线 SID 必须重连后才能抢占任务",
                )

        decisions: list[dict[str, object]] = []
        for protected_label in labels:
            task_id = self.index.find_label(protected_label)
            if task_id is None:
                raise RequestRejected(
                    404,
                    "unknown_protected_label",
                    "protected label has not been registered / 受保护标签尚未登记",
                )
            if not self.index.task_has_owner(task_id, sid):
                raise RequestRejected(
                    403,
                    "sid_not_label_owner",
                    "SID does not own this protected label / SID 不拥有该受保护标签",
                )
            initial_snapshot = self.index.snapshot(task_id)
            if initial_snapshot.state == TaskState.EMPTY:
                claimed = self._claim_task(task_id, sid)
                final_snapshot = self.index.snapshot(task_id)
                operation = (
                    "TRAIN"
                    if claimed or (
                        final_snapshot.state == TaskState.PENDING
                        and final_snapshot.trainer == sid
                    )
                    else "DEDUP"
                )
            else:
                final_snapshot = initial_snapshot
                # Concurrent request shards from the same client can observe a
                # PENDING transition that another shard has just won for the
                # same SID. This is an idempotent TRAIN acknowledgement, not a
                # deduplication loss; a different SID remains DEDUP. 同一客户
                # 端的并发请求分片可能观察到另一分片刚刚为同一 SID 赢得的 PENDING
                # 状态。这应幂等地确认 TRAIN，而不是误判为去重；不同 SID 仍为 DEDUP。
                operation = (
                    "TRAIN"
                    if initial_snapshot.state == TaskState.PENDING
                    and initial_snapshot.trainer == sid
                    else "DEDUP"
                )
            decisions.append(
                {
                    "protected_label": protected_label,
                    "task_id": task_id,
                    "operation": operation,
                    "state": final_snapshot.state.name,
                }
            )
        return WireMessage.create(
            AS_CLAIM_TASKS_RESPONSE,
            {"sid": sid, "decisions": decisions},
            request_id=message.request_id,
        )

    def _claim_task(self, task_id: int, sid: int) -> bool:
        """Claim with production CAS or the explicit pessimistic-lock ablation.

        通过生产 CAS 或显式悲观锁消融版本抢占任务。

        The mutex mode serializes the entire claim decision before calling the
        same native state transition. It is intentionally a pessimistic control,
        not an alternative result mislabeled as lock-free CAS. ``mutex`` 模式在
        调用同一原生状态迁移前串行化完整抢占决策；它是刻意的悲观控制组，不会被错误
        标记为无锁 CAS 的替代结果。
        """
        if self.claim_mode == "mutex":
            with self._mutex_claim_lock:
                return self.index.try_claim(task_id, sid)
        return self.index.try_claim(task_id, sid)

    def model_update(self, message: WireMessage) -> WireMessage:
        """Receive one bounded checkpoint chunk or finalize a client update.

        接收一个有界检查点分块，或完成一个客户端更新。
        """
        if message.message_type == AS_MODEL_CHUNK_REQUEST:
            return self._append_model_chunk(message)
        if message.message_type == AS_MODEL_FINALIZE_REQUEST:
            return self._finalize_model_update(message)
        raise RequestRejected(
            400,
            "unexpected_message_type",
            "expected model chunk or finalize request / 应为模型分块或完成请求",
        )

    def _round_instructions_for_sid(self, sid: int) -> list[dict[str, object]]:
        """Issue idempotent next-round TRAIN or DEDUP work on a heartbeat.

        在心跳中下发幂等的下一轮 TRAIN 或 DEDUP 工作。
        """
        with self._lock:
            if not self._round_dispatch_enabled:
                return []
            dispatch_sealed = self._round_dispatch_sealed
        instructions: list[dict[str, object]] = []
        for task_id in self.index.client_tasks(sid):
            snapshot = self.index.snapshot(task_id)
            operation = "DEDUP"
            if snapshot.state == TaskState.PENDING and snapshot.trainer == sid:
                operation = "TRAIN"
            elif snapshot.state == TaskState.EMPTY and not dispatch_sealed:
                selected_sid = self._select_next_trainer(task_id)
                if selected_sid == sid and self._claim_task(task_id, sid):
                    was_recovery = self.index.recovery_required(task_id)
                    self.index.set_recovery_required(task_id, False)
                    if was_recovery:
                        with self._lock:
                            self._recovery_takeover_at[sid] = self._clock()
                    operation = "TRAIN"
            instructions.append(
                {
                    "protected_label": self.index.task_label(task_id),
                    "task_id": task_id,
                    "operation": operation,
                }
            )
        return instructions

    def _select_next_trainer(self, task_id: int) -> int | None:
        """Choose the next-round trainer under the configured history policy.

        按已配置的历史策略选择下一轮训练者。

        When history scheduling is enabled, a client that timed out in an
        earlier round remains marked as ``recovery_risk`` after it reconnects.
        The scheduler excludes that client from duplicated tasks whenever a
        healthy online owner exists, as required by the paper's training-right
        allocation adjustment.  The explicit ``w/o history scheduling``
        ablation deliberately does not consume this historical risk record:
        it follows the ordinary stable-case rule and reuses the previous online
        trainer.  历史调度启用时，曾在早前轮次超时的客户端即使重连仍保留
        ``recovery_risk`` 标记；只要存在健康在线所有者，调度器便会将重复数据
        排除在该风险客户端之外，符合论文的训练权分配策略调整。显式的“去除历史
        调度”消融则刻意不读取该历史风险记录，而按稳定场景的常规规则复用上一轮
        在线训练者。
        """
        owners = self.index.owners(task_id)
        with self._lock:
            online_owners = tuple(
                owner
                for owner in owners
                if (session := self._sessions_by_sid.get(owner)) is not None and session.online
            )
            safe_owners = tuple(
                owner
                for owner in online_owners
                if not self._sessions_by_sid[owner].recovery_risk
            )
        previous_trainer = self.index.previous_trainer(task_id)
        if not online_owners:
            return None
        if not self.history_scheduling_enabled:
            return (
                previous_trainer
                if previous_trainer in online_owners
                else min(online_owners)
            )
        if self.index.recovery_required(task_id):
            return min(safe_owners or online_owners)
        if previous_trainer in safe_owners:
            return previous_trainer
        return min(safe_owners or online_owners)

    def aggregate_model_updates(self, message: WireMessage) -> WireMessage:
        """Apply FedAvg to the requested complete client-update set.

        对请求的完整客户端更新集合执行 FedAvg。
        """
        if message.message_type != AS_MODEL_AGGREGATE_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.model.aggregate.request / 应为 as.model.aggregate.request",
            )
        round_id, expected_sids = self._aggregate_payload(message.payload)
        with self._lock:
            updates = self._updates_by_round.get(round_id, {})
            missing_sids = sorted(set(expected_sids).difference(updates))
            if missing_sids:
                raise RequestRejected(
                    409,
                    "model_updates_incomplete",
                    f"missing model updates for SIDs {missing_sids} / 缺少 SID {missing_sids} 的模型更新",
                )
            descriptors = tuple(updates[sid] for sid in expected_sids)
            configured = self._round_configurations.get(round_id)
            if configured is not None and configured.participant_sids != expected_sids:
                raise RequestRejected(
                    409,
                    "round_roster_mismatch",
                    "FedAvg participants differ from the configured round roster / "
                    "FedAvg 参与者与已配置轮次名册不同",
                )
            pending_tasks = [
                task_id
                for task_id in self.index.all_task_ids()
                if self.index.snapshot(task_id).state == TaskState.PENDING
            ]
            if pending_tasks:
                raise RequestRejected(
                    409,
                    "round_tasks_incomplete",
                    "all PENDING tasks must finish or recover before aggregation / "
                    "聚合前所有 PENDING 任务必须完成或恢复",
                )
        output_path = self.model_update_store.global_checkpoint_path(round_id)
        try:
            aggregate_safetensors(
                [descriptor.checkpoint_path for descriptor in descriptors],
                [descriptor.sample_count for descriptor in descriptors],
                output_path,
            )
        except (FedAvgError, FileNotFoundError, RuntimeError) as error:
            raise RequestRejected(
                422,
                "fedavg_failed",
                f"FedAvg could not aggregate updates: {error} / FedAvg 无法聚合更新：{error}",
            ) from error
        descriptor = GlobalModelDescriptor(
            round_id=round_id,
            checkpoint_path=output_path,
            sha256=sha256_file(output_path),
            byte_count=output_path.stat().st_size,
        )
        with self._lock:
            self._global_models[round_id] = descriptor
            self._advance_to_next_round(round_id)
        return WireMessage.create(
            AS_MODEL_AGGREGATE_RESPONSE,
            {
                "round": round_id,
                "participant_sids": list(expected_sids),
                "total_samples": sum(descriptor.sample_count for descriptor in descriptors),
                "global_sha256": descriptor.sha256,
                "global_bytes": descriptor.byte_count,
            },
            request_id=message.request_id,
        )

    def download_global_model(self, message: WireMessage) -> WireMessage:
        """Return one bounded global-checkpoint chunk to an online client.

        向在线客户端返回一个有界全局检查点分块。
        """
        if message.message_type != AS_GLOBAL_MODEL_CHUNK_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.global_model.chunk.request / 应为 as.global_model.chunk.request",
            )
        sid, round_id, offset, max_bytes = self._global_model_chunk_payload(message.payload)
        self._require_online_sid(
            sid,
            operation=("downloading the global model", "下载全局模型"),
        )
        with self._lock:
            descriptor = self._global_models.get(round_id)
        if descriptor is None or not descriptor.checkpoint_path.is_file():
            raise RequestRejected(
                404,
                "global_model_not_found",
                "global model is not available for this round / 该轮全局模型尚不可用",
            )
        if offset > descriptor.byte_count:
            raise RequestRejected(
                416,
                "invalid_global_model_offset",
                "offset exceeds global model size / 偏移量超过全局模型大小",
            )
        with descriptor.checkpoint_path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(max_bytes)
        complete = offset + len(chunk) == descriptor.byte_count
        return WireMessage.create(
            AS_GLOBAL_MODEL_CHUNK_RESPONSE,
            {
                "round": round_id,
                "offset": offset,
                "global_bytes": descriptor.byte_count,
                "global_sha256": descriptor.sha256,
                "chunk_base64": b64encode(chunk).decode("ascii"),
                "complete": complete,
            },
            request_id=message.request_id,
        )

    def configure_round(self, message: WireMessage) -> WireMessage:
        """Freeze an explicit FedAvg roster so clients may join sequentially.

        固定一个显式 FedAvg 名册，使客户端可以先后加入。
        """
        if message.message_type != AS_CONFIGURE_ROUND_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.round.configure.request / 应为 as.round.configure.request",
            )
        round_id, participant_sids = self._round_configuration_payload(message.payload)
        with self._lock:
            if any(sid not in self._sessions_by_sid for sid in participant_sids):
                raise RequestRejected(
                    404,
                    "unknown_round_participant",
                    "every configured SID must be registered / 每个配置 SID 必须已注册",
                )
            if self._updates_by_round.get(round_id):
                raise RequestRejected(
                    409,
                    "round_already_started",
                    "cannot change roster after an update arrives / 更新到达后不可更改名册",
                )
            configuration = RoundConfiguration(round_id, participant_sids)
            self._round_configurations[round_id] = configuration
            # Once FedAvg participants are frozen, background heartbeats may
            # report existing ownership but must not assign a newly released
            # task to a SID that may already have submitted its sole update for
            # this round. A later round, or the original SID's explicit
            # A frozen FedAvg roster fixes participants, not their unfinished
            # task set. Before aggregation a heartbeat may transfer a released
            # EMPTY task to an existing participant; that participant performs
            # incremental training and replaces its update before FedAvg. 固定
            # FedAvg 名册只固定参与者，不固定其未完成任务集合。聚合前心跳可将释放的
            # EMPTY 任务转移给既有参与者；该参与者增量训练并替换更新后再聚合。
            self._round_dispatch_sealed = False
        return WireMessage.create(
            AS_CONFIGURE_ROUND_RESPONSE,
            {"round": round_id, "participant_sids": list(participant_sids)},
            request_id=message.request_id,
        )

    def evaluation_metrics(self, message: WireMessage) -> WireMessage:
        """Return read-only AS metadata required by the experiment protocol.

        返回实验协议所需的只读 AS 元数据。

        This endpoint intentionally exposes capacities and byte counts only; it
        never returns protected labels, plaintext records, owners, or model
        content. 该端点刻意只公开容量和字节计数，绝不返回受保护标签、明文记录、
        所有者或模型内容。
        """
        if message.message_type != AS_METRICS_REQUEST or dict(message.payload):
            raise RequestRejected(
                400,
                "invalid_metrics_request",
                "metrics request must have the expected type and an empty payload / "
                "指标请求必须具有预期类型且负载为空",
            )
        with self._lock:
            sessions = tuple(self._sessions_by_sid.values())
            update_bytes = sum(
                descriptor.byte_count
                for updates in self._updates_by_round.values()
                for descriptor in updates.values()
            )
        return WireMessage.create(
            AS_METRICS_RESPONSE,
            {
                "native_index_bytes": self.index.memory_bytes,
                "task_count": len(self.index.all_task_ids()),
                "owner_edge_count": self.index.edge_count,
                "total_sessions": len(sessions),
                "online_sessions": sum(session.online for session in sessions),
                "accepted_model_update_bytes": update_bytes,
                "offline_detection_by_sid_seconds": {
                    str(sid): detected_at
                    for sid, detected_at in sorted(self._offline_detected_at.items())
                },
                "recovery_takeover_by_sid_seconds": {
                    str(sid): taken_at
                    for sid, taken_at in sorted(self._recovery_takeover_at.items())
                },
                "server_resources": self._server_resources(),
            },
            request_id=message.request_id,
        )

    def reset_evaluation(self, message: WireMessage) -> WireMessage:
        """Reset a dedicated experimental AS after constant-time token validation.

        在恒定时间令牌校验后重置专用实验 AS。

        This route is disabled unless deployment config supplies a token. It
        clears all in-memory sessions, indexes, and AS-owned update artifacts;
        never enable it on a shared production AS. 该路由仅在部署配置提供令牌时
        启用；它会清除所有内存会话、索引和 AS 拥有的更新产物，绝不可在共享生产 AS 启用。
        """
        if message.message_type != AS_EVALUATION_RESET_REQUEST or set(message.payload) != {
            "token", "backend_worker_count", "heartbeat_interval_seconds",
            "heartbeat_timeout_seconds",
        }:
            raise RequestRejected(
                400,
                "invalid_evaluation_reset_request",
                "reset request must contain token, backend workers, and heartbeat lease values / "
                "重置请求必须包含令牌、后端工作线程数与心跳租约参数",
            )
        supplied_token = message.payload["token"]
        backend_worker_count = message.payload["backend_worker_count"]
        heartbeat_interval_seconds = message.payload["heartbeat_interval_seconds"]
        heartbeat_timeout_seconds = message.payload["heartbeat_timeout_seconds"]
        if (
            isinstance(backend_worker_count, bool)
            or not isinstance(backend_worker_count, int)
            or backend_worker_count < 1
        ):
            raise RequestRejected(
                400,
                "invalid_backend_worker_count",
                "backend worker count must be positive / 后端工作线程数必须为正数",
            )
        if (
            isinstance(heartbeat_interval_seconds, bool)
            or not isinstance(heartbeat_interval_seconds, (int, float))
            or not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds <= 0
            or isinstance(heartbeat_timeout_seconds, bool)
            or not isinstance(heartbeat_timeout_seconds, (int, float))
            or not math.isfinite(heartbeat_timeout_seconds)
            or heartbeat_timeout_seconds <= heartbeat_interval_seconds
        ):
            raise RequestRejected(
                400,
                "invalid_heartbeat_lease",
                "heartbeat timeout must be finite, positive, and exceed its interval / "
                "心跳超时必须为有限正数且大于发送周期",
            )
        if (
            not isinstance(supplied_token, str)
            or self._evaluation_reset_token is None
            or self._evaluation_reset_callback is None
            or not compare_digest(supplied_token, self._evaluation_reset_token)
        ):
            raise RequestRejected(
                403,
                "evaluation_reset_forbidden",
                "evaluation reset is disabled or the token is invalid / 实验重置未启用或令牌无效",
            )
        # The evaluator changes this only at a destructive, token-protected,
        # case boundary. It separates a normal long GPT-training lease from the
        # deliberately short dropout-recovery lease. 评估器仅在受令牌保护的破坏性
        # 用例边界修改该值，从而分离正常长时 GPT 训练租约和专门的短时掉线恢复租约。
        self._evaluation_reset_callback(
            backend_worker_count,
            float(heartbeat_interval_seconds),
            float(heartbeat_timeout_seconds),
        )
        return WireMessage.create(
            AS_EVALUATION_RESET_RESPONSE,
            {
                "reset": True,
                "backend_worker_count": backend_worker_count,
                "heartbeat_interval_seconds": float(heartbeat_interval_seconds),
                "heartbeat_timeout_seconds": float(heartbeat_timeout_seconds),
            },
            request_id=message.request_id,
        )

    def configure_evaluation_lease(self, message: WireMessage) -> WireMessage:
        """Change only the lease of a dedicated experiment without resetting state.

        仅修改专用实验的租约，不重置任何状态。

        A training-dropout measurement must finish OPRF, label registration, and
        CAS under the normal long lease before it activates a short failure
        lease.  Resetting here would erase the exact task ownership that the
        measurement must recover, so this route changes no index, SID, model,
        or round state. 训练掉线测量必须先在正常长租约下完成 OPRF、标签登记和
        CAS，再启用短故障租约。此处若重置会抹除待恢复的精确任务所有权，因此该
        路由不会修改索引、SID、模型或轮次状态。
        """
        if message.message_type != AS_EVALUATION_LEASE_REQUEST or set(message.payload) != {
            "token", "heartbeat_timeout_seconds",
        }:
            raise RequestRejected(
                400,
                "invalid_evaluation_lease_request",
                "lease request must contain token and heartbeat timeout / "
                "租约请求必须包含令牌和心跳超时",
            )
        supplied_token = message.payload["token"]
        heartbeat_timeout_seconds = message.payload["heartbeat_timeout_seconds"]
        if (
            isinstance(heartbeat_timeout_seconds, bool)
            or not isinstance(heartbeat_timeout_seconds, (int, float))
            or not math.isfinite(heartbeat_timeout_seconds)
            or heartbeat_timeout_seconds <= self.heartbeat_interval_seconds
        ):
            raise RequestRejected(
                400,
                "invalid_heartbeat_lease",
                "heartbeat timeout must be finite and exceed its interval / "
                "心跳超时必须为有限数且大于发送周期",
            )
        if (
            not isinstance(supplied_token, str)
            or self._evaluation_reset_token is None
            or not compare_digest(supplied_token, self._evaluation_reset_token)
        ):
            raise RequestRejected(
                403,
                "evaluation_lease_forbidden",
                "evaluation lease control is disabled or the token is invalid / "
                "实验租约控制未启用或令牌无效",
            )
        with self._lock:
            # The lease switch is one atomic server-side event. Refresh every
            # currently online SID at the same instant before installing the
            # shorter timeout; otherwise the evaluator has to create a burst
            # of client heartbeats and an otherwise healthy SID can expire
            # between the last refresh and this control request. 租约切换是一次
            # 原子的服务端事件。在安装更短超时前，以同一时刻刷新全部在线 SID；
            # 否则评估器必须制造客户端心跳突发，并且健康 SID 可能在最后一次刷新与
            # 本控制请求之间被错误判定为超时。
            refreshed_at = self._clock()
            for session in self._sessions_by_sid.values():
                if session.online:
                    session.last_heartbeat_seconds = refreshed_at
            self.heartbeat_timeout_seconds = float(heartbeat_timeout_seconds)
        return WireMessage.create(
            AS_EVALUATION_LEASE_RESPONSE,
            {
                "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
                "heartbeat_timeout_seconds": float(heartbeat_timeout_seconds),
            },
            request_id=message.request_id,
        )

    def _clear_evaluation_state(self) -> None:
        """Clear service state while the entity owns an exclusive reset boundary.

        在实体持有独占重置边界时清除服务状态。
        """
        self._sessions_by_sid.clear()
        self._sid_by_client_id.clear()
        self._offline_detected_at.clear()
        self._recovery_takeover_at.clear()
        self._next_sid = 1
        self._updates_by_round.clear()
        self._global_models.clear()
        self._round_configurations.clear()
        self._active_round = 0
        self._round_dispatch_enabled = False
        self._round_dispatch_sealed = False
        self.model_update_store.clear()

    @staticmethod
    def _server_resources() -> dict[str, object]:
        """Return this AS process's current resource observation when available.

        可用时返回当前 AS 进程的资源观测值。
        """
        try:
            import os
            import psutil

            process = psutil.Process(os.getpid())
            return {
                "status": "available",
                "cpu_percent": process.cpu_percent(None),
                "rss_bytes": process.memory_info().rss,
            }
        except Exception as error:
            return {"status": "unavailable", "reason": type(error).__name__}

    def _advance_to_next_round(self, completed_round: int) -> None:
        """Retain successful trainers, reset index state, and enable heartbeats.

        保留成功训练者、重置索引状态，并启用心跳下发。
        """
        for task_id in self.index.all_task_ids():
            snapshot = self.index.snapshot(task_id)
            if snapshot.state == TaskState.COMMITTED and snapshot.trainer:
                self.index.set_previous_trainer(task_id, snapshot.trainer)
            self.index.reset(task_id)
            self.index.set_recovery_required(task_id, False)
        self._active_round = completed_round + 1
        self._round_dispatch_enabled = True
        self._round_dispatch_sealed = False

    def _append_model_chunk(self, message: WireMessage) -> WireMessage:
        """Decode and persist one ordered bounded base64 checkpoint chunk.

        解码并持久化一个有序且有界的 Base64 检查点分块。
        """
        sid, round_id, update_id, total_bytes, sha256, offset, encoded_chunk = (
            self._model_chunk_payload(message.payload)
        )
        self._require_online_sid(sid)
        try:
            chunk = b64decode(encoded_chunk.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise RequestRejected(
                400,
                "invalid_model_chunk",
                "chunk must be valid base64 / 分块必须是有效 Base64",
            ) from error
        if len(chunk) > MAX_MODEL_CHUNK_BYTES:
            raise RequestRejected(
                413,
                "model_chunk_too_large",
                "decoded model chunk exceeds limit / 解码后的模型分块超过上限",
            )
        if total_bytes > self.model_update_store.max_update_bytes:
            raise RequestRejected(
                413,
                "model_update_too_large",
                "model update exceeds configured limit / 模型更新超过配置上限",
            )
        try:
            received_bytes = self.model_update_store.append_chunk(
                round_id=round_id,
                sid=sid,
                update_id=update_id,
                total_bytes=total_bytes,
                sha256=sha256,
                offset=offset,
                chunk=chunk,
            )
        except ValueError as error:
            raise RequestRejected(409, "invalid_model_upload", str(error)) from error
        return WireMessage.create(
            AS_MODEL_CHUNK_RESPONSE,
            {
                "sid": sid,
                "round": round_id,
                "update_id": update_id,
                "received_bytes": received_bytes,
            },
            request_id=message.request_id,
        )

    def _finalize_model_update(self, message: WireMessage) -> WireMessage:
        """Publish one verified update and commit all tasks trained by its SID.

        发布一个经验证的更新，并提交其 SID 训练的全部任务。
        """
        sid, round_id, update_id, total_bytes, sha256, sample_count, task_ids = (
            self._model_finalize_payload(message.payload)
        )
        self._require_online_sid(sid)
        with self._lock:
            existing = self._updates_by_round.get(round_id, {}).get(sid)
            replacement_task_ids: set[int] = set()
            if existing is not None:
                if existing.update_id == update_id and existing.sha256 == sha256:
                    return self._finalize_response(existing, message.request_id)
                previous_ids = set(existing.task_ids)
                submitted_ids = set(task_ids)
                if not previous_ids.issubset(submitted_ids):
                    raise RequestRejected(
                        409,
                        "replacement_update_drops_committed_tasks",
                        "replacement update must retain every previously committed task / "
                        "替换更新必须保留所有已提交任务",
                    )
                replacement_task_ids = submitted_ids.difference(previous_ids)
            for task_id in task_ids:
                snapshot = self.index.snapshot(task_id)
                is_existing_commit = (
                    existing is not None
                    and task_id in existing.task_ids
                    and snapshot.state == TaskState.COMMITTED
                    and snapshot.trainer == sid
                )
                is_new_pending = snapshot.state == TaskState.PENDING and snapshot.trainer == sid
                if not (is_existing_commit or is_new_pending):
                    self.model_update_store.discard(
                        round_id=round_id,
                        sid=sid,
                        update_id=update_id,
                    )
                    # The rejected client must be told which of its originally
                    # trained protected labels have already been safely taken
                    # over.  It receives only protected labels, task IDs, and
                    # DEDUP instructions; another client's identity is never
                    # exposed. 被拒绝客户端必须获知其原训练标签中哪些已被安全接管；
                    # 响应仅包含受保护标签、任务 ID 与 DEDUP 指令，绝不暴露其他客户端身份。
                    raise RequestRejected(
                        409,
                        "task_not_pending_for_sid",
                        "all tasks must remain PENDING for the submitting SID / "
                        "所有任务必须仍由提交 SID 处于 PENDING 状态",
                        {
                            "dedup_instructions": self._lost_training_dedup_instructions(
                                sid,
                                task_ids,
                            )
                        },
                    )
        try:
            checkpoint_path = self.model_update_store.finalize(
                round_id=round_id,
                sid=sid,
                update_id=update_id,
                total_bytes=total_bytes,
                sha256=sha256,
            )
        except ValueError as error:
            raise RequestRejected(409, "model_upload_incomplete", str(error)) from error
        with self._lock:
            for task_id in (replacement_task_ids if existing is not None else task_ids):
                if not self.index.mark_committed(task_id, sid):
                    raise RequestRejected(
                        409,
                        "task_commit_failed",
                        "task was no longer pending for the submitting SID / "
                        "任务不再由提交 SID 处于挂起状态",
                    )
            # FedAvg weights are authoritative AS state, not a client-declared
            # dataset size.  A replacement update retains earlier COMMITTED
            # tasks, while a first update has just committed its PENDING tasks;
            # both cases are counted from the state table after this transition.
            # FedAvg 权重以 AS 状态表为准，而非客户端声明的数据集大小。替换更新会
            # 保留先前的 COMMITTED 任务，首次更新则刚刚提交其 PENDING 任务；两种
            # 情况均在上述状态转换后直接由状态表计数。
            committed_sample_count = sum(
                1
                for task_id in self.index.all_task_ids()
                if (
                    snapshot := self.index.snapshot(task_id)
                ).state == TaskState.COMMITTED and snapshot.trainer == sid
            )
            if committed_sample_count < 1:
                raise RequestRejected(
                    409,
                    "no_committed_tasks_for_sid",
                    "the update has no COMMITTED tasks for its SID / "
                    "该更新没有属于其 SID 的 COMMITTED 任务",
                )
            descriptor = ModelUpdateDescriptor(
                round_id=round_id,
                sid=sid,
                update_id=update_id,
                sample_count=committed_sample_count,
                task_ids=task_ids,
                sha256=sha256,
                byte_count=total_bytes,
                checkpoint_path=checkpoint_path,
            )
            self._updates_by_round.setdefault(round_id, {})[sid] = descriptor
        return self._finalize_response(descriptor, message.request_id)

    def _lost_training_dedup_instructions(
        self,
        submitting_sid: int,
        task_ids: Sequence[int],
    ) -> list[dict[str, object]]:
        """Describe only labels already unusable by the rejected trainer.

        仅描述对被拒绝训练者已不可用的标签。

        A released ``EMPTY`` task is deliberately omitted: it has not yet been
        taken over, so the reconnecting client may legally win a later CAS. A
        ``PENDING`` task belongs in this payload only when another currently
        online SID owns it. ``COMMITTED`` work is also immutable and therefore
        returned as DEDUP. 已释放的 ``EMPTY`` 任务被刻意省略：其尚未被接管，重连
        客户端仍可在之后合法赢得 CAS。``PENDING`` 任务仅在另一在线 SID 拥有时才加入
        该载荷；``COMMITTED`` 工作同样不可变，因此也以 DEDUP 返回。
        """
        instructions: list[dict[str, object]] = []
        for task_id in task_ids:
            snapshot = self.index.snapshot(task_id)
            taken_over = (
                snapshot.state == TaskState.PENDING
                and snapshot.trainer != submitting_sid
                and self._is_online_sid(snapshot.trainer)
            )
            committed_elsewhere = (
                snapshot.state == TaskState.COMMITTED
                and snapshot.trainer != submitting_sid
            )
            if taken_over or committed_elsewhere:
                instructions.append(
                    {
                        "protected_label": self.index.task_label(task_id),
                        "task_id": task_id,
                        "operation": "DEDUP",
                        "state": snapshot.state.name,
                    }
                )
        return instructions

    def _is_online_sid(self, sid: int) -> bool:
        """Return liveness without revealing session data outside this service.

        在不向服务外泄露会话数据的情况下返回存活状态。
        """
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            return session is not None and session.online

    @staticmethod
    def _finalize_response(descriptor: ModelUpdateDescriptor, request_id: str) -> WireMessage:
        """Return an idempotent model-update acknowledgement. / 返回幂等的模型更新确认。"""
        return WireMessage.create(
            AS_MODEL_FINALIZE_RESPONSE,
            {
                "sid": descriptor.sid,
                "round": descriptor.round_id,
                "update_id": descriptor.update_id,
                "committed_task_ids": list(descriptor.task_ids),
                "checkpoint_sha256": descriptor.sha256,
            },
            request_id=request_id,
        )

    def _require_online_sid(
        self,
        sid: int,
        *,
        operation: tuple[str, str] = ("uploading", "上传"),
    ) -> None:
        """Ensure the requested operation is attributed to an online SID.

        确保请求操作仅归属于在线且已注册的 SID。

        ``operation`` contains English and Chinese response text only; it makes a rejected immutable
        download diagnosable without changing the shared SID state machine.
        ``operation`` 仅用于响应文本；它使不可变下载的拒绝可诊断，而不改变共享 SID
        状态机。
        """
        operation_english, operation_chinese = operation
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(404, "unknown_sid", "SID has not been registered / SID 尚未注册")
            if not session.online:
                raise RequestRejected(
                    409,
                    "sid_offline",
                    f"offline SID must reconnect before {operation_english} / "
                    f"离线 SID 必须重连后才能{operation_chinese}",
                )

    def expire_sessions(self) -> tuple[int, ...]:
        """Mark timers older than tau as offline and return newly expired SIDs.

        将超过 tau 的计时器标记为离线，并返回新超时的 SID。
        """
        now = self._clock()
        expired: list[int] = []
        with self._lock:
            for session in self._sessions_by_sid.values():
                if (
                    session.online
                    and now - session.last_heartbeat_seconds > self.heartbeat_timeout_seconds
                ):
                    session.online = False
                    session.recovery_risk = True
                    self._offline_detected_at[session.sid] = now
                    expired.append(session.sid)
        for sid in expired:
            self._release_dropped_trainer_tasks(sid)
        return tuple(expired)

    def _release_dropped_trainer_tasks(self, sid: int) -> None:
        """Release one dropped SID's tasks through index or scan recovery.

        通过倒排索引或扫描恢复路径释放一个掉线 SID 的任务。

        ``scan`` is the explicit ``w/o inverse index`` ablation: it deliberately
        traverses every task and preserves the same state semantics. The native
        inverse table remains allocated for ABI compatibility but is not read by
        this recovery path. ``scan`` 是显式的“去除倒排索引”消融：它刻意遍历每个任务，
        同时保留完全相同的状态语义。为 ABI 兼容性原生倒排表仍会被分配，但该恢复路径
        不读取它。
        """
        task_ids = (
            self.index.client_tasks(sid)
            if self.recovery_index_mode == "inverse"
            else tuple(
                task_id
                for task_id in self.index.all_task_ids()
                if self.index.snapshot(task_id).trainer == sid
            )
        )
        for task_id in task_ids:
            snapshot = self.index.snapshot(task_id)
            if snapshot.state == TaskState.PENDING and snapshot.trainer == sid:
                if self.index.release_if_trainer(task_id, sid):
                    self.index.set_recovery_required(task_id, True)
                    with self._lock:
                        self._round_dispatch_enabled = True

    def session_snapshot(self, sid: int) -> ClientSessionSnapshot | None:
        """Return one current session snapshot after applying timeout detection.

        应用超时检测后返回一个当前会话快照。
        """
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            return None if session is None else session.snapshot()

    @staticmethod
    def _client_id_from_payload(payload: Mapping[str, Any]) -> str:
        """Validate the one-field client registration payload.

        验证仅含一个字段的客户端注册载荷。
        """
        if set(payload) != {"client_id"}:
            raise RequestRejected(
                400,
                "invalid_registration_payload",
                "payload must contain only client_id / 载荷只能包含 client_id",
            )
        client_id = payload["client_id"]
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise RequestRejected(
                400,
                "invalid_client_id",
                "client_id must contain 1..128 characters / client_id 必须包含 1..128 个字符",
            )
        return client_id

    @staticmethod
    def _sid_from_payload(payload: Mapping[str, Any]) -> int:
        """Validate the one-field heartbeat payload.

        验证仅含一个字段的心跳载荷。
        """
        if set(payload) != {"sid"}:
            raise RequestRejected(
                400,
                "invalid_heartbeat_payload",
                "payload must contain only sid / 载荷只能包含 sid",
            )
        sid = payload["sid"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(
                400,
                "invalid_sid",
                "sid must be a positive integer / sid 必须是正整数",
            )
        return sid

    @staticmethod
    def _label_submission_from_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, tuple[str, ...]]:
        """Validate a complete SID, round, and protected-label-set submission.

        验证完整的 SID、轮次与受保护标签集合提交。
        """
        if set(payload) != {"sid", "round", "protected_labels"}:
            raise RequestRejected(
                400,
                "invalid_label_payload",
                "payload must contain sid, round, and protected_labels / "
                "载荷必须包含 sid、round 和 protected_labels",
            )
        sid = payload["sid"]
        created_round = payload["round"]
        protected_labels = payload["protected_labels"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be a positive integer / sid 必须是正整数")
        if (
            isinstance(created_round, bool)
            or not isinstance(created_round, int)
            or not 0 <= created_round <= 0xFFFFFFFF
        ):
            raise RequestRejected(
                400,
                "invalid_round",
                "round must fit uint32 / round 必须适配 uint32",
            )
        if (
            not isinstance(protected_labels, Sequence)
            or isinstance(protected_labels, (str, bytes))
            or not 1 <= len(protected_labels) <= MAX_LABELS_PER_SUBMISSION
        ):
            raise RequestRejected(
                400,
                "invalid_label_batch",
                "protected_labels must be a non-empty bounded array / "
                "protected_labels 必须为非空且有上限的数组",
            )
        labels = tuple(protected_labels)
        if any(not isinstance(label, str) for label in labels):
            raise RequestRejected(
                400,
                "invalid_protected_label",
                "every protected label must be a string / 每个受保护标签必须是字符串",
            )
        if len(set(labels)) != len(labels):
            raise RequestRejected(
                400,
                "duplicate_protected_label",
                "protected-label set must not contain duplicates / "
                "受保护标签集合不得包含重复值",
            )
        try:
            for label in labels:
                validate_protected_label(label)
        except OprfValidationError as error:
            raise RequestRejected(
                400,
                "invalid_protected_label",
                str(error),
            ) from error
        return sid, created_round, labels

    @staticmethod
    def _claim_payload_from_message(payload: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
        """Validate one claim request without re-registering its label set.

        验证一个任务抢占请求，不重复登记其中的标签集合。
        """
        if set(payload) != {"sid", "protected_labels"}:
            raise RequestRejected(
                400,
                "invalid_claim_payload",
                "payload must contain sid and protected_labels / "
                "载荷必须包含 sid 和 protected_labels",
            )
        sid = payload["sid"]
        protected_labels = payload["protected_labels"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be a positive integer / sid 必须是正整数")
        labels = AggregationServerService._validated_protected_label_set(protected_labels)
        return sid, labels

    @staticmethod
    def _model_chunk_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, str, int, str, int, str]:
        """Validate metadata for one bounded base64 model-update chunk.

        验证一个有界 Base64 模型更新分块的元数据。
        """
        expected_fields = {
            "sid",
            "round",
            "update_id",
            "total_bytes",
            "sha256",
            "offset",
            "chunk_base64",
        }
        if set(payload) != expected_fields:
            raise RequestRejected(
                400,
                "invalid_model_chunk_payload",
                "model chunk payload fields are invalid / 模型分块载荷字段无效",
            )
        sid, round_id, update_id, total_bytes, sha256 = (
            AggregationServerService._upload_identity_from_payload(payload)
        )
        offset = payload["offset"]
        encoded_chunk = payload["chunk_base64"]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise RequestRejected(
                400,
                "invalid_model_offset",
                "offset must be non-negative / 偏移量必须非负",
            )
        if not isinstance(encoded_chunk, str) or not encoded_chunk:
            raise RequestRejected(
                400,
                "invalid_model_chunk",
                "chunk_base64 must be non-empty / chunk_base64 必须非空",
            )
        return sid, round_id, update_id, total_bytes, sha256, offset, encoded_chunk

    @staticmethod
    def _model_finalize_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, str, int, str, int, tuple[int, ...]]:
        """Validate a completed update before task commits become visible.

        在任务提交变得可见前，验证一个完成的更新。
        """
        expected_fields = {
            "sid",
            "round",
            "update_id",
            "total_bytes",
            "sha256",
            "sample_count",
            "task_ids",
        }
        if set(payload) != expected_fields:
            raise RequestRejected(
                400,
                "invalid_model_finalize_payload",
                "model finalize payload fields are invalid / 模型完成载荷字段无效",
            )
        sid, round_id, update_id, total_bytes, sha256 = (
            AggregationServerService._upload_identity_from_payload(payload)
        )
        sample_count = payload["sample_count"]
        task_ids = payload["task_ids"]
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise RequestRejected(
                400,
                "invalid_sample_count",
                "sample_count must be positive / 样本数必须为正数",
            )
        if not isinstance(task_ids, list) or not task_ids:
            raise RequestRejected(
                400,
                "invalid_task_ids",
                "task_ids must be a non-empty list / task_ids 必须为非空列表",
            )
        if any(
            isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0
            for task_id in task_ids
        ):
            raise RequestRejected(
                400,
                "invalid_task_ids",
                "task_ids must contain non-negative integers / "
                "task_ids 必须包含非负整数",
            )
        if len(set(task_ids)) != len(task_ids):
            raise RequestRejected(
                400,
                "duplicate_task_id",
                "task_ids must not repeat / task_ids 不得重复",
            )
        return sid, round_id, update_id, total_bytes, sha256, sample_count, tuple(task_ids)

    @staticmethod
    def _upload_identity_from_payload(payload: Mapping[str, Any]) -> tuple[int, int, str, int, str]:
        """Validate common safe identifiers for model-update filesystem storage.

        验证模型更新文件系统存储共用的安全标识。
        """
        sid = payload["sid"]
        round_id = payload["round"]
        update_id = payload["update_id"]
        total_bytes = payload["total_bytes"]
        sha256 = payload["sha256"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be a positive integer / SID 必须是正整数")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / 轮次必须非负")
        if (
            not isinstance(update_id, str)
            or len(update_id) != 32
            or any(character not in "0123456789abcdef" for character in update_id)
        ):
            raise RequestRejected(
                400,
                "invalid_update_id",
                "update_id must be 32 lowercase hex characters / "
                "update_id 必须为 32 位小写十六进制字符",
            )
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
            raise RequestRejected(
                400,
                "invalid_total_bytes",
                "total_bytes must be positive / 总字节数必须为正数",
            )
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RequestRejected(
                400,
                "invalid_sha256",
                "sha256 must be 64 lowercase hex characters / "
                "sha256 必须为 64 位小写十六进制字符",
            )
        return sid, round_id, update_id, total_bytes, sha256

    @staticmethod
    def _aggregate_payload(payload: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
        """Validate an explicit FedAvg participant set for one round.

        验证一个轮次显式指定的 FedAvg 参与者集合。
        """
        if set(payload) != {"round", "expected_sids"}:
            raise RequestRejected(
                400,
                "invalid_aggregate_payload",
                "aggregate payload fields are invalid / 聚合载荷字段无效",
            )
        round_id = payload["round"]
        expected_sids = payload["expected_sids"]
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / 轮次必须非负")
        if not isinstance(expected_sids, list) or not expected_sids:
            raise RequestRejected(
                400,
                "invalid_expected_sids",
                "expected_sids must be non-empty / expected_sids 必须非空",
            )
        if any(
            isinstance(sid, bool) or not isinstance(sid, int) or sid < 1
            for sid in expected_sids
        ):
            raise RequestRejected(
                400,
                "invalid_expected_sids",
                "expected_sids must be positive integers / "
                "expected_sids 必须为正整数",
            )
        if len(set(expected_sids)) != len(expected_sids):
            raise RequestRejected(
                400,
                "duplicate_expected_sid",
                "expected_sids must not repeat / expected_sids 不得重复",
            )
        return round_id, tuple(expected_sids)

    @staticmethod
    def _global_model_chunk_payload(payload: Mapping[str, Any]) -> tuple[int, int, int, int]:
        """Validate one registered client's bounded global-model read request.

        验证一个已注册客户端的有界全局模型读取请求。
        """
        if set(payload) != {"sid", "round", "offset", "max_bytes"}:
            raise RequestRejected(
                400,
                "invalid_global_model_payload",
                "global model payload fields are invalid / 全局模型载荷字段无效",
            )
        sid = payload["sid"]
        round_id = payload["round"]
        offset = payload["offset"]
        max_bytes = payload["max_bytes"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be positive / SID 必须为正数")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / 轮次必须非负")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise RequestRejected(400, "invalid_offset", "offset must be non-negative / 偏移量必须非负")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 1 <= max_bytes <= MAX_MODEL_CHUNK_BYTES
        ):
            raise RequestRejected(
                400,
                "invalid_max_bytes",
                "max_bytes is outside the chunk bound / max_bytes 超出分块上限",
            )
        return sid, round_id, offset, max_bytes

    @staticmethod
    def _round_configuration_payload(payload: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
        """Validate a fixed FedAvg roster for one not-yet-started round.

        验证一个尚未开始轮次的固定 FedAvg 名册。
        """
        if set(payload) != {"round", "participant_sids"}:
            raise RequestRejected(
                400,
                "invalid_round_configuration",
                "round configuration fields are invalid / 轮次配置字段无效",
            )
        round_id = payload["round"]
        participant_sids = payload["participant_sids"]
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / 轮次必须非负")
        if not isinstance(participant_sids, list) or not participant_sids:
            raise RequestRejected(
                400,
                "invalid_round_participants",
                "participant_sids must be non-empty / participant_sids 必须非空",
            )
        if any(
            isinstance(sid, bool) or not isinstance(sid, int) or sid < 1
            for sid in participant_sids
        ) or len(set(participant_sids)) != len(participant_sids):
            raise RequestRejected(
                400,
                "invalid_round_participants",
                "participant_sids must be unique positive integers / "
                "participant_sids 必须为唯一正整数",
            )
        return round_id, tuple(participant_sids)

    @staticmethod
    def _validated_protected_label_set(protected_labels: Any) -> tuple[str, ...]:
        """Validate one bounded, canonical, duplicate-free OPRF label set.

        验证一个有界、规范且无重复的 OPRF 标签集合。
        """
        if (
            not isinstance(protected_labels, Sequence)
            or isinstance(protected_labels, (str, bytes))
            or not 1 <= len(protected_labels) <= MAX_LABELS_PER_SUBMISSION
        ):
            raise RequestRejected(
                400,
                "invalid_label_batch",
                "protected_labels must be a non-empty bounded array / "
                "protected_labels 必须为非空且有上限的数组",
            )
        labels = tuple(protected_labels)
        if any(not isinstance(label, str) for label in labels):
            raise RequestRejected(
                400,
                "invalid_protected_label",
                "every protected label must be a string / 每个受保护标签必须是字符串",
            )
        if len(set(labels)) != len(labels):
            raise RequestRejected(
                400,
                "duplicate_protected_label",
                "protected-label set must not contain duplicates / "
                "受保护标签集合不得包含重复值",
            )
        try:
            for label in labels:
                validate_protected_label(label)
        except OprfValidationError as error:
            raise RequestRejected(
                400,
                "invalid_protected_label",
                str(error),
            ) from error
        return labels


@dataclass(frozen=True, slots=True)
class AggregationServerConfig:
    """HTTP deployment and native-index sizing parameters for the AS.

    AS 的 HTTP 部署参数与原生索引容量参数。
    """

    capacity: int
    max_clients: int
    max_edges: int
    host: str = "0.0.0.0"
    port: int = 18080
    heartbeat_interval_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 300.0
    native_library_path: str | Path | None = None
    advertised_host: str | None = None
    model_update_directory: Path = Path("results") / "model-updates"
    max_model_update_bytes: int = 2 * 1024 * 1024 * 1024
    backend_worker_count: int = 32
    claim_mode: str = "cas"
    recovery_index_mode: str = "inverse"
    history_scheduling_enabled: bool = True
    evaluation_reset_token: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid resource and heartbeat configuration before binding.

        绑定前拒绝无效的资源与心跳配置。
        """
        if not self.host:
            raise ValueError("AS host must not be empty / AS 主机不得为空")
        if not 0 <= self.port <= 65535:
            raise ValueError("AS port must be in 0..65535 / AS 端口必须位于 0..65535")
        if self.max_model_update_bytes < 1:
            raise ValueError("max_model_update_bytes must be positive / 最大模型更新字节数必须为正数")
        if self.backend_worker_count < 1:
            raise ValueError("backend_worker_count must be positive / AS 后端工作线程数必须为正数")
        if self.claim_mode not in {"cas", "mutex"}:
            raise ValueError("claim_mode must be cas or mutex / 抢占模式必须为 cas 或 mutex")
        if self.recovery_index_mode not in {"inverse", "scan"}:
            raise ValueError("recovery_index_mode must be inverse or scan / 恢复索引模式必须为 inverse 或 scan")


class AggregationServerEntity:
    """Run AS HTTP session endpoints while owning the compatible native index.

    运行 AS HTTP 会话端点，同时持有兼容的原生索引。
    """

    def __init__(self, config: AggregationServerConfig) -> None:
        """Allocate the native index and bind AS HTTP without starting it.

        分配原生索引并绑定 AS HTTP，但不启动服务。
        """
        self.config = config
        self.index = NativeIndex(
            config.capacity,
            config.max_clients,
            config.max_edges,
            library_path=config.native_library_path,
        )
        self.service = AggregationServerService(
            self.index,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            heartbeat_timeout_seconds=config.heartbeat_timeout_seconds,
            model_update_directory=config.model_update_directory,
            max_model_update_bytes=config.max_model_update_bytes,
            claim_mode=config.claim_mode,
            recovery_index_mode=config.recovery_index_mode,
            history_scheduling_enabled=config.history_scheduling_enabled,
            evaluation_reset_token=config.evaluation_reset_token,
            evaluation_reset_callback=self._reset_for_evaluation,
        )
        self._server = ThreadedJsonServer(
            config.host,
            config.port,
            self.service.router,
            max_concurrent_requests=config.backend_worker_count,
            control_paths=(
                AggregationServerPath.HEARTBEAT.value,
                AggregationServerPath.METRICS.value,
            ),
            advertised_host=config.advertised_host,
        )
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._closed = False

    @property
    def base_url(self) -> str:
        """Return the HTTP AS endpoint. / 返回 HTTP AS 端点。"""
        return self._server.base_url

    @property
    def port(self) -> int:
        """Return the actual AS port after binding. / 返回绑定后的实际 AS 端口。"""
        return self._server.port

    def start(self) -> None:
        """Start AS HTTP and its independent heartbeat timeout monitor.

        启动 AS HTTP 以及独立的心跳超时监控器。
        """
        if self._closed:
            raise RuntimeError("AS entity is closed / AS 实体已关闭")
        self._server.start()
        if self._monitor_thread is None:
            monitor_period = max(
                0.01,
                min(
                    self.config.heartbeat_interval_seconds / 2,
                    self.config.heartbeat_timeout_seconds / 2,
                ),
            )
            self._monitor_thread = threading.Thread(
                target=self._monitor_sessions,
                args=(monitor_period,),
                daemon=True,
            )
            self._monitor_thread.start()

    def _reset_for_evaluation(
        self,
        backend_worker_count: int,
        heartbeat_interval_seconds: float,
        heartbeat_timeout_seconds: float,
    ) -> None:
        """Replace the complete native index only at an authorized idle boundary.

        仅在已授权的空闲边界替换完整原生索引并配置下一用例的租约。
        """
        replacement = NativeIndex(
            self.config.capacity,
            self.config.max_clients,
            self.config.max_edges,
            library_path=self.config.native_library_path,
        )
        with self.service._lock:
            previous = self.index
            self.index = replacement
            self.service.index = replacement
            self.service.heartbeat_interval_seconds = heartbeat_interval_seconds
            self.service.heartbeat_timeout_seconds = heartbeat_timeout_seconds
            self.service._clear_evaluation_state()
            self._server.set_max_concurrent_requests(backend_worker_count)
        previous.close()

    def session_snapshot(self, sid: int) -> ClientSessionSnapshot | None:
        """Return AS-observed liveness for one SID. / 返回 AS 观察到的一个 SID 的存活状态。"""
        return self.service.session_snapshot(sid)

    def close(self) -> None:
        """Stop AS components and release the native index exactly once.

        停止 AS 组件并恰好一次释放原生索引。
        """
        if self._closed:
            return
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)
        self._server.close()
        self.index.close()
        self._closed = True

    def __enter__(self) -> "AggregationServerEntity":
        """Start the AS entity at context entry. / 上下文进入时启动 AS 实体。"""
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Close the AS entity at context exit. / 上下文退出时关闭 AS 实体。"""
        self.close()

    def _monitor_sessions(self, monitor_period: float) -> None:
        """Run timeout detection without interfering with request processing.

        运行超时检测，且不干扰请求处理。
        """
        while not self._monitor_stop.wait(monitor_period):
            self.service.expire_sessions()
