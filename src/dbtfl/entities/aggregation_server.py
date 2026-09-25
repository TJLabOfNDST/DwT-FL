'Aggregation Server session management compatible with the DwT-FL index.'

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
    'Current AS-side liveness state for one globally unique SID.'

    sid: int
    client_id: str
    online: bool
    heartbeat_count: int
    recovery_risk: bool


@dataclass(frozen=True, slots=True)
class GlobalModelDescriptor:
    'AS-local global checkpoint metadata available for a completed round.'

    round_id: int
    checkpoint_path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class RoundConfiguration:
    'Fixed FedAvg roster selected before client updates arrive.'

    round_id: int
    participant_sids: tuple[int, ...]


@dataclass(slots=True)
class _ClientSession:
    'Mutable session timer state private to the AS service.\n    AS'

    sid: int
    client_id: str
    last_heartbeat_seconds: float
    online: bool = True
    heartbeat_count: int = 0
    recovery_risk: bool = False

    def snapshot(self) -> ClientSessionSnapshot:
        'Copy public liveness information without exposing timer internals.'
        return ClientSessionSnapshot(
            sid=self.sid,
            client_id=self.client_id,
            online=self.online,
            heartbeat_count=self.heartbeat_count,
            recovery_risk=self.recovery_risk,
        )


class AggregationServerService:
    'Issue SIDs and maintain heartbeat timers alongside the native index.'

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
        'Create session routes without exposing index internals over HTTP.'
        if heartbeat_interval_seconds <= 0 or heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat values must be positive / ")
        if heartbeat_timeout_seconds <= heartbeat_interval_seconds:
            raise ValueError(
                "heartbeat timeout must exceed interval / "
            )
        if claim_mode not in {"cas", "mutex"}:
            raise ValueError("claim_mode must be cas or mutex /  cas  mutex")
        if recovery_index_mode not in {"inverse", "scan"}:
            raise ValueError("recovery_index_mode must be inverse or scan /  inverse  scan")
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
        'Issue an unused SID or reconnect a known client with its same SID.'
        if message.message_type != AS_REGISTER_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.client.register.request /  as.client.register.request",
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
                        "AS has no remaining SID capacity / AS  SID ",
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
        'Refresh one known SID timer and confirm its active connection.'
        if message.message_type != AS_HEARTBEAT_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.client.heartbeat.request /  as.client.heartbeat.request",
            )
        sid = self._sid_from_payload(message.payload)
        now = self._clock()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(
                    404,
                    "unknown_sid",
                    "SID has not been registered / SID ",
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
        'Append both directions while leaving every new task in EMPTY state.'
        if message.message_type != AS_REGISTER_LABELS_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.labels.register.request /  as.labels.register.request",
            )
        sid, created_round, labels = self._label_submission_from_payload(message.payload)
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(
                    404,
                    "unknown_sid",
                    "SID has not been registered / SID ",
                )
            if not session.online:
                raise RequestRejected(
                    409,
                    "sid_offline",
                    "offline SID must reconnect before submitting labels / "
                    " SID ",
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
                    "AS ",
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
        'Perform the separate paper CAS phase for already registered labels.'
        if message.message_type != AS_CLAIM_TASKS_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.tasks.claim.request /  as.tasks.claim.request",
            )
        sid, labels = self._claim_payload_from_message(message.payload)
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(404, "unknown_sid", "SID has not been registered / SID ")
            if not session.online:
                raise RequestRejected(
                    409,
                    "sid_offline",
                    "offline SID must reconnect before claiming tasks / "
                    " SID ",
                )

        decisions: list[dict[str, object]] = []
        for protected_label in labels:
            task_id = self.index.find_label(protected_label)
            if task_id is None:
                raise RequestRejected(
                    404,
                    "unknown_protected_label",
                    "protected label has not been registered / ",
                )
            if not self.index.task_has_owner(task_id, sid):
                raise RequestRejected(
                    403,
                    "sid_not_label_owner",
                    "SID does not own this protected label / SID ",
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
                # deduplication loss; a different SID remains DEDUP.
                
                
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
        'Claim with production CAS or the explicit pessimistic-lock ablation.\n        The mutex mode serializes the entire claim decision before calling the\n        same native state transition. It is intentionally a pessimistic control\n        not an alternative result mislabeled as lock-free CAS. ``mutex``'
        if self.claim_mode == "mutex":
            with self._mutex_claim_lock:
                return self.index.try_claim(task_id, sid)
        return self.index.try_claim(task_id, sid)

    def model_update(self, message: WireMessage) -> WireMessage:
        'Receive one bounded checkpoint chunk or finalize a client update.'
        if message.message_type == AS_MODEL_CHUNK_REQUEST:
            return self._append_model_chunk(message)
        if message.message_type == AS_MODEL_FINALIZE_REQUEST:
            return self._finalize_model_update(message)
        raise RequestRejected(
            400,
            "unexpected_message_type",
            "expected model chunk or finalize request / ",
        )

    def _round_instructions_for_sid(self, sid: int) -> list[dict[str, object]]:
        'Issue idempotent next-round TRAIN or DEDUP work on a heartbeat.'
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
        "Choose the next-round trainer under the configured history policy.\n        When history scheduling is enabled, a client that timed out in an\n        earlier round remains marked as ``recovery_risk`` after it reconnects.\n        The scheduler excludes that client from duplicated tasks whenever a\n        healthy online owner exists, as required by the paper's training-right\n        allocation adjustment.  The explicit ``w/o history scheduling``\n        ablation deliberately does not consume this historical risk record\n        it follows the ordinary stable-case rule and reuses the previous online\n        trainer.\n        ``recovery_risk``"
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
        'Apply FedAvg to the requested complete client-update set.'
        if message.message_type != AS_MODEL_AGGREGATE_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.model.aggregate.request /  as.model.aggregate.request",
            )
        round_id, expected_sids = self._aggregate_payload(message.payload)
        with self._lock:
            updates = self._updates_by_round.get(round_id, {})
            missing_sids = sorted(set(expected_sids).difference(updates))
            if missing_sids:
                raise RequestRejected(
                    409,
                    "model_updates_incomplete",
                    f"missing model updates for SIDs {missing_sids} /  SID {missing_sids} ",
                )
            descriptors = tuple(updates[sid] for sid in expected_sids)
            configured = self._round_configurations.get(round_id)
            if configured is not None and configured.participant_sids != expected_sids:
                raise RequestRejected(
                    409,
                    "round_roster_mismatch",
                    "FedAvg participants differ from the configured round roster / "
                    "FedAvg ",
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
                    " PENDING ",
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
                f"FedAvg could not aggregate updates: {error} / FedAvg {error}",
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
        'Return one bounded global-checkpoint chunk to an online client.'
        if message.message_type != AS_GLOBAL_MODEL_CHUNK_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.global_model.chunk.request /  as.global_model.chunk.request",
            )
        sid, round_id, offset, max_bytes = self._global_model_chunk_payload(message.payload)
        self._require_online_sid(
            sid,
            operation=("downloading the global model", ""),
        )
        with self._lock:
            descriptor = self._global_models.get(round_id)
        if descriptor is None or not descriptor.checkpoint_path.is_file():
            raise RequestRejected(
                404,
                "global_model_not_found",
                "global model is not available for this round / ",
            )
        if offset > descriptor.byte_count:
            raise RequestRejected(
                416,
                "invalid_global_model_offset",
                "offset exceeds global model size / ",
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
        'Freeze an explicit FedAvg roster so clients may join sequentially.'
        if message.message_type != AS_CONFIGURE_ROUND_REQUEST:
            raise RequestRejected(
                400,
                "unexpected_message_type",
                "expected as.round.configure.request /  as.round.configure.request",
            )
        round_id, participant_sids = self._round_configuration_payload(message.payload)
        with self._lock:
            if any(sid not in self._sessions_by_sid for sid in participant_sids):
                raise RequestRejected(
                    404,
                    "unknown_round_participant",
                    "every configured SID must be registered /  SID ",
                )
            if self._updates_by_round.get(round_id):
                raise RequestRejected(
                    409,
                    "round_already_started",
                    "cannot change roster after an update arrives / ",
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
            # incremental training and replaces its update before FedAvg.
            # FedAvg
            # EMPTY
            self._round_dispatch_sealed = False
        return WireMessage.create(
            AS_CONFIGURE_ROUND_RESPONSE,
            {"round": round_id, "participant_sids": list(participant_sids)},
            request_id=message.request_id,
        )

    def evaluation_metrics(self, message: WireMessage) -> WireMessage:
        'Return read-only AS metadata required by the experiment protocol.\n        This endpoint intentionally exposes capacities and byte counts only; it\n        never returns protected labels, plaintext records, owners, or model\n        content.'
        if message.message_type != AS_METRICS_REQUEST or dict(message.payload):
            raise RequestRejected(
                400,
                "invalid_metrics_request",
                "metrics request must have the expected type and an empty payload / "
                "",
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
        'Reset a dedicated experimental AS after constant-time token validation.\n        This route is disabled unless deployment config supplies a token. It\n        clears all in-memory sessions, indexes, and AS-owned update artifacts\n        never enable it on a shared production AS.'
        if message.message_type != AS_EVALUATION_RESET_REQUEST or set(message.payload) != {
            "token", "backend_worker_count", "heartbeat_interval_seconds",
            "heartbeat_timeout_seconds",
        }:
            raise RequestRejected(
                400,
                "invalid_evaluation_reset_request",
                "reset request must contain token, backend workers, and heartbeat lease values / "
                "",
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
                "backend worker count must be positive / ",
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
                "",
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
                "evaluation reset is disabled or the token is invalid / ",
            )
        # The evaluator changes this only at a destructive, token-protected,
        # case boundary. It separates a normal long GPT-training lease from the
        # deliberately short dropout-recovery lease.
        
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
        'Change only the lease of a dedicated experiment without resetting state.\n        A training-dropout measurement must finish OPRF, label registration, and\n        CAS under the normal long lease before it activates a short failure\n        lease.  Resetting here would erase the exact task ownership that the\n        measurement must recover, so this route changes no index, SID, model\n        or round state.\n        CAS'
        if message.message_type != AS_EVALUATION_LEASE_REQUEST or set(message.payload) != {
            "token", "heartbeat_timeout_seconds",
        }:
            raise RequestRejected(
                400,
                "invalid_evaluation_lease_request",
                "lease request must contain token and heartbeat timeout / "
                "",
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
                "",
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
                "",
            )
        with self._lock:
            # The lease switch is one atomic server-side event. Refresh every
            # currently online SID at the same instant before installing the
            # shorter timeout; otherwise the evaluator has to create a burst
            # of client heartbeats and an otherwise healthy SID can expire
            # between the last refresh and this control request.
            
            
            
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
        'Clear service state while the entity owns an exclusive reset boundary.'
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
        "Return this AS process's current resource observation when available."
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
        'Retain successful trainers, reset index state, and enable heartbeats.'
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
        'Decode and persist one ordered bounded base64 checkpoint chunk.'
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
                "chunk must be valid base64 /  Base64",
            ) from error
        if len(chunk) > MAX_MODEL_CHUNK_BYTES:
            raise RequestRejected(
                413,
                "model_chunk_too_large",
                "decoded model chunk exceeds limit / ",
            )
        if total_bytes > self.model_update_store.max_update_bytes:
            raise RequestRejected(
                413,
                "model_update_too_large",
                "model update exceeds configured limit / ",
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
        'Publish one verified update and commit all tasks trained by its SID.'
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
                        "",
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
                    # exposed.
                    
                    raise RequestRejected(
                        409,
                        "task_not_pending_for_sid",
                        "all tasks must remain PENDING for the submitting SID / "
                        " SID  PENDING ",
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
                        " SID ",
                    )
            # FedAvg weights are authoritative AS state, not a client-declared
            # dataset size.  A replacement update retains earlier COMMITTED
            # tasks, while a first update has just committed its PENDING tasks;
            # both cases are counted from the state table after this transition.
            # FedAvg
            
            
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
                    " SID  COMMITTED ",
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
        'Describe only labels already unusable by the rejected trainer.\n        A released ``EMPTY`` task is deliberately omitted: it has not yet been\n        taken over, so the reconnecting client may legally win a later CAS. A\n        ``PENDING`` task belongs in this payload only when another currently\n        online SID owns it. ``COMMITTED`` work is also immutable and therefore\n        returned as DEDUP.'
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
        'Return liveness without revealing session data outside this service.'
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            return session is not None and session.online

    @staticmethod
    def _finalize_response(descriptor: ModelUpdateDescriptor, request_id: str) -> WireMessage:
        'Return an idempotent model-update acknowledgement.'
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
        operation: tuple[str, str] = ("uploading", ""),
    ) -> None:
        'Ensure the requested operation is attributed to an online SID.\n        ``operation`` contains English and Chinese response text only; it makes a rejected immutable\n        download diagnosable without changing the shared SID state machine.\n        ``operation``'
        operation_english, operation_chinese = operation
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            if session is None:
                raise RequestRejected(404, "unknown_sid", "SID has not been registered / SID ")
            if not session.online:
                raise RequestRejected(
                    409,
                    "sid_offline",
                    f"offline SID must reconnect before {operation_english} / "
                    f" SID {operation_chinese}",
                )

    def expire_sessions(self) -> tuple[int, ...]:
        'Mark timers older than tau as offline and return newly expired SIDs.'
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
        "Release one dropped SID's tasks through index or scan recovery.\n        ``scan`` is the explicit ``w/o inverse index`` ablation: it deliberately\n        traverses every task and preserves the same state semantics. The native\n        inverse table remains allocated for ABI compatibility but is not read by\n        this recovery path. ``scan``"
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
        'Return one current session snapshot after applying timeout detection.'
        self.expire_sessions()
        with self._lock:
            session = self._sessions_by_sid.get(sid)
            return None if session is None else session.snapshot()

    @staticmethod
    def _client_id_from_payload(payload: Mapping[str, Any]) -> str:
        'Validate the one-field client registration payload.'
        if set(payload) != {"client_id"}:
            raise RequestRejected(
                400,
                "invalid_registration_payload",
                "payload must contain only client_id /  client_id",
            )
        client_id = payload["client_id"]
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise RequestRejected(
                400,
                "invalid_client_id",
                "client_id must contain 1..128 characters / client_id  1..128 ",
            )
        return client_id

    @staticmethod
    def _sid_from_payload(payload: Mapping[str, Any]) -> int:
        'Validate the one-field heartbeat payload.'
        if set(payload) != {"sid"}:
            raise RequestRejected(
                400,
                "invalid_heartbeat_payload",
                "payload must contain only sid /  sid",
            )
        sid = payload["sid"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(
                400,
                "invalid_sid",
                "sid must be a positive integer / sid ",
            )
        return sid

    @staticmethod
    def _label_submission_from_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, tuple[str, ...]]:
        'Validate a complete SID, round, and protected-label-set submission.'
        if set(payload) != {"sid", "round", "protected_labels"}:
            raise RequestRejected(
                400,
                "invalid_label_payload",
                "payload must contain sid, round, and protected_labels / "
                " sidround  protected_labels",
            )
        sid = payload["sid"]
        created_round = payload["round"]
        protected_labels = payload["protected_labels"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be a positive integer / sid ")
        if (
            isinstance(created_round, bool)
            or not isinstance(created_round, int)
            or not 0 <= created_round <= 0xFFFFFFFF
        ):
            raise RequestRejected(
                400,
                "invalid_round",
                "round must fit uint32 / round  uint32",
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
                "protected_labels ",
            )
        labels = tuple(protected_labels)
        if any(not isinstance(label, str) for label in labels):
            raise RequestRejected(
                400,
                "invalid_protected_label",
                "every protected label must be a string / ",
            )
        if len(set(labels)) != len(labels):
            raise RequestRejected(
                400,
                "duplicate_protected_label",
                "protected-label set must not contain duplicates / "
                "",
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
        'Validate one claim request without re-registering its label set.'
        if set(payload) != {"sid", "protected_labels"}:
            raise RequestRejected(
                400,
                "invalid_claim_payload",
                "payload must contain sid and protected_labels / "
                " sid  protected_labels",
            )
        sid = payload["sid"]
        protected_labels = payload["protected_labels"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be a positive integer / sid ")
        labels = AggregationServerService._validated_protected_label_set(protected_labels)
        return sid, labels

    @staticmethod
    def _model_chunk_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, str, int, str, int, str]:
        'Validate metadata for one bounded base64 model-update chunk.'
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
                "model chunk payload fields are invalid / ",
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
                "offset must be non-negative / ",
            )
        if not isinstance(encoded_chunk, str) or not encoded_chunk:
            raise RequestRejected(
                400,
                "invalid_model_chunk",
                "chunk_base64 must be non-empty / chunk_base64 ",
            )
        return sid, round_id, update_id, total_bytes, sha256, offset, encoded_chunk

    @staticmethod
    def _model_finalize_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int, int, str, int, str, int, tuple[int, ...]]:
        'Validate a completed update before task commits become visible.'
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
                "model finalize payload fields are invalid / ",
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
                "sample_count must be positive / ",
            )
        if not isinstance(task_ids, list) or not task_ids:
            raise RequestRejected(
                400,
                "invalid_task_ids",
                "task_ids must be a non-empty list / task_ids ",
            )
        if any(
            isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0
            for task_id in task_ids
        ):
            raise RequestRejected(
                400,
                "invalid_task_ids",
                "task_ids must contain non-negative integers / "
                "task_ids ",
            )
        if len(set(task_ids)) != len(task_ids):
            raise RequestRejected(
                400,
                "duplicate_task_id",
                "task_ids must not repeat / task_ids ",
            )
        return sid, round_id, update_id, total_bytes, sha256, sample_count, tuple(task_ids)

    @staticmethod
    def _upload_identity_from_payload(payload: Mapping[str, Any]) -> tuple[int, int, str, int, str]:
        'Validate common safe identifiers for model-update filesystem storage.'
        sid = payload["sid"]
        round_id = payload["round"]
        update_id = payload["update_id"]
        total_bytes = payload["total_bytes"]
        sha256 = payload["sha256"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be a positive integer / SID ")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / ")
        if (
            not isinstance(update_id, str)
            or len(update_id) != 32
            or any(character not in "0123456789abcdef" for character in update_id)
        ):
            raise RequestRejected(
                400,
                "invalid_update_id",
                "update_id must be 32 lowercase hex characters / "
                "update_id  32 ",
            )
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
            raise RequestRejected(
                400,
                "invalid_total_bytes",
                "total_bytes must be positive / ",
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
                "sha256  64 ",
            )
        return sid, round_id, update_id, total_bytes, sha256

    @staticmethod
    def _aggregate_payload(payload: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
        'Validate an explicit FedAvg participant set for one round.'
        if set(payload) != {"round", "expected_sids"}:
            raise RequestRejected(
                400,
                "invalid_aggregate_payload",
                "aggregate payload fields are invalid / ",
            )
        round_id = payload["round"]
        expected_sids = payload["expected_sids"]
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / ")
        if not isinstance(expected_sids, list) or not expected_sids:
            raise RequestRejected(
                400,
                "invalid_expected_sids",
                "expected_sids must be non-empty / expected_sids ",
            )
        if any(
            isinstance(sid, bool) or not isinstance(sid, int) or sid < 1
            for sid in expected_sids
        ):
            raise RequestRejected(
                400,
                "invalid_expected_sids",
                "expected_sids must be positive integers / "
                "expected_sids ",
            )
        if len(set(expected_sids)) != len(expected_sids):
            raise RequestRejected(
                400,
                "duplicate_expected_sid",
                "expected_sids must not repeat / expected_sids ",
            )
        return round_id, tuple(expected_sids)

    @staticmethod
    def _global_model_chunk_payload(payload: Mapping[str, Any]) -> tuple[int, int, int, int]:
        "Validate one registered client's bounded global-model read request."
        if set(payload) != {"sid", "round", "offset", "max_bytes"}:
            raise RequestRejected(
                400,
                "invalid_global_model_payload",
                "global model payload fields are invalid / ",
            )
        sid = payload["sid"]
        round_id = payload["round"]
        offset = payload["offset"]
        max_bytes = payload["max_bytes"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise RequestRejected(400, "invalid_sid", "sid must be positive / SID ")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / ")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise RequestRejected(400, "invalid_offset", "offset must be non-negative / ")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 1 <= max_bytes <= MAX_MODEL_CHUNK_BYTES
        ):
            raise RequestRejected(
                400,
                "invalid_max_bytes",
                "max_bytes is outside the chunk bound / max_bytes ",
            )
        return sid, round_id, offset, max_bytes

    @staticmethod
    def _round_configuration_payload(payload: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
        'Validate a fixed FedAvg roster for one not-yet-started round.'
        if set(payload) != {"round", "participant_sids"}:
            raise RequestRejected(
                400,
                "invalid_round_configuration",
                "round configuration fields are invalid / ",
            )
        round_id = payload["round"]
        participant_sids = payload["participant_sids"]
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RequestRejected(400, "invalid_round", "round must be non-negative / ")
        if not isinstance(participant_sids, list) or not participant_sids:
            raise RequestRejected(
                400,
                "invalid_round_participants",
                "participant_sids must be non-empty / participant_sids ",
            )
        if any(
            isinstance(sid, bool) or not isinstance(sid, int) or sid < 1
            for sid in participant_sids
        ) or len(set(participant_sids)) != len(participant_sids):
            raise RequestRejected(
                400,
                "invalid_round_participants",
                "participant_sids must be unique positive integers / "
                "participant_sids ",
            )
        return round_id, tuple(participant_sids)

    @staticmethod
    def _validated_protected_label_set(protected_labels: Any) -> tuple[str, ...]:
        'Validate one bounded, canonical, duplicate-free OPRF label set.'
        if (
            not isinstance(protected_labels, Sequence)
            or isinstance(protected_labels, (str, bytes))
            or not 1 <= len(protected_labels) <= MAX_LABELS_PER_SUBMISSION
        ):
            raise RequestRejected(
                400,
                "invalid_label_batch",
                "protected_labels must be a non-empty bounded array / "
                "protected_labels ",
            )
        labels = tuple(protected_labels)
        if any(not isinstance(label, str) for label in labels):
            raise RequestRejected(
                400,
                "invalid_protected_label",
                "every protected label must be a string / ",
            )
        if len(set(labels)) != len(labels):
            raise RequestRejected(
                400,
                "duplicate_protected_label",
                "protected-label set must not contain duplicates / "
                "",
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
    'HTTP deployment and native-index sizing parameters for the AS.\n    AS'

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
        'Reject invalid resource and heartbeat configuration before binding.'
        if not self.host:
            raise ValueError("AS host must not be empty / AS ")
        if not 0 <= self.port <= 65535:
            raise ValueError("AS port must be in 0..65535 / AS  0..65535")
        if self.max_model_update_bytes < 1:
            raise ValueError("max_model_update_bytes must be positive / ")
        if self.backend_worker_count < 1:
            raise ValueError("backend_worker_count must be positive / AS ")
        if self.claim_mode not in {"cas", "mutex"}:
            raise ValueError("claim_mode must be cas or mutex /  cas  mutex")
        if self.recovery_index_mode not in {"inverse", "scan"}:
            raise ValueError("recovery_index_mode must be inverse or scan /  inverse  scan")


class AggregationServerEntity:
    'Run AS HTTP session endpoints while owning the compatible native index.'

    def __init__(self, config: AggregationServerConfig) -> None:
        'Allocate the native index and bind AS HTTP without starting it.'
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
        'Return the HTTP AS endpoint.'
        return self._server.base_url

    @property
    def port(self) -> int:
        'Return the actual AS port after binding.'
        return self._server.port

    def start(self) -> None:
        'Start AS HTTP and its independent heartbeat timeout monitor.'
        if self._closed:
            raise RuntimeError("AS entity is closed / AS ")
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
        'Replace the complete native index only at an authorized idle boundary.'
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
        'Return AS-observed liveness for one SID.'
        return self.service.session_snapshot(sid)

    def close(self) -> None:
        'Stop AS components and release the native index exactly once.'
        if self._closed:
            return
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)
        self._server.close()
        self.index.close()
        self._closed = True

    def __enter__(self) -> "AggregationServerEntity":
        'Start the AS entity at context entry.'
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        'Close the AS entity at context exit.'
        self.close()

    def _monitor_sessions(self, monitor_period: float) -> None:
        'Run timeout detection without interfering with request processing.'
        while not self._monitor_stop.wait(monitor_period):
            self.service.expire_sessions()
