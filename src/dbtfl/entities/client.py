'Client entity that persists local record--protected-label correspondences.'

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
"""Version for the client-private label store. / """

MODEL_CHUNK_TRANSPORT_ATTEMPTS: Final[int] = 3
"""Bounded attempts for one idempotent chunk request. / """

MODEL_CHUNK_RETRY_DELAY_SECONDS: Final[float] = 0.5
"""Initial backoff for one lost model-chunk response. / """

GLOBAL_MODEL_CHUNK_TRANSPORT_ATTEMPTS: Final[int] = 4
"""Bounded attempts for one idempotent global-model read. / """

GLOBAL_MODEL_CHUNK_RETRY_DELAY_SECONDS: Final[float] = 0.25
"""Initial backoff for a lost global-model chunk response. / """

MODEL_HASH_READ_BYTES: Final[int] = 1024 * 1024
"""Streaming read size for checkpoint hashing. / """

AS_OFFLINE_RECOVERY_INITIAL_DELAY_SECONDS: Final[float] = 0.05
"""Initial reconnect backoff after an AS offline response. / AS """

AS_OFFLINE_RECOVERY_MAX_DELAY_SECONDS: Final[float] = 1.0
"""Maximum reconnect backoff that prevents a hot retry loop. / """

AS_REGISTRATION_TRANSPORT_ATTEMPTS: Final[int] = 4
"""Attempts for a pre-SID idempotent registration request. / SID """

AS_REGISTRATION_RETRY_DELAY_SECONDS: Final[float] = 0.05
"""Initial backoff for a transient pre-SID registration failure. / SID """

@dataclass(frozen=True, slots=True)
class ClientAsSession:
    'Client-side AS session assigned during registration.'

    sid: int
    heartbeat_interval_seconds: float


@dataclass(frozen=True, slots=True)
class RoundInstruction:
    'One AS heartbeat instruction for the active or recovered round.'

    protected_label: str
    task_id: int
    operation: str


@dataclass(frozen=True, slots=True)
class LabelRegistration:
    'One AS index-registration acknowledgement for a protected label.\n    AS'

    protected_label: str
    task_id: int


@dataclass(frozen=True, slots=True)
class TaskClaimDecision:
    'One separate CAS-phase decision returned by the AS.\n    AS'

    protected_label: str
    task_id: int
    operation: str
    state: str


@dataclass(frozen=True, slots=True)
class LocalTrainingQueues:
    'Private plaintext records routed by paper TRAIN and DEDUP decisions.'

    hot_records: tuple[bytes, ...]
    cold_records: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class ModelUpdateSubmission:
    'AS acknowledgement for one completed client model-update submission.\n    AS'

    round_id: int
    sid: int
    update_id: str
    committed_task_ids: tuple[int, ...]
    checkpoint_sha256: str


class ModelUpdateOwnershipLostError(RuntimeError):
    'A recovered upload no longer owns every task used for local training.'

    def __init__(
        self,
        message: str,
        *,
        dedup_instructions: Sequence[TaskClaimDecision] = (),
        recovered_decisions: Sequence[TaskClaimDecision] = (),
    ) -> None:
        'Retain paper-level recovery instructions with the rejection.\n        ``dedup_instructions`` originates in the AS 409 payload and identifies\n        labels safely taken by another trainer. ``recovered_decisions`` records\n        a later client-side CAS observation when the loss was found while\n        reconnecting.'
        # Do not use zero-argument ``super()`` in this protocol exception.
        # The exception is deliberately raised from recovery callbacks that may
        # be wrapped by test doubles or subprocess boundaries; explicitly
        # initializing RuntimeError keeps the wire-recovery error constructible
        # in every supported Python runtime.
        # ``super()``
        
        
        RuntimeError.__init__(self, message)
        self.dedup_instructions = tuple(dedup_instructions)
        self.recovered_decisions = tuple(recovered_decisions)


class ClientLabelStoreError(RuntimeError):
    'Raised when the client-private record-label store is invalid or unsafe.'


@dataclass(frozen=True, slots=True)
class ClientConfig:
    'Client identity, KS endpoint, and private label-store configuration.'

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
        'Reject incomplete client configuration before any RPC is attempted.'
        if not self.client_id:
            raise ValueError("client_id must not be empty / client_id ")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive / timeout_seconds ")
        if self.heartbeat_rpc_timeout_seconds <= 0:
            raise ValueError(
                "heartbeat_rpc_timeout_seconds must be positive / "
                "heartbeat_rpc_timeout_seconds "
            )
        if self.registration_rpc_timeout_seconds <= 0:
            raise ValueError(
                "registration_rpc_timeout_seconds must be positive / "
                "registration_rpc_timeout_seconds "
            )
        if not 1 <= self.oprf_batch_size <= MAX_BATCH_ELEMENTS:
            raise ValueError(
                f"oprf_batch_size must be in 1..{MAX_BATCH_ELEMENTS} / "
                f"oprf_batch_size  1..{MAX_BATCH_ELEMENTS}"
            )
        if self.oprf_suite != OPRF_SUITE_IDENTIFIER:
            raise ValueError(
                "client must use the active Ristretto255 OPRF suite / "
                " Ristretto255 OPRF "
            )
        if self.model_chunk_bytes < 1 or self.model_chunk_bytes > 2 * 1024 * 1024:
            raise ValueError("model_chunk_bytes must be in 1..2097152 /  1..2097152")


class ClientEntity:
    'Generate and remember protected labels while retaining raw records locally.\n    The store encodes records with Base64 for lossless JSON storage.  Base64 is\n    not encryption; deploy it on a private client filesystem with OS access\n    controls.'

    def __init__(self, config: ClientConfig) -> None:
        'Load existing local correspondences without contacting the KS.'
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
        
        
        self._heartbeat_request_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_heartbeat_error: CommunicationError | None = None
        self._active_round: int | None = None
        self._last_round_instructions: tuple[RoundInstruction, ...] = ()

    @property
    def record_count(self) -> int:
        'Return the number of locally stored record-label associations.'
        with self._lock:
            return len(self._labels_by_record)

    @property
    def as_session(self) -> ClientAsSession | None:
        'Return the current in-memory AS session, if connected.'
        with self._lock:
            return self._as_session

    @property
    def last_heartbeat_error(self) -> CommunicationError | None:
        'Return the latest background heartbeat error without raising it.'
        with self._lock:
            return self._last_heartbeat_error

    @property
    def active_round(self) -> int | None:
        'Return the latest active round announced by an AS heartbeat.'
        with self._lock:
            return self._active_round

    @property
    def last_round_instructions(self) -> tuple[RoundInstruction, ...]:
        'Return the latest AS instructions without exposing mutable state.'
        with self._lock:
            return self._last_round_instructions

    def connect_to_as(self) -> ClientAsSession:
        'Register this client, receive its SID, and start periodic heartbeats.'
        if self._as_registration_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
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
        'Synchronously refresh the AS timer for the registered SID.'
        session, _instructions = self.send_as_heartbeat_with_instructions()
        return session

    def send_as_heartbeat_with_instructions(
        self,
    ) -> tuple[ClientAsSession, tuple[RoundInstruction, ...]]:
        'Refresh an SID and return the immutable instructions from this response.\n        The periodic heartbeat worker may finish another request immediately\n        before or after this call. Recovery code must therefore use the\n        returned snapshot rather than rereading ``last_round_instructions``\n        the latter is deliberately only a latest-state convenience view.'
        if self._as_heartbeat_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        with self._heartbeat_request_lock:
            with self._lock:
                session = self._as_session
            if session is None:
                raise RuntimeError("client is not registered with AS /  AS")
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
        'Route the latest heartbeat work without treating DEDUP data as input.'
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
        'Download and SHA-256-verify one AS global checkpoint in chunks.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / round_id ")
        # A real GPT round can take substantially longer than the short lease
        # deliberately used by the training-fault evaluation. Refresh liveness
        # at the read boundary, after all training and aggregation work has
        # completed.
        
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
                    
                    # SHA-256
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
                        # AS
                        
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
                        raise TransportError("global model changed during download / ")
                    stream.write(chunk)
                    offset += len(chunk)
                    if complete:
                        break
                    if not chunk:
                        raise TransportError("global model download made no progress / ")
            if (
                descriptor is None
                or offset != descriptor[0]
                or sha256_file(Path(temporary_name)) != descriptor[1]
            ):
                raise TransportError("global model digest verification failed / ")
            Path(temporary_name).replace(destination)
            return destination
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def _refresh_global_model_read_session(self, recovery_deadline: float) -> ClientAsSession:
        'Synchronously refresh or safely restore a read-only model session.\n        This helper is intentionally restricted to immutable global-model\n        distribution. It must never be reused by the stateful checkpoint-upload\n        state machine, whose recovery has to revalidate task ownership first.'
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
        'Reconnect an expired SID while a SHA-256-verified model read is active.'
        if time.monotonic() >= recovery_deadline:
            raise offline_error
        return self.connect_to_as()

    @staticmethod
    def _is_offline_sid_error(error: RemoteServiceError) -> bool:
        'Identify the explicit AS precondition used for reconnect recovery.'
        return error.status_code == 409 and error.code == "sid_offline"

    @staticmethod
    def _global_model_liveness_refresh_period(session: ClientAsSession) -> float:
        'Use half the advertised heartbeat period, with a small safe floor.'
        return max(0.01, session.heartbeat_interval_seconds / 2.0)

    def configure_federated_round_at_as(
        self,
        round_id: int,
        participant_sids: Sequence[int],
    ) -> tuple[int, ...]:
        'Freeze the explicit FedAvg roster before any client uploads.\n        The evaluator uses this public method instead of reaching into the\n        transport implementation, so a real multi-round experiment exercises\n        the same client--AS wire contract as deployment code.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / ")
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
                                 "AS ")
        payload = dict(response.payload)
        if payload != {"round": round_id, "participant_sids": list(normalized_sids)}:
            raise TransportError("AS round configuration acknowledgement does not match / "
                                 "AS ")
        return normalized_sids

    def aggregate_federated_round_at_as(
        self,
        round_id: int,
        participant_sids: Sequence[int],
    ) -> dict[str, object]:
        'Request FedAvg for the previously frozen complete roster.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / ")
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
                                 "AS  FedAvg ")
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
            raise TransportError("AS FedAvg acknowledgement is invalid / AS FedAvg ")
        return payload

    def register_records_with_as(
        self,
        records: Sequence[str | bytes],
        *,
        created_round: int = 0,
        refresh_lease: bool = True,
    ) -> list[LabelRegistration]:
        'Generate labels, then register the complete set at AS.'
        protected_labels = self.generate_protected_labels(records)
        if refresh_lease:
            # Real OPRF batches can take longer than an AS heartbeat interval
            # on a CPU-bound client. Refresh synchronously before the AS-only
            # registration phase so a delayed background thread cannot leave
            # the SID offline.
            
            # SID
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
        'Register one canonical OPRF label set without claiming training.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS /  AS")
        labels = self._validated_submission_labels(protected_labels)
        if isinstance(created_round, bool) or not isinstance(created_round, int):
            raise ValueError("created_round must fit uint32 / created_round  uint32")
        if not 0 <= created_round <= 0xFFFFFFFF:
            raise ValueError("created_round must fit uint32 / created_round  uint32")
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
        'Send an AS request, reconnecting only after AS reports an offline SID.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS /  AS")

        # Use the caller-configured RPC window instead of an arbitrary retry
        # count. A finite deadline prevents a permanently unavailable AS from
        # blocking an evaluation and its GPU workers forever.
        # RPC
        
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
            # cannot duplicate an index or CAS transition. AS
            
            # AS
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
        'Run the later CAS phase for labels acknowledged by registration.'
        if isinstance(registrations, (str, bytes)) or not isinstance(
            registrations,
            Sequence,
        ):
            raise TypeError("registrations must be a sequence / registrations ")
        registrations_tuple = tuple(registrations)
        if not registrations_tuple:
            return []
        if any(
            not isinstance(registration, LabelRegistration)
            for registration in registrations_tuple
        ):
            raise TypeError(
                "registrations must contain LabelRegistration values / "
                "registrations  LabelRegistration "
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
        'Claim already registered labels without modifying index ownership.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS /  AS")
        labels = self._validated_submission_labels(protected_labels)
        if refresh_lease:
            # Registration can involve many native-index writes. Refresh
            # immediately before the later independent CAS phase so
            # server-queue delay is measured from a current lease.
            
            
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
        'Route locally retained records into paper hot and cold queues.'
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("decisions must be a sequence / decisions ")
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
                    "decisions  TaskClaimDecision"
                )
            record = records_by_label.get(decision.protected_label)
            if record is None:
                raise ClientLabelStoreError(
                    "AS decision has no local record correspondence / AS "
                )
            if decision.operation == "TRAIN":
                hot_records.append(record)
            elif decision.operation == "DEDUP":
                cold_records.append(record)
            else:
                raise ValueError("decision operation is invalid / ")
        return LocalTrainingQueues(tuple(hot_records), tuple(cold_records))

    def reconcile_recovery_claims(
        self,
        decisions: Sequence[TaskClaimDecision],
        *,
        forced_dedup_labels: Sequence[str] = (),
    ) -> list[TaskClaimDecision]:
        "Merge a fresh CAS result with this SID's heartbeat ownership proof.\n        CAS correctly returns DEDUP for an already PENDING task, including one\n        still owned by this SID. During recovery, a same-SID TRAIN heartbeat\n        instruction is therefore authoritative for retaining that hot item.\n        Labels explicitly supplied by AS as transferred always remain DEDUP."
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("decisions must be a sequence / decisions ")
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
                    "decisions  TaskClaimDecision"
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
        "Upload a trained checkpoint and commit this SID's TRAIN tasks.\n        The file is sent in bounded base64 chunks because the existing DwT-FL\n        protocol is JSON-only.  AS verifies a SHA-256 digest before it marks\n        the associated PENDING tasks COMMITTED.\n        compatibility and diagnostics, but AS derives FedAvg weights only from\n        actual COMMITTED state-table tasks. ``sample_count``"
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be non-negative / round_id ")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("sample_count must be positive / sample_count ")
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 1:
            raise FileNotFoundError("checkpoint must be a non-empty file / ")
        train_decisions = tuple(decision for decision in decisions if decision.operation == "TRAIN")
        task_ids = self._train_task_ids(train_decisions)
        with self._lock:
            session = self._as_session
        if session is None:
            raise RuntimeError("client is not registered with AS /  AS")
        update_id = uuid.uuid4().hex
        total_bytes = checkpoint_path.stat().st_size
        checkpoint_sha256 = self._hash_checkpoint_with_heartbeats(checkpoint_path)
        # Refresh again immediately before the stateful upload phase.
        
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
                # same ownership proof is required before any replay.
                # AS
                
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
                        # record. Do not replay the stale checkpoint.
                        # AS
                        
                        raise ModelUpdateOwnershipLostError(
                            "AS rejected the stale model update after task takeover / "
                            "AS ",
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
            # AS
            
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
        'Reconnect and re-claim every task before replaying a rejected upload.'
        expected_tasks = {decision.protected_label: decision.task_id for decision in train_decisions}
        if not expected_tasks or len(expected_tasks) != len(train_decisions):
            raise ValueError(
                "TRAIN decisions must have unique protected labels / "
                "TRAIN "
            )
        session = self.connect_to_as()
        recovered = self.claim_protected_labels_at_as(tuple(expected_tasks))
        recovered_by_label = {decision.protected_label: decision for decision in recovered}
        # The AS can atomically claim a released task for this same SID while it
        # answers the synchronous heartbeat that precedes CAS. In that case the
        # following explicit CAS correctly reports DEDUP/PENDING, while the
        # heartbeat instruction is the proof that this SID owns the task. AS
        # CAS
        
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
            # checkpoint.
            
            raise ModelUpdateOwnershipLostError(
                "model upload ownership was transferred during recovery / "
                "",
                recovered_decisions=tuple(recovered),
            )
        return session

    @staticmethod
    def _ownership_loss_instructions_from_error(
        error: RemoteServiceError,
    ) -> tuple[TaskClaimDecision, ...]:
        "Parse AS's 409 DEDUP instructions before any local retraining."
        raw_instructions = error.payload.get("dedup_instructions")
        if raw_instructions is None:
            return ()
        if not isinstance(raw_instructions, list):
            raise TransportError(
                "AS ownership-loss payload is invalid / AS "
            )
        parsed: list[TaskClaimDecision] = []
        for instruction in raw_instructions:
            if not isinstance(instruction, Mapping) or set(instruction) != {
                "protected_label", "task_id", "operation", "state"
            }:
                raise TransportError(
                    "AS ownership-loss instruction is invalid / "
                    "AS "
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
                    "AS "
                )
            try:
                validate_protected_label(protected_label)
            except OprfValidationError as validation_error:
                raise TransportError(
                    "AS ownership-loss label is invalid / AS "
                ) from validation_error
            parsed.append(TaskClaimDecision(protected_label, task_id, operation, state))
        if len({item.protected_label for item in parsed}) != len(parsed):
            raise TransportError(
                "AS ownership-loss labels must be unique / "
                "AS "
            )
        return tuple(parsed)

    def _hash_checkpoint_with_heartbeats(self, checkpoint_path: Path) -> str:
        'Hash one checkpoint while retaining the AS liveness lease.'
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
        'Send one idempotent upload chunk with bounded transport recovery.'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
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
        'Read one immutable global-model chunk with bounded transport recovery.\n        The AS does not mutate state for this endpoint: the request identifies\n        one round, offset, and maximum byte count, and the client subsequently\n        verifies the final SHA-256.  Retrying only a transport failure is thus\n        safe and avoids turning a temporary local accept-backlog spike into a\n        failed experiment case. AS'
        if self._as_transport is None:
            raise RuntimeError("AS endpoint is not configured /  AS ")
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
        'Return a locally known protected label without making a network call.'
        record_key = self._record_key(normalize_record(record))
        with self._lock:
            return self._labels_by_record.get(record_key)

    def generate_protected_labels(self, records: Sequence[str | bytes]) -> list[str]:
        'Generate labels through KS and atomically persist new correspondences.'
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError(
                "records must be a sequence of str or bytes / "
                "records  str  bytes "
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
                        "KS  OPRF "
                    )
            if changed:
                self._persist_store()
            return [
                self._labels_by_record[self._record_key(record)]
                for record in normalized_records
            ]

    def close(self) -> None:
        'Stop the background heartbeat worker without altering local mappings.'
        self._stop_heartbeat_worker()

    def __enter__(self) -> "ClientEntity":
        'Enter a managed client lifetime.'
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        'Stop heartbeats at managed-lifetime exit.'
        self.close()

    def _load_store(self) -> dict[str, str]:
        'Load a strict, client-bound mapping without creating an empty file.'
        path = self.config.label_store_path
        if not path.exists():
            return {}
        if path.is_symlink():
            raise ClientLabelStoreError(
                "label-store path must not be a symlink / "
            )
        try:
            serialized = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClientLabelStoreError("cannot read label store / ") from error
        if not isinstance(serialized, dict) or set(serialized) != {
            "schema_version",
            "oprf_suite",
            "client_id",
            "entries",
        }:
            raise ClientLabelStoreError("invalid label-store schema / ")
        if serialized["schema_version"] != LABEL_STORE_SCHEMA_VERSION:
            raise ClientLabelStoreError(
                "unsupported label-store version; create a fresh Ristretto255 cache instead of mixing legacy labels / "
                " Ristretto255 "
            )
        if serialized["oprf_suite"] != self.config.oprf_suite:
            raise ClientLabelStoreError(
                "label store uses another OPRF suite /  OPRF "
            )
        if serialized["client_id"] != self.config.client_id:
            raise ClientLabelStoreError(
                "label store belongs to another client / "
            )
        entries = serialized["entries"]
        if not isinstance(entries, list):
            raise ClientLabelStoreError("label-store entries must be an array / ")
        loaded: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"data_b64", "protected_label"}:
                raise ClientLabelStoreError("invalid label-store entry / ")
            encoded_record = entry["data_b64"]
            protected_label = entry["protected_label"]
            if not isinstance(encoded_record, str) or not isinstance(protected_label, str):
                raise ClientLabelStoreError("label-store value has invalid type / ")
            try:
                record = base64.b64decode(encoded_record.encode("ascii"), validate=True)
                normalized_record = normalize_record(record)
                validate_protected_label(protected_label)
            except (UnicodeEncodeError, ValueError, OprfValidationError) as error:
                raise ClientLabelStoreError(
                    "invalid protected-label mapping / "
                ) from error
            canonical_key = self._record_key(normalized_record)
            if encoded_record != canonical_key or canonical_key in loaded:
                raise ClientLabelStoreError(
                    "duplicate or non-canonical record mapping / "
                )
            loaded[canonical_key] = protected_label
        return loaded

    def _persist_store(self) -> None:
        'Atomically replace the private mapping after a complete OPRF response.'
        path = self.config.label_store_path
        if path.exists() and path.is_symlink():
            raise ClientLabelStoreError(
                "label-store path must not be a symlink / "
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
            raise ClientLabelStoreError("cannot persist label store / ") from error

    @staticmethod
    def _validated_submission_labels(protected_labels: Sequence[str]) -> tuple[str, ...]:
        'Validate the client-side set before it reaches the AS boundary.'
        if isinstance(protected_labels, (str, bytes)) or not isinstance(
            protected_labels,
            Sequence,
        ):
            raise TypeError(
                "protected_labels must be a sequence of strings / "
                "protected_labels "
            )
        labels = tuple(protected_labels)
        if not labels:
            raise ValueError("protected_labels must not be empty / protected_labels ")
        if any(not isinstance(label, str) for label in labels):
            raise TypeError(
                "protected_labels must contain only strings / "
                "protected_labels "
            )
        if len(set(labels)) != len(labels):
            raise ValueError("protected_labels must be unique / protected_labels ")
        try:
            for label in labels:
                validate_protected_label(label)
        except OprfValidationError as error:
            raise ValueError(
                "protected_labels contain an invalid OPRF label / "
                " OPRF "
            ) from error
        return labels

    def _start_heartbeat_worker(self, interval_seconds: float) -> None:
        'Start the client timer after registration has completed successfully.'
        self._heartbeat_stop = threading.Event()
        # Clients that connect in one protocol cohort must not subsequently
        # create a perfectly phase-aligned 0.1-second control burst. Derive a
        # deterministic per-client phase in one interval: this preserves the
        # configured heartbeat period and reproducibility, while spreading only
        # control-plane arrivals.
        # 0.1
        
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
        'Stop any prior heartbeat worker before a new connection is created.'
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
        'Send periodic heartbeats and retain background failures for inspection.'
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
                        "background heartbeat failed / "
                    )
            if stop_event.wait(interval_seconds):
                return

    @staticmethod
    def _validated_registration_response(response: WireMessage) -> ClientAsSession:
        'Validate an AS registration response before retaining its SID.'
        if response.message_type != AS_REGISTER_RESPONSE:
            raise TransportError("AS returned an invalid registration response / AS ")
        payload = response.payload
        if set(payload) != {"sid", "reused", "heartbeat_interval_seconds"}:
            raise TransportError("AS returned an invalid registration payload / AS ")
        return ClientEntity._session_from_payload(payload, expected_sid=None)

    @staticmethod
    def _validated_heartbeat_response(
        response: WireMessage,
        sid: int,
    ) -> tuple[ClientAsSession, int, tuple[RoundInstruction, ...]]:
        'Validate a matching successful heartbeat response.'
        if response.message_type != AS_HEARTBEAT_RESPONSE:
            raise TransportError("AS returned an invalid heartbeat response / AS ")
        payload = response.payload
        if set(payload) != {
            "sid",
            "online",
            "heartbeat_interval_seconds",
            "round",
            "instructions",
        }:
            raise TransportError("AS returned an invalid heartbeat payload / AS ")
        if payload["online"] is not True:
            raise TransportError("AS did not confirm client liveness / AS ")
        session = ClientEntity._session_from_payload(payload, expected_sid=sid)
        active_round = payload["round"]
        raw_instructions = payload["instructions"]
        if isinstance(active_round, bool) or not isinstance(active_round, int) or active_round < 0:
            raise TransportError("AS returned an invalid active round / AS ")
        if not isinstance(raw_instructions, list):
            raise TransportError("AS returned invalid round instructions / AS ")
        instructions: list[RoundInstruction] = []
        for raw_instruction in raw_instructions:
            if not isinstance(raw_instruction, Mapping) or set(raw_instruction) != {
                "protected_label",
                "task_id",
                "operation",
            }:
                raise TransportError("AS returned an invalid round instruction / AS ")
            protected_label = raw_instruction["protected_label"]
            task_id = raw_instruction["task_id"]
            operation = raw_instruction["operation"]
            if not isinstance(protected_label, str) or operation not in {"TRAIN", "DEDUP"}:
                raise TransportError("AS round instruction fields are invalid / AS ")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise TransportError("AS round task ID is invalid / AS  ID ")
            instructions.append(RoundInstruction(protected_label, task_id, operation))
        return session, active_round, tuple(instructions)

    @staticmethod
    def _session_from_payload(
        payload: Mapping[str, object],
        *,
        expected_sid: int | None,
    ) -> ClientAsSession:
        'Construct one typed session only from a canonical AS payload.'
        sid = payload["sid"]
        interval = payload["heartbeat_interval_seconds"]
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise TransportError("AS returned an invalid SID / AS  SID")
        if expected_sid is not None and sid != expected_sid:
            raise TransportError("AS heartbeat SID does not match / AS  SID ")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval <= 0:
            raise TransportError("AS returned an invalid heartbeat interval / AS ")
        return ClientAsSession(sid=sid, heartbeat_interval_seconds=float(interval))

    @staticmethod
    def _validated_label_registration_response(
        response: WireMessage,
        *,
        expected_sid: int,
        expected_round: int,
        expected_labels: tuple[str, ...],
    ) -> list[LabelRegistration]:
        'Validate every AS index-registration acknowledgement.'
        if response.message_type != AS_REGISTER_LABELS_RESPONSE:
            raise TransportError("AS returned an invalid label response / AS ")
        payload = response.payload
        if set(payload) != {"sid", "round", "registrations"}:
            raise TransportError("AS returned an invalid label payload / AS ")
        if payload["sid"] != expected_sid or payload["round"] != expected_round:
            raise TransportError("AS response does not match submission / AS ")
        raw_registrations = payload["registrations"]
        if not isinstance(raw_registrations, list) or len(raw_registrations) != len(
            expected_labels
        ):
            raise TransportError(
                "AS returned an invalid registration count / AS "
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
                raise TransportError("AS returned an invalid registration / AS ")
            label = raw_registration["protected_label"]
            task_id = raw_registration["task_id"]
            if label != expected_label:
                raise TransportError("AS registration label is invalid / AS ")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise TransportError("AS registration task_id is invalid / AS  task_id ")
            registrations.append(LabelRegistration(label, task_id))
        return registrations

    @staticmethod
    def _validated_task_claim_response(
        response: WireMessage,
        *,
        expected_sid: int,
        expected_labels: tuple[str, ...],
    ) -> list[TaskClaimDecision]:
        "Validate the paper's TRAIN or DEDUP state mapping from the AS."
        if response.message_type != AS_CLAIM_TASKS_RESPONSE:
            raise TransportError("AS returned an invalid task-claim response / AS ")
        payload = response.payload
        if set(payload) != {"sid", "decisions"} or payload["sid"] != expected_sid:
            raise TransportError("AS returned an invalid task-claim payload / AS ")
        raw_decisions = payload["decisions"]
        if not isinstance(raw_decisions, list) or len(raw_decisions) != len(expected_labels):
            raise TransportError("AS returned an invalid task-decision count / AS ")
        decisions: list[TaskClaimDecision] = []
        for expected_label, raw_decision in zip(expected_labels, raw_decisions, strict=True):
            if not isinstance(raw_decision, Mapping) or set(raw_decision) != {
                "protected_label",
                "task_id",
                "operation",
                "state",
            }:
                raise TransportError("AS returned an invalid task decision / AS ")
            label = raw_decision["protected_label"]
            task_id = raw_decision["task_id"]
            operation = raw_decision["operation"]
            state = raw_decision["state"]
            if label != expected_label or operation not in {"TRAIN", "DEDUP"}:
                raise TransportError("AS task decision is invalid / AS ")
            if state not in {"EMPTY", "PENDING", "COMMITTED"}:
                raise TransportError("AS task state is invalid / AS ")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise TransportError("AS task_id is invalid / AS task_id ")
            if operation == "TRAIN" and state != "PENDING":
                raise TransportError("TRAIN decision must be PENDING / TRAIN  PENDING")
            decisions.append(TaskClaimDecision(label, task_id, operation, state))
        return decisions

    @staticmethod
    def _train_task_ids(decisions: Sequence[TaskClaimDecision]) -> tuple[int, ...]:
        'Extract the unique PENDING task identifiers owned by this client.'
        if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
            raise TypeError("decisions must be a sequence / decisions ")
        task_ids: list[int] = []
        for decision in decisions:
            if not isinstance(decision, TaskClaimDecision):
                raise TypeError(
                    "decisions must contain TaskClaimDecision / "
                    "decisions  TaskClaimDecision"
                )
            if decision.operation == "TRAIN":
                if decision.state != "PENDING":
                    raise ValueError("TRAIN task must be PENDING / TRAIN  PENDING")
                task_ids.append(decision.task_id)
            elif decision.operation != "DEDUP":
                raise ValueError("decision operation is invalid / ")
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("at least one unique TRAIN task is required /  TRAIN ")
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
        'Validate one ordered model-upload chunk acknowledgement.'
        if response.message_type != AS_MODEL_CHUNK_RESPONSE:
            raise TransportError("AS returned an invalid model-chunk response / AS ")
        payload = response.payload
        if set(payload) != {"sid", "round", "update_id", "received_bytes"}:
            raise TransportError("AS returned an invalid model-chunk payload / AS ")
        received_bytes = payload["received_bytes"]
        if (
            payload["sid"] != sid
            or payload["round"] != round_id
            or payload["update_id"] != update_id
            or isinstance(received_bytes, bool)
            or not isinstance(received_bytes, int)
            or not expected_received_bytes <= received_bytes <= total_bytes
        ):
            raise TransportError("AS model-chunk acknowledgement does not match / AS ")

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
        'Validate final task commits bound to one checkpoint digest.'
        if response.message_type != AS_MODEL_FINALIZE_RESPONSE:
            raise TransportError("AS returned an invalid model-finalize response / AS ")
        payload = response.payload
        expected_fields = {
            "sid",
            "round",
            "update_id",
            "committed_task_ids",
            "checkpoint_sha256",
        }
        if set(payload) != expected_fields:
            raise TransportError("AS returned an invalid model-finalize payload / AS ")
        if (
            payload["sid"] != sid
            or payload["round"] != round_id
            or payload["update_id"] != update_id
            or payload["checkpoint_sha256"] != checkpoint_sha256
            or payload["committed_task_ids"] != list(task_ids)
        ):
            raise TransportError("AS model-finalize acknowledgement does not match / AS ")
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
        'Validate and decode one bounded AS global-model chunk response.'
        if response.message_type != AS_GLOBAL_MODEL_CHUNK_RESPONSE:
            raise TransportError("AS returned an invalid global-model response / AS ")
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
            raise TransportError("AS global-model response does not match / AS ")
        total_bytes = payload["global_bytes"]
        sha256 = payload["global_sha256"]
        encoded_chunk = payload["chunk_base64"]
        complete = payload["complete"]
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
            raise TransportError("AS global model size is invalid / AS ")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise TransportError("AS global model digest is invalid / AS ")
        if not isinstance(encoded_chunk, str) or not isinstance(complete, bool):
            raise TransportError("AS global model chunk is invalid / AS ")
        try:
            chunk = base64.b64decode(encoded_chunk.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise TransportError(
                "AS global model chunk is not base64 / AS  Base64"
            ) from error
        if len(chunk) > self.config.model_chunk_bytes:
            raise TransportError("AS global model chunk exceeds bound / AS ")
        if offset + len(chunk) > total_bytes or (complete and offset + len(chunk) != total_bytes):
            raise TransportError("AS global model chunk bounds are invalid / AS ")
        return total_bytes, sha256, chunk, complete

    @staticmethod
    def _record_key(record: bytes) -> str:
        'Encode raw local data losslessly as one canonical JSON-safe key.'
        return base64.b64encode(record).decode("ascii")


def _validated_sid_sequence(participant_sids: Sequence[int]) -> tuple[int, ...]:
    'Validate a non-empty, ordered, duplicate-free FedAvg roster.'
    if isinstance(participant_sids, (str, bytes)) or not isinstance(participant_sids, Sequence):
        raise TypeError("participant_sids must be a sequence / participant_sids ")
    normalized = tuple(participant_sids)
    if (
        not normalized
        or any(isinstance(sid, bool) or not isinstance(sid, int) or sid < 1 for sid in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("participant_sids must be unique positive integers / "
                         "participant_sids ")
    return normalized
