'Typed ctypes binding for the DwT-FL native index. / DwT-FL'

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Final


LABEL_HEX_LENGTH: Final[int] = 512
"""Canonical OPRF-label length in lowercase hexadecimal characters. /  OPRF """


class NativeIndexError(RuntimeError):
    'Raised when the native index rejects a valid binding operation.'


class TaskState(IntEnum):
    'Native task-state values exposed without duplicating bit packing.'

    EMPTY = 0
    PENDING = 1
    COMMITTED = 2


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    'Decoded task state returned by the native library.'

    state: TaskState
    trainer: int
    version: int


def _library_filename() -> str:
    'Return the platform-specific shared-library filename.'
    return "atomic_word.dll" if platform.system() == "Windows" else "atomic_word.so"


def _default_library_path() -> Path:
    'Locate a packaged or locally built native library.'
    configured_path = os.environ.get("DBTFL_NATIVE_LIBRARY")
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    filename = _library_filename()
    candidates = (
        package_root / "native" / filename,
        project_root / "native" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "DwT-FL native library was not found. "
        f"Run scripts/build_native.py first. Searched: {searched}"
    )


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    'Declare the complete stable C ABI used by Python.'
    pointer = ctypes.c_void_p
    uint32 = ctypes.c_uint32
    uint64 = ctypes.c_uint64
    int32 = ctypes.c_int
    int64 = ctypes.c_int64
    char_pointer = ctypes.c_char_p

    library.dbt_index_create.argtypes = [uint32, uint32, uint32]
    library.dbt_index_create.restype = pointer
    library.dbt_index_destroy.argtypes = [pointer]
    library.dbt_index_destroy.restype = None
    library.dbt_index_is_lock_free.argtypes = [pointer]
    library.dbt_index_is_lock_free.restype = int32
    library.dbt_index_edge_count.argtypes = [pointer]
    library.dbt_index_edge_count.restype = int32
    library.dbt_index_register_label.argtypes = [
        pointer,
        char_pointer,
        uint32,
        uint32,
        ctypes.POINTER(int32),
    ]
    library.dbt_index_register_label.restype = int32
    library.dbt_index_find_label.argtypes = [pointer, char_pointer, ctypes.POINTER(int32)]
    library.dbt_index_find_label.restype = int32
    library.dbt_index_task_has_owner.argtypes = [pointer, int32, uint32]
    library.dbt_index_task_has_owner.restype = int32
    library.dbt_index_copy_client_tasks.argtypes = [pointer, uint32, ctypes.POINTER(int32), uint32]
    library.dbt_index_copy_client_tasks.restype = int32
    library.dbt_index_copy_task_owners.argtypes = [pointer, int32, ctypes.POINTER(uint32), uint32]
    library.dbt_index_copy_task_owners.restype = int32
    library.dbt_index_task_label.argtypes = [pointer, int32, ctypes.c_char_p, uint32]
    library.dbt_index_task_label.restype = int32
    library.dbt_task_load.argtypes = [pointer, int32]
    library.dbt_task_load.restype = int64
    library.dbt_task_state.argtypes = [pointer, int32]
    library.dbt_task_state.restype = int32
    library.dbt_task_trainer.argtypes = [pointer, int32]
    library.dbt_task_trainer.restype = uint32
    library.dbt_task_version.argtypes = [pointer, int32]
    library.dbt_task_version.restype = uint32
    library.dbt_task_try_claim.argtypes = [pointer, int32, uint32]
    library.dbt_task_try_claim.restype = int32
    library.dbt_task_mark_committed.argtypes = [pointer, int32, uint32]
    library.dbt_task_mark_committed.restype = int32
    library.dbt_task_release_if_trainer.argtypes = [pointer, int32, uint32]
    library.dbt_task_release_if_trainer.restype = int32
    library.dbt_task_compare_exchange.argtypes = [pointer, int32, int64, int64]
    library.dbt_task_compare_exchange.restype = int32
    library.dbt_task_reset.argtypes = [pointer, int32]
    library.dbt_task_reset.restype = int32
    library.dbt_task_previous_get.argtypes = [pointer, int32]
    library.dbt_task_previous_get.restype = uint32
    library.dbt_task_previous_set.argtypes = [pointer, int32, uint32]
    library.dbt_task_previous_set.restype = int32
    library.dbt_task_recovery_get.argtypes = [pointer, int32]
    library.dbt_task_recovery_get.restype = int32
    library.dbt_task_recovery_set.argtypes = [pointer, int32, int32]
    library.dbt_task_recovery_set.restype = int32
    library.dbt_task_created_round.argtypes = [pointer, int32]
    library.dbt_task_created_round.restype = uint32
    library.dbt_index_copy_all_tasks.argtypes = [pointer, ctypes.POINTER(int32), uint32]
    library.dbt_index_copy_all_tasks.restype = int32
    library.dbt_index_memory_bytes.argtypes = [pointer]
    library.dbt_index_memory_bytes.restype = uint64
    return library


@lru_cache(maxsize=None)
def _load_library(path: str) -> ctypes.CDLL:
    'Load and configure one absolute library path once.'
    return _configure_library(ctypes.CDLL(path))


def _canonical_label(label: str) -> bytes:
    'Validate and encode one canonical protected label.'
    if not isinstance(label, str):
        raise TypeError("label must be a string / ")
    is_canonical = len(label) == LABEL_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in label
    )
    if not is_canonical:
        raise ValueError(
            "label must contain exactly 512 lowercase hexadecimal characters / "
            " 512 "
        )
    return label.encode("ascii")


class NativeIndex:
    'Own one native concurrent index through an explicit lifetime.'

    def __init__(
        self,
        capacity: int,
        max_clients: int,
        max_edges: int,
        *,
        library_path: str | Path | None = None,
    ) -> None:
        'Allocate an index with fixed capacities.'
        if capacity <= 0 or max_clients <= 0 or max_edges <= 0:
            raise ValueError(
                "all native-index capacities must be positive / "
            )
        self.capacity = capacity
        self.max_clients = max_clients
        self.max_edges = max_edges
        resolved_library = (
            Path(library_path).resolve()
            if library_path is not None
            else _default_library_path()
        )
        self._library = _load_library(str(resolved_library))
        self._pointer = self._library.dbt_index_create(capacity, max_clients, max_edges)
        if not self._pointer:
            raise NativeIndexError("native index allocation failed / ")

    def __enter__(self) -> "NativeIndex":
        'Enter a context-managed native-index lifetime.'
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        'Release native memory at context exit.'
        self.close()

    def close(self) -> None:
        'Release the owned native index exactly once.'
        if self._pointer:
            self._library.dbt_index_destroy(self._pointer)
            self._pointer = None

    def _require_open(self) -> ctypes.c_void_p:
        'Return the live native pointer or raise a clear error.'
        if not self._pointer:
            raise NativeIndexError("native index is closed / ")
        return self._pointer

    @property
    def is_lock_free(self) -> bool:
        'Report hardware support for the native atomic primitives.'
        return bool(self._library.dbt_index_is_lock_free(self._require_open()))

    @property
    def edge_count(self) -> int:
        'Return successfully reserved owner-edge count.'
        count = self._library.dbt_index_edge_count(self._require_open())
        if count < 0:
            raise NativeIndexError("native edge count failed / ")
        return count

    @property
    def memory_bytes(self) -> int:
        'Return native reserved metadata bytes.'
        return int(self._library.dbt_index_memory_bytes(self._require_open()))

    def register_label(self, label: str, token: int, created_round: int) -> int:
        'Register one owner-label relation and return its task identifier.'
        self._validate_token(token)
        self._validate_round(created_round)
        task_id = ctypes.c_int()
        accepted = self._library.dbt_index_register_label(
            self._require_open(),
            _canonical_label(label),
            token,
            created_round,
            ctypes.byref(task_id),
        )
        if not accepted:
            raise NativeIndexError("native label registration failed / ")
        return task_id.value

    def find_label(self, label: str) -> int | None:
        'Return a task identifier for a registered label, if present.'
        task_id = ctypes.c_int()
        found = self._library.dbt_index_find_label(
            self._require_open(), _canonical_label(label), ctypes.byref(task_id)
        )
        return task_id.value if found else None

    def owners(self, task_id: int) -> tuple[int, ...]:
        'Return the deterministic owner set for one task.'
        self._validate_task_id(task_id)
        output = (ctypes.c_uint32 * self.max_clients)()
        count = self._library.dbt_index_copy_task_owners(
            self._require_open(), task_id, output, self.max_clients
        )
        if count < 0:
            raise NativeIndexError("native owner export failed / ")
        return tuple(sorted(output[index] for index in range(count)))

    def task_has_owner(self, task_id: int, token: int) -> bool:
        'Return whether one SID owns a task before task-state operations.'
        self._validate_task_id(task_id)
        self._validate_token(token)
        return bool(
            self._library.dbt_index_task_has_owner(
                self._require_open(),
                task_id,
                token,
            )
        )

    def client_tasks(self, token: int) -> tuple[int, ...]:
        'Return the deterministic task set owned by one client.'
        self._validate_token(token)
        output = (ctypes.c_int * self.capacity)()
        count = self._library.dbt_index_copy_client_tasks(
            self._require_open(), token, output, self.capacity
        )
        if count < 0:
            raise NativeIndexError("native client-task export failed / ")
        return tuple(sorted(output[index] for index in range(count)))

    def task_label(self, task_id: int) -> str:
        'Return the canonical label stored for one task.'
        self._validate_task_id(task_id)
        output = ctypes.create_string_buffer(LABEL_HEX_LENGTH + 1)
        copied = self._library.dbt_index_task_label(
            self._require_open(), task_id, output, len(output)
        )
        if not copied:
            raise NativeIndexError("native task-label export failed / ")
        return output.value.decode("ascii")

    def snapshot(self, task_id: int) -> TaskSnapshot:
        'Return decoded state, trainer, and version for one task.'
        self._validate_task_id(task_id)
        state_value = self._library.dbt_task_state(self._require_open(), task_id)
        if state_value < 0:
            raise NativeIndexError("native task-state query failed / ")
        try:
            state = TaskState(state_value)
        except ValueError as error:
            raise NativeIndexError(f"unknown native task state: {state_value}") from error
        return TaskSnapshot(
            state=state,
            trainer=int(self._library.dbt_task_trainer(self._require_open(), task_id)),
            version=int(self._library.dbt_task_version(self._require_open(), task_id)),
        )

    def try_claim(self, task_id: int, trainer: int) -> bool:
        'Attempt an EMPTY-to-PENDING state transition.'
        self._validate_task_id(task_id)
        self._validate_token(trainer)
        return bool(self._library.dbt_task_try_claim(self._require_open(), task_id, trainer))

    def mark_committed(self, task_id: int, trainer: int) -> bool:
        'Commit a task only for its current trainer.'
        self._validate_task_id(task_id)
        self._validate_token(trainer)
        return bool(self._library.dbt_task_mark_committed(self._require_open(), task_id, trainer))

    def release_if_trainer(self, task_id: int, trainer: int) -> bool:
        'Release a pending task only for its current trainer.'
        self._validate_task_id(task_id)
        self._validate_token(trainer)
        return bool(
            self._library.dbt_task_release_if_trainer(
                self._require_open(), task_id, trainer
            )
        )

    def reset(self, task_id: int) -> None:
        'Reset a task while incrementing its native version.'
        self._validate_task_id(task_id)
        if not self._library.dbt_task_reset(self._require_open(), task_id):
            raise NativeIndexError("native task reset failed / ")

    def previous_trainer(self, task_id: int) -> int:
        'Return the trainer retained from the last completed round.'
        self._validate_task_id(task_id)
        return int(self._library.dbt_task_previous_get(self._require_open(), task_id))

    def set_previous_trainer(self, task_id: int, trainer: int) -> None:
        'Persist one prior-round trainer independently from current state.'
        self._validate_task_id(task_id)
        self._validate_token(trainer)
        if not self._library.dbt_task_previous_set(self._require_open(), task_id, trainer):
            raise NativeIndexError("native previous trainer update failed / ")

    def recovery_required(self, task_id: int) -> bool:
        'Return whether a dropped trainer left this task needing recovery.'
        self._validate_task_id(task_id)
        required = self._library.dbt_task_recovery_get(self._require_open(), task_id)
        if required < 0:
            raise NativeIndexError("native recovery query failed / ")
        return bool(required)

    def set_recovery_required(self, task_id: int, required: bool) -> None:
        'Mark whether this task needs a safe trainer reassignment.'
        self._validate_task_id(task_id)
        if not isinstance(required, bool):
            raise TypeError("required must be bool / required ")
        if not self._library.dbt_task_recovery_set(self._require_open(), task_id, int(required)):
            raise NativeIndexError("native recovery update failed / ")

    def all_task_ids(self) -> tuple[int, ...]:
        'Return all allocated tasks for round reset and recovery scans.'
        output = (ctypes.c_int * self.capacity)()
        count = self._library.dbt_index_copy_all_tasks(
            self._require_open(),
            output,
            self.capacity,
        )
        if count < 0:
            raise NativeIndexError("native task export failed / ")
        return tuple(sorted(output[index] for index in range(count)))

    def _validate_token(self, token: int) -> None:
        'Reject tokens outside the native index domain.'
        if not isinstance(token, int) or not 1 <= token <= self.max_clients:
            raise ValueError("token must be an allocated client identifier / ")

    def _validate_round(self, created_round: int) -> None:
        'Reject invalid unsigned round identifiers.'
        if not isinstance(created_round, int) or not 0 <= created_round <= 0xFFFFFFFF:
            raise ValueError("created_round must fit uint32 / created_round  uint32")

    def _validate_task_id(self, task_id: int) -> None:
        'Reject task identifiers outside the allocated table.'
        if not isinstance(task_id, int) or not 0 <= task_id < self.capacity:
            raise ValueError("task_id is outside the native table / task_id ")
