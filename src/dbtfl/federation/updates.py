'Streaming-safe metadata and chunks for HTTP model-update transfer.'

from __future__ import annotations

import hashlib
import re
import shutil
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final


_UPDATE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9]{32}$")
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class ModelUpdateDescriptor:
    'Immutable AS-accepted checkpoint metadata for one client and round.\n    AS'

    round_id: int
    sid: int
    update_id: str
    sample_count: int
    task_ids: tuple[int, ...]
    sha256: str
    byte_count: int
    checkpoint_path: Path


@dataclass(slots=True)
class _StagedUpload:
    'Private sequential-write state for one incomplete checkpoint upload.'

    round_id: int
    sid: int
    update_id: str
    total_bytes: int
    sha256: str
    temporary_path: Path
    received_bytes: int = 0


class ModelUpdateStore:
    'Filesystem-backed bounded chunk store for full-model HTTP uploads.'

    def __init__(self, root_directory: Path, *, max_update_bytes: int) -> None:
        'Create an AS-local update store without exposing it to clients.'
        if max_update_bytes < 1:
            raise ValueError("max_update_bytes must be positive / ")
        self.root_directory = Path(root_directory).resolve()
        self.max_update_bytes = max_update_bytes
        self._staged: dict[tuple[int, int, str], _StagedUpload] = {}
        self._lock = threading.RLock()

    def append_chunk(
        self,
        *,
        round_id: int,
        sid: int,
        update_id: str,
        total_bytes: int,
        sha256: str,
        offset: int,
        chunk: bytes,
    ) -> int:
        'Append one contiguous verified-size chunk and return bytes received.'
        _validate_upload_fields(round_id, sid, update_id, total_bytes, sha256)
        if offset < 0 or not chunk or offset + len(chunk) > total_bytes:
            raise ValueError("invalid upload chunk bounds / ")
        key = (round_id, sid, update_id)
        with self._lock:
            staged = self._staged.get(key)
            if staged is None:
                if offset != 0:
                    raise ValueError(
                        "first upload chunk must start at offset zero / "
                        ""
                    )
                temporary_path = self._temporary_path(round_id, sid, update_id)
                temporary_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path.unlink(missing_ok=True)
                staged = _StagedUpload(
                    round_id=round_id,
                    sid=sid,
                    update_id=update_id,
                    total_bytes=total_bytes,
                    sha256=sha256,
                    temporary_path=temporary_path,
                )
                self._staged[key] = staged
            if (
                staged.total_bytes != total_bytes
                or staged.sha256 != sha256
            ):
                raise ValueError("upload metadata is inconsistent / ")
            if offset < staged.received_bytes:
                # A response can be lost after AS durably writes a chunk. Accept
                # an exact replay of already stored bytes so the client can safely
                # retry that one request without duplicating file contents.
                
                
                if offset + len(chunk) > staged.received_bytes:
                    raise ValueError(
                        "upload chunk overlaps unwritten bytes / "
                    )
                with staged.temporary_path.open("rb") as stream:
                    stream.seek(offset)
                    stored_chunk = stream.read(len(chunk))
                if stored_chunk != chunk:
                    raise ValueError(
                        "replayed upload chunk differs from stored bytes / "
                        ""
                    )
                return staged.received_bytes
            if offset != staged.received_bytes:
                raise ValueError("upload chunk is out of order / ")
            with staged.temporary_path.open("ab") as stream:
                stream.write(chunk)
            staged.received_bytes += len(chunk)
            return staged.received_bytes

    def finalize(
        self,
        *,
        round_id: int,
        sid: int,
        update_id: str,
        total_bytes: int,
        sha256: str,
    ) -> Path:
        'Verify digest and atomically publish one completed checkpoint.'
        _validate_upload_fields(round_id, sid, update_id, total_bytes, sha256)
        key = (round_id, sid, update_id)
        with self._lock:
            staged = self._staged.get(key)
            if staged is None or staged.received_bytes != total_bytes:
                raise ValueError("upload is incomplete / ")
            if sha256_file(staged.temporary_path) != sha256:
                staged.temporary_path.unlink(missing_ok=True)
                del self._staged[key]
                raise ValueError("checkpoint digest does not match / ")
            destination = self._published_path(round_id, sid, update_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.temporary_path.replace(destination)
            del self._staged[key]
            return destination

    def global_checkpoint_path(self, round_id: int) -> Path:
        'Return the AS-local path reserved for one aggregated global model.'
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be a non-negative integer / ")
        return self.root_directory / f"round-{round_id}" / "global_model.safetensors"

    def discard(self, *, round_id: int, sid: int, update_id: str) -> None:
        'Discard one known invalidated staged upload without publishing it.\n        This is used after AS proves the training ownership changed before\n        finalization. The caller has already validated the update identity at\n        the protocol boundary, so no model content is inspected here.'
        if (
            isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0
            or isinstance(sid, bool) or not isinstance(sid, int) or sid < 1
            or not isinstance(update_id, str) or not _UPDATE_ID_PATTERN.fullmatch(update_id)
        ):
            raise ValueError("invalid staged-upload identity / ")
        key = (round_id, sid, update_id)
        with self._lock:
            staged = self._staged.pop(key, None)
            if staged is not None:
                staged.temporary_path.unlink(missing_ok=True)

    def clear(self) -> None:
        "Remove only this store's experiment artifacts after an authorized reset."
        with self._lock:
            if self.root_directory.exists():
                shutil.rmtree(self.root_directory)
            self._staged.clear()

    def _temporary_path(self, round_id: int, sid: int, update_id: str) -> Path:
        'Construct a non-published partial-upload path.'
        return self.root_directory / "staging" / f"r{round_id}-s{sid}-{update_id}.part"

    def _published_path(self, round_id: int, sid: int, update_id: str) -> Path:
        'Construct a deterministic published-update path.'
        return self.root_directory / f"round-{round_id}" / f"sid-{sid}-{update_id}.safetensors"


def sha256_file(path: Path) -> str:
    'Return a streaming SHA-256 digest of one local checkpoint.'
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunk_file(path: Path, chunk_bytes: int) -> Iterator[tuple[int, bytes]]:
    'Yield sequential bounded chunks with offsets for a checkpoint file.'
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive / ")
    offset = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            yield offset, chunk
            offset += len(chunk)


def _validate_upload_fields(
    round_id: int,
    sid: int,
    update_id: str,
    total_bytes: int,
    sha256: str,
) -> None:
    'Validate identifiers before they can influence a filesystem path.'
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError("round_id must be a non-negative integer / ")
    if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
        raise ValueError("sid must be a positive integer / SID ")
    if not isinstance(update_id, str) or not _UPDATE_ID_PATTERN.fullmatch(update_id):
        raise ValueError(
            "update_id must be a 32-character lowercase hex UUID / "
            "update_id  32  UUID"
        )
    if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
        raise ValueError("total_bytes must be a positive integer / ")
    if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError(
            "sha256 must be a lowercase SHA-256 hex digest / "
            "sha256  SHA-256 "
        )
