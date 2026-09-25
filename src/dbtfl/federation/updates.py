"""Streaming-safe metadata and chunks for HTTP model-update transfer.

用于 HTTP 模型更新传输的流式安全元数据与分块工具。
"""

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
    """Immutable AS-accepted checkpoint metadata for one client and round.

    AS 为一个客户端和轮次接受的不可变检查点元数据。
    """

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
    """Private sequential-write state for one incomplete checkpoint upload.

    一个未完成检查点上传的私有顺序写入状态。
    """

    round_id: int
    sid: int
    update_id: str
    total_bytes: int
    sha256: str
    temporary_path: Path
    received_bytes: int = 0


class ModelUpdateStore:
    """Filesystem-backed bounded chunk store for full-model HTTP uploads.

    面向完整模型 HTTP 上传、基于文件系统的有界分块存储。
    """

    def __init__(self, root_directory: Path, *, max_update_bytes: int) -> None:
        """Create an AS-local update store without exposing it to clients.

        创建一个仅 AS 本地可见的更新存储。
        """
        if max_update_bytes < 1:
            raise ValueError("max_update_bytes must be positive / 最大更新字节数必须为正数")
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
        """Append one contiguous verified-size chunk and return bytes received.

        追加一个连续且大小经验证的分块，并返回已接收字节数。
        """
        _validate_upload_fields(round_id, sid, update_id, total_bytes, sha256)
        if offset < 0 or not chunk or offset + len(chunk) > total_bytes:
            raise ValueError("invalid upload chunk bounds / 上传分块边界无效")
        key = (round_id, sid, update_id)
        with self._lock:
            staged = self._staged.get(key)
            if staged is None:
                if offset != 0:
                    raise ValueError(
                        "first upload chunk must start at offset zero / "
                        "首个上传分块必须从偏移零开始"
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
                raise ValueError("upload metadata is inconsistent / 上传元数据不一致")
            if offset < staged.received_bytes:
                # A response can be lost after AS durably writes a chunk. Accept
                # an exact replay of already stored bytes so the client can safely
                # retry that one request without duplicating file contents. 响应可能
                # 在 AS 已持久化分块后丢失；接受与已存字节完全一致的重放，使客户端能安全
                # 重试该请求而不会重复写入文件内容。
                if offset + len(chunk) > staged.received_bytes:
                    raise ValueError(
                        "upload chunk overlaps unwritten bytes / 上传分块覆盖未写入字节"
                    )
                with staged.temporary_path.open("rb") as stream:
                    stream.seek(offset)
                    stored_chunk = stream.read(len(chunk))
                if stored_chunk != chunk:
                    raise ValueError(
                        "replayed upload chunk differs from stored bytes / "
                        "重放的上传分块与已存字节不同"
                    )
                return staged.received_bytes
            if offset != staged.received_bytes:
                raise ValueError("upload chunk is out of order / 上传分块顺序错误")
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
        """Verify digest and atomically publish one completed checkpoint.

        验证摘要并原子发布一个完成的检查点。
        """
        _validate_upload_fields(round_id, sid, update_id, total_bytes, sha256)
        key = (round_id, sid, update_id)
        with self._lock:
            staged = self._staged.get(key)
            if staged is None or staged.received_bytes != total_bytes:
                raise ValueError("upload is incomplete / 上传尚未完成")
            if sha256_file(staged.temporary_path) != sha256:
                staged.temporary_path.unlink(missing_ok=True)
                del self._staged[key]
                raise ValueError("checkpoint digest does not match / 检查点摘要不匹配")
            destination = self._published_path(round_id, sid, update_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.temporary_path.replace(destination)
            del self._staged[key]
            return destination

    def global_checkpoint_path(self, round_id: int) -> Path:
        """Return the AS-local path reserved for one aggregated global model.

        返回为一个已聚合全局模型保留的 AS 本地路径。
        """
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be a non-negative integer / 轮次标识必须为非负整数")
        return self.root_directory / f"round-{round_id}" / "global_model.safetensors"

    def discard(self, *, round_id: int, sid: int, update_id: str) -> None:
        """Discard one known invalidated staged upload without publishing it.

        丢弃一个已知失效的暂存上传，且绝不发布它。

        This is used after AS proves the training ownership changed before
        finalization. The caller has already validated the update identity at
        the protocol boundary, so no model content is inspected here. 当 AS 在
        最终提交前证明训练所有权已改变时使用。调用方已在线协议边界验证更新身份，
        因此此处不会检查任何模型内容。
        """
        if (
            isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0
            or isinstance(sid, bool) or not isinstance(sid, int) or sid < 1
            or not isinstance(update_id, str) or not _UPDATE_ID_PATTERN.fullmatch(update_id)
        ):
            raise ValueError("invalid staged-upload identity / 无效暂存上传身份")
        key = (round_id, sid, update_id)
        with self._lock:
            staged = self._staged.pop(key, None)
            if staged is not None:
                staged.temporary_path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove only this store's experiment artifacts after an authorized reset.

        在已授权重置后，仅删除本存储拥有的实验产物。
        """
        with self._lock:
            if self.root_directory.exists():
                shutil.rmtree(self.root_directory)
            self._staged.clear()

    def _temporary_path(self, round_id: int, sid: int, update_id: str) -> Path:
        """Construct a non-published partial-upload path. / 构造未发布的部分上传路径。"""
        return self.root_directory / "staging" / f"r{round_id}-s{sid}-{update_id}.part"

    def _published_path(self, round_id: int, sid: int, update_id: str) -> Path:
        """Construct a deterministic published-update path. / 构造确定性的已发布更新路径。"""
        return self.root_directory / f"round-{round_id}" / f"sid-{sid}-{update_id}.safetensors"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest of one local checkpoint.

    返回一个本地检查点的流式 SHA-256 摘要。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunk_file(path: Path, chunk_bytes: int) -> Iterator[tuple[int, bytes]]:
    """Yield sequential bounded chunks with offsets for a checkpoint file.

    为检查点文件产生带偏移量的顺序有界分块。
    """
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive / 分块字节数必须为正数")
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
    """Validate identifiers before they can influence a filesystem path.

    在标识影响文件系统路径前验证它们。
    """
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError("round_id must be a non-negative integer / 轮次标识必须为非负整数")
    if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
        raise ValueError("sid must be a positive integer / SID 必须为正整数")
    if not isinstance(update_id, str) or not _UPDATE_ID_PATTERN.fullmatch(update_id):
        raise ValueError(
            "update_id must be a 32-character lowercase hex UUID / "
            "update_id 必须为 32 位小写十六进制 UUID"
        )
    if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 1:
        raise ValueError("total_bytes must be a positive integer / 总字节数必须为正整数")
    if not isinstance(sha256, str) or not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError(
            "sha256 must be a lowercase SHA-256 hex digest / "
            "sha256 必须为小写 SHA-256 十六进制摘要"
        )
