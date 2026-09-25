"""Persistent fixed-allocation OPRF precomputation for DwT-FL experiments.

用于 DwT-FL 实验的持久化固定分配 OPRF 预计算。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dbtfl.entities import ClientConfig, ClientEntity, KeyServerConfig, KeyServerEntity
from dbtfl.oprf import OPRF_SUITE_IDENTIFIER
from dbtfl.training import PreparedRecord, iter_prepared_records


_SCHEMA_VERSION = "2.0"
_CLIENTS = 10
_RECORDS_PER_CLIENT = 1024
_DUPLICATE_RATIO = 0.30
_SEED = 17


@dataclass(frozen=True, slots=True)
class PrecomputedOprfDataset:
    """Fixed data allocation and persistent label stores for one KS key.

    一个 KS 密钥对应的固定数据分配和持久化标签存储。
    """

    directory: Path
    records_by_client: dict[str, tuple[str, ...]]
    manifest: dict[str, object]

    @property
    def key_path(self) -> Path:
        """Return the persistent local KS key. / 返回持久化本地 KS 密钥。"""
        return self.directory / "ks-oprf-ristretto255-key.json"

    def label_store_path(self, client_id: str) -> Path:
        """Return one client's durable protected-label mapping path.

        返回一个客户端的持久化保护标签映射路径。
        """
        return self.directory / "client-label-stores" / f"{client_id}.json"


def build_or_load_precomputed_dataset(
    prepared_data_path: Path,
    directory: Path,
    *,
    oprf_batch_size: int = 1024,
) -> PrecomputedOprfDataset:
    """Compute the fixed 10x1024/r=0.3 labels once, then only load them.

    一次性计算固定 10x1024/r=0.3 标签，此后仅加载。
    """
    root = Path(directory).resolve()
    records = _load_unique_records(prepared_data_path)
    source_digest = _records_digest(records)
    if (root / "manifest.json").is_file():
        return _load(root, source_digest)
    root.mkdir(parents=True, exist_ok=True)
    allocation = _allocate(records)
    _write_json(root / "fixed-allocation.json", {
        "schema_version": _SCHEMA_VERSION,
        "clients": {
            client_id: [base64.b64encode(text.encode("utf-8")).decode("ascii") for text in values]
            for client_id, values in allocation.items()
        },
    })
    ks = KeyServerEntity(KeyServerConfig(root / "ks-oprf-ristretto255-key.json", host="127.0.0.1", port=0))
    started = time.perf_counter()
    per_client: list[dict[str, object]] = []
    try:
        ks.start()
        def evaluate(client_id: str) -> dict[str, object]:
            client = ClientEntity(ClientConfig(
                client_id=client_id,
                ks_base_url=ks.base_url,
                label_store_path=root / "client-label-stores" / f"{client_id}.json",
                oprf_batch_size=oprf_batch_size,
                timeout_seconds=120.0,
            ))
            item_started = time.perf_counter()
            labels = client.generate_protected_labels(allocation[client_id])
            return {
                "client_id": client_id,
                "record_count": len(allocation[client_id]),
                "label_count": len(labels),
                "oprf_wall_seconds": time.perf_counter() - item_started,
            }
        with ThreadPoolExecutor(max_workers=_CLIENTS) as executor:
            per_client = list(executor.map(evaluate, sorted(allocation)))
    finally:
        ks.close()
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "oprf_suite": OPRF_SUITE_IDENTIFIER,
        "status": "enabled",
        "configuration": {
            "clients": _CLIENTS,
            "records_per_client": _RECORDS_PER_CLIENT,
            "duplicate_ratio": _DUPLICATE_RATIO,
            "seed": _SEED,
            "oprf_batch_size": oprf_batch_size,
        },
        "prepared_data_sha256": source_digest,
        "allocation_file": "fixed-allocation.json",
        "oprf_execution_wall_seconds": time.perf_counter() - started,
        "oprf_client_metrics": sorted(per_client, key=lambda item: str(item["client_id"])),
    }
    _write_json(root / "manifest.json", manifest)
    return _load(root, source_digest)


def allocate_fixed_paper_records(prepared_data_path: Path) -> dict[str, tuple[str, ...]]:
    """Allocate the fixed paper dataset without creating or reading OPRF labels.

    The live protocol-ablation benchmark must use the same deterministic
    10-client, 1,024-record, r=0.30 allocation as the full evaluation while
    executing a fresh OPRF in every branch. This helper deliberately reads
    only the prepared corpus and never touches a precomputation directory or
    a protected-label store. 实时协议消融必须使用与完整评估相同的确定性 10 客户端、
    每客户端 1,024 条、r=0.30 分配，同时在每个分支重新执行 OPRF。该函数仅读取
    已处理语料，绝不访问预计算目录或保护标签存储。
    """
    return _allocate(_load_unique_records(Path(prepared_data_path)))


def load_precomputed_dataset(prepared_data_path: Path, directory: Path) -> PrecomputedOprfDataset:
    """Load an enabled dataset without contacting KS or recomputing OPRF.

    加载已启用数据集，不联系 KS 也不重新计算 OPRF。
    """
    return _load(Path(directory).resolve(), _records_digest(_load_unique_records(prepared_data_path)))


def precomputation_is_enabled(directory: Path) -> bool:
    """Read the persisted operator switch without loading OPRF records.

    读取持久化操作员开关，而不加载 OPRF 记录。
    """
    state_path = Path(directory).resolve() / "feature-state.json"
    if not state_path.is_file():
        return True
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read OPRF precomputation state / 无法读取 OPRF 预计算状态") from error
    return isinstance(payload, dict) and payload.get("status") == "enabled"


def _load(root: Path, source_digest: str) -> PrecomputedOprfDataset:
    """Validate persistent allocation and every base-client label store.

    验证持久分配及每个基础客户端标签存储。
    """
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected = {"clients": _CLIENTS, "records_per_client": _RECORDS_PER_CLIENT, "duplicate_ratio": _DUPLICATE_RATIO, "seed": _SEED, "oprf_batch_size": manifest["configuration"]["oprf_batch_size"]}
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != _SCHEMA_VERSION
            or manifest.get("oprf_suite") != OPRF_SUITE_IDENTIFIER
            or manifest.get("status") != "enabled"
            or manifest.get("prepared_data_sha256") != source_digest
            or manifest.get("configuration") != expected
        ):
            raise ValueError("manifest mismatch")
        allocation = json.loads((root / str(manifest["allocation_file"])).read_text(encoding="utf-8"))
        raw_clients = allocation["clients"]
        expected_ids = {f"client-{index}" for index in range(_CLIENTS)}
        if not isinstance(raw_clients, dict) or set(raw_clients) != expected_ids:
            raise ValueError("client IDs mismatch")
        records = {
            client_id: tuple(base64.b64decode(value.encode("ascii"), validate=True).decode("utf-8") for value in values)
            for client_id, values in raw_clients.items()
        }
        if any(len(values) != _RECORDS_PER_CLIENT for values in records.values()):
            raise ValueError("record count mismatch")
        if not (root / "ks-oprf-ristretto255-key.json").is_file() or any(
            not (root / "client-label-stores" / f"client-{index}.json").is_file()
            for index in range(_CLIENTS)
        ):
            raise ValueError("persistent OPRF material is incomplete")
    except (OSError, KeyError, TypeError, ValueError, UnicodeDecodeError, binascii.Error) as error:
        raise ValueError(
            "precomputed DwT-FL OPRF dataset is missing or incompatible; use a new Ristretto255 cache directory and run the enable command once / "
            "预计算 DwT-FL OPRF 数据集缺失或不兼容；请使用新的 Ristretto255 缓存目录并先执行一次启用命令"
        ) from error
    return PrecomputedOprfDataset(root, records, manifest)


def _load_unique_records(path: Path) -> tuple[PreparedRecord, ...]:
    """Load a stable unique-text source pool. / 加载稳定的唯一文本源池。"""
    unique: list[PreparedRecord] = []
    seen: set[str] = set()
    for record in iter_prepared_records(Path(path).resolve()):
        if record.text not in seen:
            unique.append(record)
            seen.add(record.text)
    if not unique:
        raise ValueError("prepared data has no usable text / 预处理数据没有可用文本")
    return tuple(unique)


def _allocate(records: Sequence[PreparedRecord]) -> dict[str, tuple[str, ...]]:
    """Allocate pairwise duplicates with the same fixed paper baseline.

    使用相同固定论文基线分配两两重复数据。
    """
    duplicate_slots = [round(_RECORDS_PER_CLIENT * _DUPLICATE_RATIO)] * _CLIENTS
    required = sum(duplicate_slots) // 2 + sum(_RECORDS_PER_CLIENT - value for value in duplicate_slots)
    if len(records) < required:
        raise ValueError("prepared data is too small for fixed OPRF allocation / 预处理数据不足以固定 OPRF 分配")
    offset = int.from_bytes(hashlib.sha256(f"dwt-precompute:{_SEED}".encode("utf-8")).digest()[:8], "big") % len(records)
    selected = (tuple(records[offset:]) + tuple(records[:offset]))[:required]
    result = {f"client-{index}": [] for index in range(_CLIENTS)}
    remaining = duplicate_slots.copy()
    cursor = 0
    while any(remaining):
        candidates = sorted((index for index, value in enumerate(remaining) if value), key=lambda index: (-remaining[index], index))
        first, second = candidates[:2]
        result[f"client-{first}"].append(selected[cursor].text)
        result[f"client-{second}"].append(selected[cursor].text)
        cursor += 1
        remaining[first] -= 1
        remaining[second] -= 1
    for index in range(_CLIENTS):
        count = _RECORDS_PER_CLIENT - duplicate_slots[index]
        result[f"client-{index}"].extend(record.text for record in selected[cursor:cursor + count])
        cursor += count
    return {client_id: tuple(values) for client_id, values in result.items()}


def allocate_joining_records(
    prepared_records: Sequence[PreparedRecord],
    base_records: dict[str, Sequence[str]],
    *,
    joining_clients: int,
    duplicate_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Allocate fresh joining-client data without changing fixed base datasets.

    在不改变固定基础数据集的情况下分配新的加入客户端数据。

    Each joiner receives the requested number of texts already held by legacy
    clients and fills the remainder from source texts absent from the base
    allocation. The joiner must therefore run OPRF for all of its own data,
    while existing clients reuse their persistent labels. 每个加入者取得指定数量的
    既有客户端文本，其余从基础分配未使用的源文本填充。因此加入者必须为自身全部数据
    运行 OPRF，而既有客户端复用其持久标签。
    """
    if joining_clients < 1 or not 0.0 <= duplicate_ratio <= 1.0:
        raise ValueError("joining allocation parameters are invalid / 加入分配参数无效")
    duplicate_count = round(_RECORDS_PER_CLIENT * duplicate_ratio)
    legacy = tuple(dict.fromkeys(text for values in base_records.values() for text in values))
    if duplicate_count > len(legacy):
        raise ValueError("not enough legacy texts for joining duplicates / 既有文本不足以提供加入重复项")
    used = set(legacy)
    pool = tuple(record.text for record in prepared_records if record.text not in used)
    required = joining_clients * (_RECORDS_PER_CLIENT - duplicate_count)
    if len(pool) < required:
        raise ValueError("prepared data is too small for joining unique records / 预处理数据不足以提供加入唯一记录")
    offset = int.from_bytes(hashlib.sha256(f"joining:{seed}".encode("utf-8")).digest()[:8], "big") % len(legacy)
    result: dict[str, list[str]] = {}
    unique_cursor = 0
    for index in range(joining_clients):
        shared = [legacy[(offset + index * duplicate_count + item) % len(legacy)] for item in range(duplicate_count)]
        unique = list(pool[unique_cursor:unique_cursor + _RECORDS_PER_CLIENT - duplicate_count])
        unique_cursor += len(unique)
        result[f"client-{_CLIENTS + index}"] = shared + unique
    return result


def _records_digest(records: Sequence[PreparedRecord]) -> str:
    """Hash the complete source pool for provenance validation.

    为溯源验证散列完整源池。
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """Atomically persist a JSON artifact. / 原子持久化 JSON 产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
