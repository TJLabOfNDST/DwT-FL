"""Portable CSV-to-JSONL preparation for causal-language-model training.

面向因果语言模型训练的跨平台 CSV 到 JSONL 数据预处理。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final


PREPARED_DATASET_SCHEMA_VERSION: Final[str] = "1.0"
"""Schema version for prepared local-training records. / 本地训练预处理记录的模式版本。"""


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """One locally retained text record ready for tokenization.

    一条已准备好可供分词的本地保留文本记录。
    """

    record_id: str
    text: str
    metadata: Mapping[str, str]

    def to_json_object(self) -> dict[str, object]:
        """Return the stable on-disk representation. / 返回稳定的磁盘表示。"""
        return {
            "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
            "record_id": self.record_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PreparedDatasetManifest:
    """Reproducibility metadata emitted beside prepared data splits.

    与预处理数据划分一同写入的可复现实验元数据。
    """

    schema_version: str
    source_csv: str
    source_sha256: str
    text_column: str
    id_column: str
    seed: int
    validation_fraction: float
    total_records: int
    train_records: int
    validation_records: int


def prepare_csv_dataset(
    source_csv: Path,
    output_directory: Path,
    *,
    text_column: str = "processed_title",
    id_column: str = "id",
    validation_fraction: float = 0.05,
    seed: int = 17,
) -> PreparedDatasetManifest:
    """Convert a text CSV into deterministic train/validation JSONL files.

    将文本 CSV 转换为确定性的训练集与验证集 JSONL 文件。

    The model sees only ``text_column``.  Every other non-export-index CSV
    column remains in local metadata, which prevents fields such as labels or
    keywords from silently becoming language-model input.  模型只读取
    ``text_column``；其他非导出索引列保留为本地元数据，避免标签、关键词等字段被
    悄然混入语言模型输入。
    """
    source_csv = Path(source_csv).resolve()
    output_directory = Path(output_directory).resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(f"CSV file was not found / 未找到 CSV 文件：{source_csv}")
    if not text_column or not id_column:
        raise ValueError("text_column and id_column must be non-empty / 文本列和标识列不能为空")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1) / 验证集比例必须位于 [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer / seed 必须是整数")

    records = list(_read_csv_records(source_csv, text_column=text_column, id_column=id_column))
    if not records:
        raise ValueError("CSV contains no usable text rows / CSV 不包含可用文本行")
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record IDs must be unique / 数据记录标识必须唯一")

    train_records: list[PreparedRecord] = []
    validation_records: list[PreparedRecord] = []
    for record in records:
        if _belongs_to_validation(record.record_id, seed, validation_fraction):
            validation_records.append(record)
        else:
            train_records.append(record)
    if not train_records:
        raise ValueError("validation split consumed every record / 验证集划分占用了全部记录")

    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_directory / "records.jsonl", records)
    _atomic_write_jsonl(output_directory / "train.jsonl", train_records)
    _atomic_write_jsonl(output_directory / "validation.jsonl", validation_records)
    manifest = PreparedDatasetManifest(
        schema_version=PREPARED_DATASET_SCHEMA_VERSION,
        source_csv=str(source_csv),
        source_sha256=_sha256_file(source_csv),
        text_column=text_column,
        id_column=id_column,
        seed=seed,
        validation_fraction=validation_fraction,
        total_records=len(records),
        train_records=len(train_records),
        validation_records=len(validation_records),
    )
    _atomic_write_json(output_directory / "manifest.json", asdict(manifest))
    return manifest


def iter_prepared_records(path: Path) -> Iterator[PreparedRecord]:
    """Yield validated prepared records without loading an entire split.

    逐条产生经验证的预处理记录，而不一次性加载整个数据划分。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                raw_record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number} / 第 {line_number} 行 JSONL 无效"
                ) from error
            yield _prepared_record_from_object(raw_record, line_number)


def materialize_hot_training_split(
    prepared_jsonl: Path,
    hot_records: Sequence[str | bytes],
    output_path: Path,
) -> int:
    """Write only paper-TRAIN records into one client-local JSONL split.

    仅将论文中标记为 TRAIN 的记录写入一个客户端本地 JSONL 划分。

    ``hot_records`` comes from ``ClientEntity.route_claim_decisions`` and is
    never sent to AS.  Exact duplicate plaintext records appear once so a local
    repeated row cannot undo the protocol's deduplication decision.
    ``hot_records`` 来自 ``ClientEntity.route_claim_decisions``，绝不会发送给
    AS；完全相同的明文记录仅出现一次，因此本地重复行不会抵消协议去重决策。
    """
    if isinstance(hot_records, (str, bytes)) or not isinstance(hot_records, Sequence):
        raise TypeError("hot_records must be a sequence / hot_records 必须是一个序列")
    selected_bytes: set[bytes] = set()
    for record in hot_records:
        if isinstance(record, str):
            selected_bytes.add(record.encode("utf-8"))
        elif isinstance(record, bytes):
            selected_bytes.add(record)
        else:
            raise TypeError("hot records must be str or bytes / 热队列记录必须为 str 或 bytes")
    selected_records: list[PreparedRecord] = []
    emitted_bytes: set[bytes] = set()
    for record in iter_prepared_records(prepared_jsonl):
        encoded_text = record.text.encode("utf-8")
        if encoded_text in selected_bytes and encoded_text not in emitted_bytes:
            selected_records.append(record)
            emitted_bytes.add(encoded_text)
    missing = selected_bytes.difference(emitted_bytes)
    if missing:
        raise ValueError(
            "hot records are absent from prepared data / "
            "热队列记录未出现在预处理数据中"
        )
    _atomic_write_jsonl(Path(output_path), selected_records)
    return len(selected_records)


def _read_csv_records(
    source_csv: Path,
    *,
    text_column: str,
    id_column: str,
) -> Iterator[PreparedRecord]:
    """Read UTF-8 CSV rows while dropping spreadsheet-export index columns.

    读取 UTF-8 CSV 行，同时丢弃电子表格导出的索引列。
    """
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if text_column not in fieldnames or id_column not in fieldnames:
            raise ValueError(
                "CSV must contain configured text and id columns / "
                "CSV 必须包含所配置的文本列和标识列"
            )
        for source_row, row in enumerate(reader, start=2):
            raw_text = row.get(text_column)
            raw_id = row.get(id_column)
            if raw_text is None or raw_id is None:
                raise ValueError(
                    f"missing required value at CSV row {source_row} / "
                    f"CSV 第 {source_row} 行缺少必填值"
                )
            text = _normalize_text(raw_text)
            record_id = raw_id.strip()
            if not text or not record_id:
                raise ValueError(
                    f"empty text or ID at CSV row {source_row} / "
                    f"CSV 第 {source_row} 行文本或 ID 为空"
                )
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {text_column, id_column}
                and not key.lower().startswith("unnamed:")
                and value is not None
            }
            yield PreparedRecord(record_id=record_id, text=text, metadata=metadata)


def _belongs_to_validation(record_id: str, seed: int, fraction: float) -> bool:
    """Assign one record by a stable ID hash, independent of CSV row order.

    基于稳定 ID 哈希划分记录，不依赖 CSV 行顺序。
    """
    if fraction == 0.0:
        return False
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def _prepared_record_from_object(raw_record: Any, line_number: int) -> PreparedRecord:
    """Validate one record decoded from the portable JSONL schema.

    验证一条从可移植 JSONL 模式解析出的记录。
    """
    if not isinstance(raw_record, dict) or set(raw_record) != {
        "schema_version",
        "record_id",
        "text",
        "metadata",
    }:
        raise ValueError(
            f"invalid prepared record at line {line_number} / "
            f"第 {line_number} 行预处理记录无效"
        )
    if raw_record["schema_version"] != PREPARED_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported prepared-data schema / 不支持的预处理数据模式")
    record_id = raw_record["record_id"]
    text = raw_record["text"]
    metadata = raw_record["metadata"]
    if not isinstance(record_id, str) or not record_id or not isinstance(text, str) or not text:
        raise ValueError(f"invalid text or ID at line {line_number} / 第 {line_number} 行文本或 ID 无效")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError(f"invalid metadata at line {line_number} / 第 {line_number} 行元数据无效")
    return PreparedRecord(record_id=record_id, text=text, metadata=metadata)


def _normalize_text(value: str) -> str:
    """Normalize line endings only; preserve the text content for training.

    仅规范化换行符，并保留用于训练的文本内容。
    """
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _atomic_write_jsonl(path: Path, records: list[PreparedRecord]) -> None:
    """Atomically replace one JSONL output after a complete successful write.

    在完整成功写入后，原子替换一个 JSONL 输出文件。
    """
    serialized = "".join(
        json.dumps(record.to_json_object(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_write_text(path, serialized)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write reproducibility metadata as UTF-8 JSON.

    将可复现元数据以 UTF-8 JSON 原子写入。
    """
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Write one text file through a same-directory temporary file.

    通过同目录临时文件写入一个文本文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    """Return one streaming SHA-256 digest without retaining file bytes.

    流式计算一个 SHA-256 摘要，而不保留整个文件的字节。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
