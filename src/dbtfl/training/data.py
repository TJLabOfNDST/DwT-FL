'Portable CSV-to-JSONL preparation for causal-language-model training.'

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
"""Schema version for prepared local-training records. / """


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    'One locally retained text record ready for tokenization.'

    record_id: str
    text: str
    metadata: Mapping[str, str]

    def to_json_object(self) -> dict[str, object]:
        'Return the stable on-disk representation.'
        return {
            "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
            "record_id": self.record_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PreparedDatasetManifest:
    'Reproducibility metadata emitted beside prepared data splits.'

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
    'Convert a text CSV into deterministic train/validation JSONL files.\n    The model sees only ``text_column``.  Every other non-export-index CSV\n    column remains in local metadata, which prevents fields such as labels or\n    keywords from silently becoming language-model input.\n    ``text_column``'
    source_csv = Path(source_csv).resolve()
    output_directory = Path(output_directory).resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(f"CSV file was not found /  CSV {source_csv}")
    if not text_column or not id_column:
        raise ValueError("text_column and id_column must be non-empty / ")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1) /  [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer / seed ")

    records = list(_read_csv_records(source_csv, text_column=text_column, id_column=id_column))
    if not records:
        raise ValueError("CSV contains no usable text rows / CSV ")
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record IDs must be unique / ")

    train_records: list[PreparedRecord] = []
    validation_records: list[PreparedRecord] = []
    for record in records:
        if _belongs_to_validation(record.record_id, seed, validation_fraction):
            validation_records.append(record)
        else:
            train_records.append(record)
    if not train_records:
        raise ValueError("validation split consumed every record / ")

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
    'Yield validated prepared records without loading an entire split.'
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                raw_record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number} /  {line_number}  JSONL "
                ) from error
            yield _prepared_record_from_object(raw_record, line_number)


def materialize_hot_training_split(
    prepared_jsonl: Path,
    hot_records: Sequence[str | bytes],
    output_path: Path,
) -> int:
    "Write only paper-TRAIN records into one client-local JSONL split.\n    ``hot_records`` comes from ``ClientEntity.route_claim_decisions`` and is\n    never sent to AS.  Exact duplicate plaintext records appear once so a local\n    repeated row cannot undo the protocol's deduplication decision.\n    ``hot_records``\n    AS"
    if isinstance(hot_records, (str, bytes)) or not isinstance(hot_records, Sequence):
        raise TypeError("hot_records must be a sequence / hot_records ")
    selected_bytes: set[bytes] = set()
    for record in hot_records:
        if isinstance(record, str):
            selected_bytes.add(record.encode("utf-8"))
        elif isinstance(record, bytes):
            selected_bytes.add(record)
        else:
            raise TypeError("hot records must be str or bytes /  str  bytes")
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
            ""
        )
    _atomic_write_jsonl(Path(output_path), selected_records)
    return len(selected_records)


def _read_csv_records(
    source_csv: Path,
    *,
    text_column: str,
    id_column: str,
) -> Iterator[PreparedRecord]:
    'Read UTF-8 CSV rows while dropping spreadsheet-export index columns.'
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if text_column not in fieldnames or id_column not in fieldnames:
            raise ValueError(
                "CSV must contain configured text and id columns / "
                "CSV "
            )
        for source_row, row in enumerate(reader, start=2):
            raw_text = row.get(text_column)
            raw_id = row.get(id_column)
            if raw_text is None or raw_id is None:
                raise ValueError(
                    f"missing required value at CSV row {source_row} / "
                    f"CSV  {source_row} "
                )
            text = _normalize_text(raw_text)
            record_id = raw_id.strip()
            if not text or not record_id:
                raise ValueError(
                    f"empty text or ID at CSV row {source_row} / "
                    f"CSV  {source_row}  ID "
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
    'Assign one record by a stable ID hash, independent of CSV row order.'
    if fraction == 0.0:
        return False
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def _prepared_record_from_object(raw_record: Any, line_number: int) -> PreparedRecord:
    'Validate one record decoded from the portable JSONL schema.'
    if not isinstance(raw_record, dict) or set(raw_record) != {
        "schema_version",
        "record_id",
        "text",
        "metadata",
    }:
        raise ValueError(
            f"invalid prepared record at line {line_number} / "
            f" {line_number} "
        )
    if raw_record["schema_version"] != PREPARED_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported prepared-data schema / ")
    record_id = raw_record["record_id"]
    text = raw_record["text"]
    metadata = raw_record["metadata"]
    if not isinstance(record_id, str) or not record_id or not isinstance(text, str) or not text:
        raise ValueError(f"invalid text or ID at line {line_number} /  {line_number}  ID ")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError(f"invalid metadata at line {line_number} /  {line_number} ")
    return PreparedRecord(record_id=record_id, text=text, metadata=metadata)


def _normalize_text(value: str) -> str:
    'Normalize line endings only; preserve the text content for training.'
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _atomic_write_jsonl(path: Path, records: list[PreparedRecord]) -> None:
    'Atomically replace one JSONL output after a complete successful write.'
    serialized = "".join(
        json.dumps(record.to_json_object(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_write_text(path, serialized)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    'Atomically write reproducibility metadata as UTF-8 JSON.'
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    'Write one text file through a same-directory temporary file.'
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
    'Return one streaming SHA-256 digest without retaining file bytes.'
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
