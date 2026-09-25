"""Run one local DwT-FL protocol round without paper evaluation workloads.

The demo starts an in-process Key Server and Aggregation Server, then runs two
clients through blind OPRF labeling, protected-label registration, CAS task
claims, a real Safetensors model-update upload, FedAvg, and global-model
download. It uses deterministic tiny tensors rather than language-model
training, so its purpose is protocol verification rather than benchmarking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.entities import (
    AggregationServerConfig,
    AggregationServerEntity,
    ClientConfig,
    ClientEntity,
    KeyServerConfig,
    KeyServerEntity,
)


ROUND_ID = 1
CLIENT_RECORDS = (
    ("demo-client-a", ("shared-record", "client-a-only"), 1.0),
    ("demo-client-b", ("shared-record", "client-b-only"), 2.0),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results" / "protocol-demo",
        help="parent directory for one retained demo run",
    )
    parser.add_argument(
        "--native-library",
        type=Path,
        default=None,
        help="optional atomic_word shared-library path",
    )
    return parser.parse_args()


def _training_decisions(decisions: tuple[Any, ...]) -> tuple[Any, ...]:
    result = tuple(decision for decision in decisions if decision.operation == "TRAIN")
    if not result:
        raise RuntimeError("each demo client must receive at least one TRAIN decision")
    return result


def _write_checkpoint(path: Path, value: float) -> None:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError(
            "The protocol demo requires the training extra. "
            "Install it with: pip install -e .[training]"
        ) from error
    save_file({"weight": torch.tensor([value], dtype=torch.float32)}, str(path))


def _read_weight(path: Path) -> float:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("Safetensors is required to inspect the global model") from error
    state = load_file(str(path))
    weight = state.get("weight")
    if weight is None or weight.numel() != 1:
        raise RuntimeError("global model does not contain the expected scalar weight")
    return float(weight.item())


def _decision_summary(decisions: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": decision.task_id,
            "operation": decision.operation,
            "state": decision.state,
        }
        for decision in decisions
    ]


def run_demo(output_parent: Path, native_library: Path | None) -> tuple[Path, dict[str, Any]]:
    run_directory = output_parent.resolve() / f"run-{uuid.uuid4().hex[:12]}"
    run_directory.mkdir(parents=True, exist_ok=False)
    key_path = run_directory / "ks-key.json"
    model_directory = run_directory / "model-updates"

    with KeyServerEntity(KeyServerConfig(key_path=key_path, host="127.0.0.1", port=0)) as key_server:
        with AggregationServerEntity(
            AggregationServerConfig(
                capacity=16,
                max_clients=len(CLIENT_RECORDS),
                max_edges=32,
                host="127.0.0.1",
                port=0,
                model_update_directory=model_directory,
                native_library_path=native_library,
            )
        ) as aggregation_server:
            clients = [
                ClientEntity(
                    ClientConfig(
                        client_id=client_id,
                        ks_base_url=key_server.base_url,
                        as_base_url=aggregation_server.base_url,
                        label_store_path=run_directory / f"{client_id}-labels.json",
                        model_chunk_bytes=64,
                    )
                )
                for client_id, _records, _value in CLIENT_RECORDS
            ]
            try:
                sessions = [client.connect_to_as() for client in clients]
                all_decisions: list[tuple[Any, ...]] = []
                queue_sizes: list[dict[str, int]] = []
                for client, (_client_id, records, _value) in zip(clients, CLIENT_RECORDS, strict=True):
                    registrations = client.register_records_with_as(records, created_round=ROUND_ID)
                    decisions = tuple(client.claim_registered_labels_at_as(registrations))
                    queues = client.route_claim_decisions(decisions)
                    all_decisions.append(decisions)
                    queue_sizes.append(
                        {"hot_records": len(queues.hot_records), "cold_records": len(queues.cold_records)}
                    )

                participant_sids = tuple(session.sid for session in sessions)
                clients[0].configure_federated_round_at_as(ROUND_ID, participant_sids)
                submissions = []
                local_weights = []
                sample_counts = []
                for index, (client, (_client_id, _records, value), decisions) in enumerate(
                    zip(clients, CLIENT_RECORDS, all_decisions, strict=True)
                ):
                    train_decisions = _training_decisions(decisions)
                    checkpoint_path = run_directory / f"client-{index}-update.safetensors"
                    _write_checkpoint(checkpoint_path, value)
                    submissions.append(
                        client.submit_model_update_at_as(
                            checkpoint_path,
                            train_decisions,
                            round_id=ROUND_ID,
                            sample_count=len(train_decisions),
                        )
                    )
                    local_weights.append(value)
                    sample_counts.append(len(train_decisions))

                aggregate = clients[0].aggregate_federated_round_at_as(ROUND_ID, participant_sids)
                global_path = clients[0].download_global_model_from_as(
                    ROUND_ID, run_directory / "global-model.safetensors"
                )
                observed_weight = _read_weight(global_path)
                expected_weight = sum(
                    value * count for value, count in zip(local_weights, sample_counts, strict=True)
                ) / sum(sample_counts)
                if not math.isclose(observed_weight, expected_weight, rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(
                        f"unexpected FedAvg weight: observed={observed_weight}, expected={expected_weight}"
                    )
                summary = {
                    "status": "ok",
                    "round": ROUND_ID,
                    "as_url": aggregation_server.base_url,
                    "ks_url": key_server.base_url,
                    "participant_sids": list(participant_sids),
                    "client_decisions": [_decision_summary(decisions) for decisions in all_decisions],
                    "local_queue_sizes": queue_sizes,
                    "committed_task_ids": [list(submission.committed_task_ids) for submission in submissions],
                    "fedavg": aggregate,
                    "global_model": str(global_path),
                    "expected_weight": expected_weight,
                    "observed_weight": observed_weight,
                }
            finally:
                for client in clients:
                    client.close()

    summary_path = run_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_directory, summary


def main() -> int:
    arguments = parse_arguments()
    run_directory, summary = run_demo(arguments.output_directory, arguments.native_library)
    print(f"DwT-FL protocol demo completed: {run_directory}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
