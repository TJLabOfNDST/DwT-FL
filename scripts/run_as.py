'Run the DwT-FL Aggregation Server session entity over HTTP.'

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path


# Permit direct execution from the project root without package installation.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.entities import AggregationServerConfig, AggregationServerEntity


def parse_arguments() -> argparse.Namespace:
    'Parse portable AS deployment and experiment-capacity arguments.'
    parser = argparse.ArgumentParser(
        description="Run the DwT-FL Aggregation Server /  DwT-FL "
    )
    parser.add_argument("--host", default="0.0.0.0", help="AS bind host / AS ")
    parser.add_argument("--port", type=int, default=18080, help="AS port / AS ")
    parser.add_argument("--capacity", type=int, default=100_000, help="task capacity / ")
    parser.add_argument(
        "--max-clients",
        type=int,
        default=10_000,
        help="SID capacity / SID ",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=200_000,
        help="owner-edge capacity / ",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=5.0,
        help="heartbeat interval in seconds / ",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=300.0,
        help="offline timeout in seconds / ",
    )
    parser.add_argument(
        "--model-update-directory",
        type=Path,
        default=Path("results") / "model-updates",
        help="AS-local checkpoint directory / AS ",
    )
    parser.add_argument(
        "--max-model-update-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="maximum accepted update size / ",
    )
    parser.add_argument(
        "--backend-workers",
        type=int,
        default=32,
        help="maximum simultaneous AS request handlers / AS ",
    )
    parser.add_argument(
        "--claim-mode",
        choices=("cas", "mutex"),
        default="cas",
        help="CAS production mode or mutex ablation / CAS  Mutex ",
    )
    parser.add_argument(
        "--recovery-index-mode",
        choices=("inverse", "scan"),
        default="inverse",
        help="inverse production recovery or full-scan ablation / ",
    )
    parser.add_argument(
        "--disable-history-scheduling",
        action="store_true",
        help="disable previous-trainer preference for ablation / ",
    )
    parser.add_argument(
        "--native-library",
        type=Path,
        default=None,
        help="compiled native index library / ",
    )
    parser.add_argument(
        "--evaluation-reset-token",
        default=os.environ.get("DBTFL_EVALUATION_RESET_TOKEN"),
        help="dedicated experiment reset token / ",
    )
    return parser.parse_args()


def main() -> int:
    'Start the AS until Ctrl+C on Windows, WSL, or Ubuntu.'
    arguments = parse_arguments()
    entity = AggregationServerEntity(
        AggregationServerConfig(
            capacity=arguments.capacity,
            max_clients=arguments.max_clients,
            max_edges=arguments.max_edges,
            host=arguments.host,
            port=arguments.port,
            heartbeat_interval_seconds=arguments.heartbeat_interval,
            heartbeat_timeout_seconds=arguments.heartbeat_timeout,
            model_update_directory=arguments.model_update_directory,
            max_model_update_bytes=arguments.max_model_update_bytes,
            backend_worker_count=arguments.backend_workers,
            claim_mode=arguments.claim_mode,
            recovery_index_mode=arguments.recovery_index_mode,
            history_scheduling_enabled=not arguments.disable_history_scheduling,
            native_library_path=arguments.native_library,
            evaluation_reset_token=arguments.evaluation_reset_token,
        )
    )
    entity.start()
    print(f"DwT-FL AS listening at {entity.base_url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        return 0
    finally:
        entity.close()


if __name__ == "__main__":
    raise SystemExit(main())
