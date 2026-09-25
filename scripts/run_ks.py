'Run the DwT-FL Key Server OPRF entity on a local or cloud host.'

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# Permit direct execution from the project root without package installation.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.entities import KeyServerConfig, KeyServerEntity


def parse_arguments() -> argparse.Namespace:
    'Parse portable KS deployment arguments.'
    parser = argparse.ArgumentParser(
        description="Run the DwT-FL Key Server OPRF endpoint /  DwT-FL  OPRF "
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="interface to bind (default: all interfaces) / ",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18081,
        help="KS port (default: 18081) / KS 18081",
    )
    parser.add_argument(
        "--key-path",
        type=Path,
        default=Path("secrets") / "ks-oprf-ristretto255-secret.json",
        help="host-local KS private-key path /  KS ",
    )
    return parser.parse_args()


def main() -> int:
    'Start the KS until Ctrl+C on Windows, WSL, or Ubuntu.'
    arguments = parse_arguments()
    entity = KeyServerEntity(
        KeyServerConfig(
            key_path=arguments.key_path,
            host=arguments.host,
            port=arguments.port,
        )
    )
    entity.start()
    print(f"DwT-FL KS OPRF listening at {entity.base_url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        return 0
    finally:
        entity.close()


if __name__ == "__main__":
    raise SystemExit(main())
