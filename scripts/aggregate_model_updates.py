'Request one AS-side FedAvg aggregation for an explicit synchronous round.'

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Permit direct execution from the project root without package installation.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.communication import AggregationServerPath, JsonHttpClient, WireMessage
from dbtfl.entities.aggregation_server import AS_MODEL_AGGREGATE_REQUEST


def parse_arguments() -> argparse.Namespace:
    'Parse one AS endpoint and the exact accepted update participants.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-url", required=True, help="AS HTTP base URL / AS HTTP  URL")
    parser.add_argument("--round", type=int, required=True, help="synchronous round / ")
    parser.add_argument(
        "--sids",
        required=True,
        help="comma-separated submitted SIDs /  SID",
    )
    return parser.parse_args()


def main() -> int:
    'Request FedAvg and print the AS-confirmed global checkpoint metadata.'
    arguments = parse_arguments()
    expected_sids = [int(value) for value in arguments.sids.split(",") if value.strip()]
    request = WireMessage.create(
        AS_MODEL_AGGREGATE_REQUEST,
        {"round": arguments.round, "expected_sids": expected_sids},
    )
    response = JsonHttpClient(arguments.as_url).send(
        AggregationServerPath.AGGREGATE_MODEL_UPDATES.value,
        request,
    )
    print(dict(response.payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
