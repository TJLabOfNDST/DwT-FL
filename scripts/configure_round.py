'Configure the fixed FedAvg participant roster for one experiment round.'

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dbtfl.communication import AggregationServerPath, JsonHttpClient, WireMessage
from dbtfl.entities.aggregation_server import AS_CONFIGURE_ROUND_REQUEST


def parse_arguments() -> argparse.Namespace:
    'Parse one AS endpoint, round, and exact synchronized client roster.'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-url", required=True, help="AS HTTP base URL / AS HTTP  URL")
    parser.add_argument("--round", type=int, required=True, help="round to configure / ")
    parser.add_argument(
        "--sids",
        required=True,
        help="comma-separated registered participant SIDs /  SID",
    )
    return parser.parse_args()


def main() -> int:
    'Freeze a roster before any client update for this round arrives.'
    arguments = parse_arguments()
    participant_sids = [int(value) for value in arguments.sids.split(",") if value.strip()]
    response = JsonHttpClient(arguments.as_url).send(
        AggregationServerPath.CONFIGURE_ROUND.value,
        WireMessage.create(
            AS_CONFIGURE_ROUND_REQUEST,
            {"round": arguments.round, "participant_sids": participant_sids},
        ),
    )
    print(dict(response.payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
