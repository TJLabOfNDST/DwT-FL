'Regression tests for the protocol-only communication benchmark.'

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _benchmark_module() -> object:
    'Load the executable benchmark without running its CLI entry point.'
    module_name = "dbtfl_communication_metadata_benchmark_test"
    specification = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "scripts" / "run_communication_metadata_benchmark.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


class CommunicationMetadataBenchmarkTest(unittest.TestCase):
    "Check the benchmark's allocation contract before real OPRF runs."

    def test_default_data_scale_matches_the_main_1024_record_baseline(self) -> None:
        'Keep reportable communication runs aligned with the FL baseline.'
        benchmark = _benchmark_module()
        with patch.object(sys, "argv", ["run_communication_metadata_benchmark.py"]):
            arguments = benchmark.parse_arguments()

        self.assertEqual(arguments.records_per_client, 1024)

    def test_pairwise_allocation_never_repeats_one_text_within_a_client(self) -> None:
        'Allow cross-client overlap while preventing local OPRF-label duplicates.'
        benchmark = _benchmark_module()
        prepared = tuple(
            [SimpleNamespace(text="repeated source text") for _ in range(12)]
            + [SimpleNamespace(text=f"unique prepared text {index}") for index in range(500)]
        )
        for clients in (2, 4, 6, 8, 10):
            for ratio in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
                case = benchmark._Case(
                    "test",
                    "ratio",
                    ratio,
                    clients,
                    32,
                    ratio,
                    4,
                )
                allocated = benchmark._pairwise_records(case, 0, prepared)

                self.assertEqual(set(allocated), {f"client-{index}" for index in range(clients)})
                for records in allocated.values():
                    self.assertEqual(len(records), 32)
                    self.assertEqual(len(records), len(set(records)))

if __name__ == "__main__":
    unittest.main()
