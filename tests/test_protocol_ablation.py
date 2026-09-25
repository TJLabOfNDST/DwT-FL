'Regression tests for paired cached-label protocol ablation reporting.'

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_protocol_ablation_benchmark.py"


def _load_script_module():
    'Load the standalone benchmark script without starting a benchmark.'
    specification = importlib.util.spec_from_file_location(
        "dbtfl_protocol_ablation_script", SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ProtocolAblationReportingTest(unittest.TestCase):
    'Keep every paper comparison paired with its complete DwT-FL baseline.'

    @classmethod
    def setUpClass(cls) -> None:
        'Load reporting helpers once.'
        cls.script = _load_script_module()

    def test_paired_result_exposes_baseline_existing_deltas_and_recovery_definition(self) -> None:
        'Verify flat CSV aliases and explicit paired JSON use one definition.'
        config = self.script.BenchmarkConfig(
            clients=10,
            records_per_client=1024,
            duplicate_ratio=0.30,
            backend_workers=4,
            rpc_timeout_seconds=30.0,
            heartbeat_interval_seconds=5.0,
            heartbeat_timeout_seconds=30.0,
            recovery_timeout_seconds=1.0,
            oprf_mode="live",
            oprf_batch_size=1024,
        )

        def completed(name: str, **metrics: float) -> dict[str, object]:
            return {"status": "completed", "name": name, "metrics": metrics}

        result = self.script._public_repetition(config, 0, {
            "cached_oprf": completed("cached_oprf"),
            "dwtfl_cas_baseline": completed(
                "dwtfl_cas_baseline", post_oprf_protocol_seconds=1.0,
                claim_seconds=0.2, full_protocol_seconds=2.0,
            ),
            "existing_mutex_claim": completed(
                "existing_mutex_claim", post_oprf_protocol_seconds=1.5,
                claim_seconds=0.5, full_protocol_seconds=2.7,
            ),
            "dwtfl_inverse_index_recovery": completed(
                "dwtfl_inverse_index_recovery",
                recovery_latency_from_detection_seconds=0.4,
                end_to_end_recovery_seconds=1.6,
                full_protocol_seconds=3.2,
            ),
            "existing_full_scan_recovery": completed(
                "existing_full_scan_recovery",
                recovery_latency_from_detection_seconds=0.9,
                end_to_end_recovery_seconds=2.1,
                full_protocol_seconds=4.0,
            ),
            "dwtfl_history_scheduling": completed(
                "dwtfl_history_scheduling",
                two_round_post_oprf_protocol_seconds=1.2,
                previous_trainer_reuse_ratio=0.8,
                full_protocol_seconds=3.0,
            ),
            "existing_no_history_scheduling": completed(
                "existing_no_history_scheduling",
                two_round_post_oprf_protocol_seconds=1.7,
                previous_trainer_reuse_ratio=0.2,
                full_protocol_seconds=3.9,
            ),
        })

        paired = result["baseline_vs_existing"]
        self.assertEqual(result["input_record_copies"], 10_240)
        self.assertAlmostEqual(
            paired["cas_vs_mutex"]["existing_minus_dwtfl_claim_seconds"], 0.3,
        )
        self.assertAlmostEqual(
            paired["inverse_index_vs_full_scan"]
            ["existing_minus_dwtfl_recovery_latency_seconds"],
            0.5,
        )
        self.assertAlmostEqual(
            paired["history_vs_stateless"]
            ["dwtfl_minus_existing_reuse_ratio"],
            0.6,
        )
        self.assertAlmostEqual(
            result["inverse_index_vs_scan"]
            ["scan_minus_inverse_recovery_latency_seconds"],
            0.5,
        )

    def test_formal_four_run_summary_discards_only_extreme_values(self) -> None:
        'Verify four-run reporting keeps the two middle observations.'
        summary = self.script._trimmed_metric_summary([10.0, 8.0, 20.0, 12.0], 4)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["successful_repetitions"], 4)
        self.assertEqual(summary["trimmed_repetitions"], 2)
        self.assertEqual(summary["excluded_min"], 8.0)
        self.assertEqual(summary["excluded_max"], 20.0)
        self.assertEqual(summary["trimmed_mean"], 11.0)

    def test_summary_does_not_average_missing_repetitions(self) -> None:
        'Verify a failed repetition remains visible instead of being hidden.'
        summary = self.script._trimmed_metric_summary([1.0, 2.0, 3.0], 4)

        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["trimmed_mean"])


if __name__ == "__main__":
    unittest.main()
