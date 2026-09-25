'Regression tests for the dedicated CAS/mutex contention benchmark.\nCAS'

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_cas_mutex_contention_benchmark.py"


def _load_script_module():
    'Load the standalone script without executing its benchmark.'
    specification = importlib.util.spec_from_file_location(
        "dbtfl_cas_mutex_contention_script", SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class CasMutexContentionBenchmarkTest(unittest.TestCase):
    'Protect the exact paired hot-contention workload definition.'

    @classmethod
    def setUpClass(cls) -> None:
        'Load benchmark helpers once.'
        cls.script = _load_script_module()

    def test_hot_records_are_shared_by_all_clients(self) -> None:
        'Verify only one deterministic record set is generated per repetition.'
        config = self.script.ContentionConfig(
            clients=10,
            records_per_client=4,
            backend_workers=4,
            oprf_batch_size=4,
            rpc_timeout_seconds=30.0,
            heartbeat_interval_seconds=5.0,
            heartbeat_timeout_seconds=30.0,
            seed=17,
        )

        records = self.script._hot_records(config)

        self.assertEqual(len(records), 4)
        self.assertEqual(len(set(records)), 4)
        self.assertEqual(records[0], "cas-mutex-hot-record:17:00000")

    def test_four_run_summary_drops_exactly_one_minimum_and_maximum(self) -> None:
        'Verify the formal report averages the middle two observations.'
        summary = self.script._trimmed([10.0, 8.0, 20.0, 12.0], expected=4)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["excluded_min"], 8.0)
        self.assertEqual(summary["excluded_max"], 20.0)
        self.assertEqual(summary["trimmed_mean"], 11.0)

    def test_report_accepts_every_paired_metric_column(self) -> None:
        'Verify CSV persistence keeps all paired timing fields.'
        fields = (
            "oprf_wall_seconds", "ks_oprf_evaluation_compute_seconds",
            "registration_seconds", "claim_seconds", "post_oprf_protocol_seconds",
            "full_protocol_seconds",
        )
        metrics = {field: float(index + 1) for index, field in enumerate(fields)}
        result = {
            "repetition": 0,
            "status": "completed",
            "branches": {
                "cas": {"status": "completed", "metrics": metrics},
                "mutex": {"status": "completed", "metrics": metrics},
            },
            "mutex_minus_cas": {field: 0.0 for field in fields},
        }
        config = self.script.ContentionConfig(
            clients=2,
            records_per_client=8,
            backend_workers=2,
            oprf_batch_size=8,
            rpc_timeout_seconds=30.0,
            heartbeat_interval_seconds=5.0,
            heartbeat_timeout_seconds=30.0,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            output = Path(temporary_name)
            self.script._write_reports(output, config, expected_repetitions=1, results=[result])
            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            header = (output / "cas_mutex_metrics.csv").read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(payload["results"][0]["status"], "completed")
        self.assertIn("mutex_minus_cas_full_protocol_seconds", header)


if __name__ == "__main__":
    unittest.main()
