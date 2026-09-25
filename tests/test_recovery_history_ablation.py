'Regression tests for dedicated recovery and history ablations.'

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_recovery_history_ablation_benchmark.py"


def _load_script_module():
    'Load the benchmark script without constructing an AS topology.'
    specification = importlib.util.spec_from_file_location(
        "dbtfl_recovery_history_ablation_script", SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class RecoveryHistoryAblationTest(unittest.TestCase):
    'Protect deterministic state keys and formal paired reporting.'

    @classmethod
    def setUpClass(cls) -> None:
        'Load benchmark helpers once.'
        cls.script = _load_script_module()
        # Windows keeps a ctypes-loaded DLL mapped until interpreter exit.
        # Build it in the persistent test-artifact directory rather than a
        # TemporaryDirectory that Python would be unable to remove. Windows
        
        
        cls.library = cls.script._build_native_library(
            PROJECT_ROOT / "results" / "native-test-artifacts",
        )

    def test_deterministic_state_key_is_native_label_compatible(self) -> None:
        'Verify setup labels are canonical but are not presented as OPRF outputs.'
        first = self.script._canonical_label("same record")
        second = self.script._canonical_label("same record")
        different = self.script._canonical_label("different record")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 512)
        self.assertNotEqual(first, different)
        self.assertTrue(all(character in "0123456789abcdef" for character in first))

    def test_four_run_summary_requires_all_paired_repetitions(self) -> None:
        'Verify an incomplete branch cannot be silently averaged.'
        complete = self.script._trimmed([1.0, 3.0, 2.0, 4.0], expected=4)
        incomplete = self.script._trimmed([1.0, 2.0, 3.0], expected=4)

        self.assertEqual(complete["status"], "completed")
        self.assertEqual(complete["trimmed_mean"], 2.5)
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertIsNone(incomplete["trimmed_mean"])

    def test_history_policy_excludes_a_reconnected_prior_dropout(self) -> None:
        'Verify only the history variant avoids the recorded risk SID.'
        labels = {
            "client-0": tuple(self.script._canonical_label(value) for value in (
                "shared-a", "shared-b", "private-a",
            )),
            "client-1": tuple(self.script._canonical_label(value) for value in (
                "shared-a", "shared-b", "private-b",
            )),
        }
        config = self.script.BenchmarkConfig(
            clients=2,
            records_per_client=3,
            duplicate_ratio=0.30,
            history_rounds=2,
            heartbeat_timeout_seconds=6.0,
            repetitions=4,
            seed=17,
        )

        enabled = self._dispatch_history_round(config, labels, enabled=True)
        disabled = self._dispatch_history_round(config, labels, enabled=False)

        self.assertTrue(enabled["risk_sid_train_count"] == 0)
        self.assertTrue(all(
            operation == "DEDUP" for operation in enabled["victim_operations"].values()
        ))
        self.assertEqual(
            disabled["risk_sid_train_count"],
            len(disabled["victim_previous_tasks"]),
        )
        self.assertTrue(any(
            operation == "TRAIN" for operation in disabled["victim_operations"].values()
        ))

    def _dispatch_history_round(self, config, labels, *, enabled: bool):
        'Seed a real dropout and collect the first next-round operations.'
        with tempfile.TemporaryDirectory(prefix="dbtfl-history-topology-") as directory:
            topology = self.script._build_topology(
                config,
                labels,
                self.library,
                root=Path(directory),
                history_scheduling_enabled=enabled,
            )
            try:
                scenario = self.script._seed_history_dropout_round(topology, config)
                assigned, operations = self.script._dispatch_next_round(topology)
                risk_sid_train_count = sum(
                    assigned[task_id] == scenario.victim_sid
                    for task_id in scenario.risk_shared_task_ids
                )
                return {
                    "risk_sid_train_count": risk_sid_train_count,
                    "victim_operations": {
                        task_id: operations[task_id][scenario.victim_sid]
                        for task_id in scenario.risk_shared_task_ids
                    },
                    "victim_previous_tasks": scenario.victim_previous_trainer_task_ids,
                }
            finally:
                topology.close()


if __name__ == "__main__":
    unittest.main()
