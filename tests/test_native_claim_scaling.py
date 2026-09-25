'Regression tests for native CAS/mutex scaling measurement.'

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_native_claim_scaling_benchmark.py"


def _load_script_module():
    'Load the standalone script without compiling native code.'
    specification = importlib.util.spec_from_file_location(
        "dbtfl_native_claim_scaling_script", SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class _FakeNativeApi:
    'Deterministic C ABI stand-in for measurement bookkeeping tests.'

    def __init__(self, script: object) -> None:
        'Keep the module constants used by this test double.'
        self.script = script
        self.calls: list[int] = []

    def run(self, mode: int, scenario: int, workers: int, operations: int) -> dict[str, float | int]:
        'Return valid exact outcomes for either native mode.'
        self.calls.append(mode)
        attempts = workers * operations
        wins = attempts if scenario == self.script.DISJOINT_SCENARIO else operations
        elapsed = 1.0 if mode == self.script.CAS_MODE else 2.0
        return {
            "elapsed_seconds": elapsed,
            "attempts": attempts,
            "wins": wins,
            "attempts_per_second": attempts / elapsed,
        }


class NativeClaimScalingTest(unittest.TestCase):
    'Protect paired ordering and exact one-winner validation.'

    @classmethod
    def setUpClass(cls) -> None:
        'Load benchmark helpers once.'
        cls.script = _load_script_module()

    def test_disjoint_case_alternates_order_and_preserves_all_wins(self) -> None:
        'Verify a disjoint task run keeps every successful state transition.'
        api = _FakeNativeApi(self.script)
        result = self.script._paired_measurement(
            api,
            scenario=self.script.DISJOINT_SCENARIO,
            workers=4,
            operations_per_worker=10,
            repetition=1,
            category="disjoint_scaling",
        )

        self.assertEqual(api.calls, [self.script.MUTEX_MODE, self.script.CAS_MODE])
        self.assertEqual(result["attempts"], 40)
        self.assertEqual(result["expected_wins"], 40)
        self.assertEqual(result["cas"]["wins"], 40)
        self.assertEqual(result["mutex"]["wins"], 40)

    def test_shared_hotset_accepts_exactly_one_winner_per_task(self) -> None:
        'Verify shared tasks remain a correctness-only one-winner control.'
        api = _FakeNativeApi(self.script)
        result = self.script._paired_measurement(
            api,
            scenario=self.script.SHARED_HOTSET_SCENARIO,
            workers=10,
            operations_per_worker=32,
            repetition=0,
            category="shared_hotset_correctness",
        )

        self.assertEqual(result["attempts"], 320)
        self.assertEqual(result["expected_wins"], 32)
        self.assertEqual(result["cas"]["wins"], 32)
        self.assertEqual(result["mutex"]["wins"], 32)


if __name__ == "__main__":
    unittest.main()
