'Regression tests for fixed OPRF allocation without expensive OPRF calls.'

from __future__ import annotations

import unittest
from pathlib import Path

from dbtfl.evaluation.oprf_precompute import _allocate, allocate_joining_records
from dbtfl.evaluation.runner import EvaluationPlan, _selected_cases, _validate_precomputed_oprf_cases
from dbtfl.training import PreparedRecord


class OprfPrecomputeAllocationTest(unittest.TestCase):
    'Verify fixed base data and fresh joining-client data contracts.'

    def setUp(self) -> None:
        'Build an in-memory corpus without altering prepared user data.'
        self.records = tuple(
            PreparedRecord(str(index), f"record-{index}", {})
            for index in range(20_000)
        )

    def test_fixed_and_joining_allocations_have_expected_overlap(self) -> None:
        'Ensure 10x1024 base and one 307-record joining overlap.'
        base = _allocate(self.records)
        joining = allocate_joining_records(
            self.records, base, joining_clients=1, duplicate_ratio=0.30, seed=17,
        )
        self.assertEqual({len(values) for values in base.values()}, {1024})
        self.assertEqual(len(joining["client-10"]), 1024)
        base_texts = {text for values in base.values() for text in values}
        self.assertEqual(len(base_texts.intersection(joining["client-10"])), 307)

    def test_precomputation_rejects_variable_scale_suites_before_gpu_work(self) -> None:
        'Reject incompatible sweeps before CUDA or OPRF execution starts.'
        plan = EvaluationPlan(
            output_directory=Path("temporary-output"),
            included_suites=("fault_dedup",),
            precomputed_oprf_directory=Path("precompute"),
        )
        _validate_precomputed_oprf_cases(plan, _selected_cases(plan))
        invalid = EvaluationPlan(
            output_directory=Path("temporary-output"),
            included_suites=("data_scale",),
            precomputed_oprf_directory=Path("precompute"),
        )
        with self.assertRaises(ValueError):
            _validate_precomputed_oprf_cases(invalid, _selected_cases(invalid))


if __name__ == "__main__":
    unittest.main()
