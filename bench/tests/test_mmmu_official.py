"""Golden parity tests for MMMU official evaluator."""

import pytest
from tests.conftest import load_fixture
from tq_bench.evaluators.mmmu_official import MMMUOfficialEvaluator

CASES = load_fixture("mmmu_golden.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_mmmu_official(case):
    ev = MMMUOfficialEvaluator(seed=42)
    score = ev.score(case["prediction"], case["reference"], metadata=case["metadata"])
    assert score == pytest.approx(case["expected"], abs=1e-6), (
        f'{case["name"]}: got {score}, expected {case["expected"]}'
    )
