"""Golden parity tests for MMMU eval-only scorer."""

import pytest
from tests.conftest import load_fixture
from tq_bench.evaluators.mmmu_official import MMMUEvalOnlyEvaluator

CASES = load_fixture("mmmu_golden.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_mmmu_eval_only(case):
    ev = MMMUEvalOnlyEvaluator()
    score = ev.score(case["prediction"], case["reference"], metadata=case["metadata"])
    assert score == pytest.approx(case["expected"], abs=1e-6), (
        f'{case["name"]}: got {score}, expected {case["expected"]}'
    )
def test_mmmu_eval_only_prefers_numeric_answer_over_trailing_letter():
    ev = MMMUEvalOnlyEvaluator()
    score = ev.score(
        "Therefore, the equilibrium price is $8.\n\nAnswer: D",
        ["8"],
        metadata={"is_mcq": False, "question_type": "open", "options": [], "id": "t13"},
    )
    assert score == pytest.approx(1.0, abs=1e-6)
