"""Golden parity tests for MathVista official evaluator."""

import pytest
from tests.conftest import load_fixture
from tq_bench.evaluators.mathvista_official import MathVistaOfficialEvaluator

CASES = load_fixture("mathvista_golden.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_mathvista_official(case):
    ev = MathVistaOfficialEvaluator()
    score = ev.score(case["prediction"], case["reference"], metadata=case["metadata"])
    assert score == pytest.approx(case["expected"], abs=1e-6), (
        f'{case["name"]}: got {score}, expected {case["expected"]}'
    )
