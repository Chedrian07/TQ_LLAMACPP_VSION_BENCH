"""Golden parity tests for ChartQAPro official evaluator."""

import pytest
from tests.conftest import load_fixture
from tq_bench.evaluators.chartqapro_official import ChartQAProOfficialEvaluator

CASES = load_fixture("chartqapro_golden.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_chartqapro_official(case):
    ev = ChartQAProOfficialEvaluator()
    score = ev.score(case["prediction"], case["reference"], metadata=case["metadata"])

    if "expected_min" in case:
        # Range check for ANLS-type scores
        assert case["expected_min"] <= score <= case["expected_max"], (
            f'{case["name"]}: got {score}, expected [{case["expected_min"]}, {case["expected_max"]}]'
        )
    else:
        assert score == pytest.approx(case["expected"], abs=1e-6), (
            f'{case["name"]}: got {score}, expected {case["expected"]}'
        )
