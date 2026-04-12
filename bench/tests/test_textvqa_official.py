"""Golden parity tests for TextVQA official evaluator."""

import pytest
from tests.conftest import load_fixture
from tq_bench.evaluators.textvqa_official import TextVQAOfficialEvaluator

CASES = load_fixture("textvqa_golden.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_textvqa_official(case):
    ev = TextVQAOfficialEvaluator()
    score = ev.score(case["prediction"], case["reference"])
    assert score == pytest.approx(case["expected"], abs=1e-6), (
        f'{case["name"]}: got {score}, expected {case["expected"]}'
    )
