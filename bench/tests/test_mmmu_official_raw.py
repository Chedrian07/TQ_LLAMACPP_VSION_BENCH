"""Golden tests for the MMMU raw official parse-and-eval scorer."""

from __future__ import annotations

import pytest

from tq_bench.evaluators.mmmu_official import MMMUOfficialEvaluator, MMMUOfficialRawEvaluator


@pytest.mark.parametrize(
    ("prediction", "reference", "metadata", "expected"),
    [
        (
            "After checking every option, the answer is C among the choices.",
            ["C", "$8"],
            {"is_mcq": True, "question_type": "multiple-choice", "options": ["$6", "$7", "$8", "$9"], "id": "raw1"},
            1.0,
        ),
        (
            "Based on the calculation the total cost would be $8 so that is the answer to this question",
            ["C", "$8"],
            {"is_mcq": True, "question_type": "multiple-choice", "options": ["$6", "$7", "$8", "$9"], "id": "raw2"},
            1.0,
        ),
        (
            "The answer is 3.14159",
            "3.14",
            {"is_mcq": False, "question_type": "open", "options": [], "id": "raw3"},
            1.0,
        ),
        (
            "The result is definitely Paris based on the evidence",
            "paris",
            {"is_mcq": False, "question_type": "open", "options": [], "id": "raw4"},
            0.0,
        ),
        (
            "Tokyo",
            "Paris",
            {"is_mcq": False, "question_type": "open", "options": [], "id": "raw5"},
            0.0,
        ),
    ],
)
def test_mmmu_official_raw(prediction, reference, metadata, expected):
    ev = MMMUOfficialRawEvaluator(seed=42)
    score = ev.score(prediction, reference, metadata=metadata)
    assert score == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("prediction", "reference", "metadata", "expected"),
    [
        (
            "After checking every option, the answer is C among the choices.",
            ["C", "$8"],
            {"is_mcq": True, "question_type": "multiple-choice", "options": ["$6", "$7", "$8", "$9"], "id": "alias1"},
            1.0,
        ),
        (
            "The answer is 3.14159",
            "3.14",
            {"is_mcq": False, "question_type": "open", "options": [], "id": "alias2"},
            1.0,
        ),
    ],
)
def test_mmmu_official_alias_uses_raw_scoring(prediction, reference, metadata, expected):
    ev = MMMUOfficialEvaluator(seed=42)
    score = ev.score(prediction, reference, metadata=metadata)
    assert score == pytest.approx(expected, abs=1e-6)


def test_mmmu_official_raw_does_not_accept_short_dotted_letter_form():
    ev = MMMUOfficialRawEvaluator(seed=42)
    score = ev.score(
        "D. Planned obsolescence",
        ["D", "Planned obsolescence"],
        metadata={
            "is_mcq": True,
            "question_type": "multiple-choice",
            "options": [
                "The conflict perspective",
                "Conspicuous consumption",
                "Media",
                "Planned obsolescence",
            ],
            "id": "raw6",
        },
    )
    assert score == pytest.approx(0.0, abs=1e-6)
