"""Regression tests ensuring existing evaluators still work after metadata kwarg addition."""

import pytest
from tq_bench.evaluators import get_evaluator


class TestOptionMatchEvaluator:
    def setup_method(self):
        self.ev = get_evaluator("option_match")

    def test_basic_letter(self):
        assert self.ev.score("A", "A") == 1.0

    def test_wrong_letter(self):
        assert self.ev.score("A", "B") == 0.0

    def test_verbose_response(self):
        assert self.ev.score("B. push particles apart", "B") == 1.0

    def test_boxed(self):
        assert self.ev.score("\\boxed{C}", "C") == 1.0

    def test_answer_is(self):
        assert self.ev.score("The answer is D.", "D") == 1.0

    def test_metadata_none(self):
        assert self.ev.score("A", "A", metadata=None) == 1.0

    def test_metadata_ignored(self):
        assert self.ev.score("A", "A", metadata={"foo": "bar"}) == 1.0

    def test_text_reference_list(self):
        assert self.ev.score("145°", ["C", "145°"]) == 1.0


class TestANLSEvaluator:
    def setup_method(self):
        self.ev = get_evaluator("anls")

    def test_exact(self):
        assert self.ev.score("hello", "hello") == 1.0

    def test_empty(self):
        assert self.ev.score("", "") == 1.0

    def test_threshold(self):
        # Very different strings → nld >= 0.5 → 0.0
        assert self.ev.score("abc", "xyz") == 0.0

    def test_metadata_ignored(self):
        assert self.ev.score("hello", "hello", metadata={"x": 1}) == 1.0


class TestRelaxedAccuracy:
    def setup_method(self):
        self.ev = get_evaluator("relaxed_accuracy")

    def test_exact_string(self):
        assert self.ev.score("42", "42") == 1.0

    def test_numeric_5pct(self):
        assert self.ev.score("42", "40") == 1.0

    def test_numeric_exceed(self):
        assert self.ev.score("50", "40") == 0.0

    def test_metadata_ignored(self):
        assert self.ev.score("42", "42", metadata={}) == 1.0


class TestMathVistaMatch:
    def setup_method(self):
        self.ev = get_evaluator("mathvista_match")

    def test_letter(self):
        assert self.ev.score("A", ["A", "foo"]) == 1.0

    def test_numeric(self):
        assert self.ev.score("42", "42") == 1.0


class TestRegistry:
    def test_all_metrics_resolve(self):
        for name in [
            "option_match", "anls", "relaxed_accuracy", "exact_match",
            "normalized_exact_match", "mathvista_match",
            "mmmu_official", "mathvista_official", "textvqa_official",
            "chartqapro_official",
        ]:
            ev = get_evaluator(name)
            assert ev is not None, f"Failed to resolve {name}"
            assert hasattr(ev, "score")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_evaluator("nonexistent_metric")
