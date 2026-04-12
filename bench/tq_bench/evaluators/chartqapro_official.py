"""ChartQAPro official-parity evaluator.

Ports the evaluation logic from the official ChartQAPro repository:
https://github.com/vis-nlp/ChartQAPro/blob/main/evaluate_predictions.py

Key differences from the local ``RelaxedAccuracyEvaluator``:
- Question-type routing: Fact Checking / Multi Choice → exact match
- Year flags: year answers use exact match instead of numeric tolerance
- List-level pairwise evaluation: index-aligned, per-element scoring
- ANLS fallback: when numeric comparison fails, uses ANLS with tau=0.5
"""

from __future__ import annotations

import ast
import re
from typing import Any

from .base import BaseEvaluator
from ._utils import levenshtein_distance


class ChartQAProOfficialEvaluator(BaseEvaluator):
    """ChartQAPro evaluation matching the official ``evaluate_predictions.py``.

    Requires ``metadata`` with:
    - ``question_type``: e.g. ``"Fact Checking"``, ``"Multi Choice"``,
      ``"Conversational"``, ``"Open-ended"``, etc.
    - ``year_flags``: list of ``"YES"``/``"NO"`` strings, or None

    Parameters
    ----------
    max_relative_change:
        Numeric tolerance (default 5%).
    """

    metric_name = "chartqapro_official"

    def __init__(self, max_relative_change: float = 0.05) -> None:
        self.max_relative_change = max_relative_change

    def score(
        self,
        prediction: Any,
        reference: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        if prediction is None or reference is None:
            return 0.0

        metadata = metadata or {}
        question_type = metadata.get("question_type", "")
        year_flags = metadata.get("year_flags")

        always_exact = question_type in ("Fact Checking", "Multi Choice")

        target_str = str(reference)
        pred_str = str(prediction).strip()

        return self._relaxed_correctness(
            target=target_str,
            prediction=pred_str,
            max_relative_change=self.max_relative_change,
            year_flags=year_flags,
            always_use_exact_match=always_exact,
            question_type=question_type,
        )

    # ------------------------------------------------------------------
    # Core evaluation — port of relaxed_correctness_chartqapro
    # ------------------------------------------------------------------

    def _relaxed_correctness(
        self,
        target: str,
        prediction: str,
        max_relative_change: float,
        year_flags: list[str] | None,
        always_use_exact_match: bool,
        question_type: str,
    ) -> float:
        target_list = _parse_to_list(target)
        pred_list = _parse_to_list(prediction)

        # For Conversational: only use last year flag
        if question_type == "Conversational" and year_flags and len(year_flags) > 0:
            year_flags = [year_flags[-1]]

        scores: list[float] = []
        for i in range(max(len(target_list), len(pred_list))):
            if i >= len(target_list) or i >= len(pred_list):
                scores.append(0.0)
                continue

            t = target_list[i]
            p = pred_list[i]

            # Year flag or always-exact routing
            flag = year_flags[i] if year_flags and i < len(year_flags) else None
            use_exact = always_use_exact_match or (
                flag is not None and str(flag).upper() == "YES"
            )

            if use_exact:
                scores.append(1.0 if t.strip().lower() == p.strip().lower() else 0.0)
            else:
                scores.append(
                    _evaluate_single_answer(t, p, max_relative_change)
                )

        return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Port of evaluate_single_answer
# ---------------------------------------------------------------------------

def _evaluate_single_answer(
    target: str, prediction: str, max_relative_change: float = 0.05
) -> float:
    """Numeric with 5% tolerance, falling back to ANLS."""
    t = target.strip().strip("%").strip()
    p = prediction.strip().strip("%").strip()

    t_f = _to_float(t)
    p_f = _to_float(p)

    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0

    # ANLS fallback for non-numeric answers
    return _anls_score(p.lower(), t.lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(s: str) -> float | None:
    """Try to parse string as float, removing thousands separators."""
    cleaned = s.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_to_list(text: str) -> list[str]:
    """Parse a string that may be a Python list literal, or return as single element."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            pass
    return [text]


def _anls_score(prediction: str, gold: str, threshold: float = 0.5) -> float:
    """ANLS (Average Normalized Levenshtein Similarity) with threshold."""
    if not prediction and not gold:
        return 1.0
    max_len = max(len(prediction), len(gold))
    if max_len == 0:
        return 1.0
    nld = levenshtein_distance(prediction, gold) / max_len
    return 1.0 - nld if nld < threshold else 0.0
