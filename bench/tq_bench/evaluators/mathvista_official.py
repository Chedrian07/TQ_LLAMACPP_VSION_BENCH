"""MathVista official-parity evaluator.

Ports the evaluation logic from the official MathVista repository:
- calculate_score.py: normalize_extracted_answer, safe_equal, get_most_similar
- extract_answer.py: answer extraction (regex-only, no GPT)

Key differences from the local ``MathVistaMatchEvaluator``:
- Multi-choice: letter is mapped to choice TEXT, then TEXT==TEXT comparison
- Fuzzy matching: Levenshtein distance selects closest choice (no 5% tolerance)
- Numeric: precision-aware rounding + strict equality (not 5% tolerance)
- Integer: ``str(int(float(x)))`` coercion
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseEvaluator
from ._utils import levenshtein_distance


class MathVistaOfficialEvaluator(BaseEvaluator):
    """MathVista evaluation matching the official ``calculate_score.py``.

    Requires ``metadata`` with:
    - ``question_type``: ``"multi_choice"`` or ``"free_form"``
    - ``answer_type``: ``"text"`` | ``"integer"`` | ``"float"`` | ``"list"``
    - ``precision``: int or None (decimal places for float comparison)
    - ``choices``: list of choice texts (for multi_choice)
    """

    metric_name = "mathvista_official"

    def score(
        self,
        prediction: Any,
        reference: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        if prediction is None or reference is None:
            return 0.0

        pred_text = str(prediction).strip()
        if not pred_text:
            return 0.0

        metadata = metadata or {}
        question_type = metadata.get("question_type", "free_form")
        answer_type = metadata.get("answer_type", "text")
        precision = metadata.get("precision")
        choices = metadata.get("choices") or []

        # Resolve reference to the gold answer string
        gold = self._resolve_gold(reference)
        if gold is None:
            return 0.0

        # Step 1: Extract answer from model response (regex-only, no GPT)
        extraction = _extract_answer(pred_text, question_type, answer_type, choices)

        # Step 2: Normalize extracted answer (official logic)
        normalized = _normalize_extracted_answer(
            extraction, choices, question_type, answer_type, precision
        )
        if normalized is None:
            return 0.0

        # Step 3: Compare (official safe_equal — strict equality)
        return 1.0 if _safe_equal(normalized, gold) else 0.0

    @staticmethod
    def _resolve_gold(reference: Any) -> str | None:
        """Get the gold answer string.

        For multi_choice, the reference may be ``[letter, text]``.
        The official scorer compares against the choice TEXT, not the letter.
        """
        if isinstance(reference, (list, tuple)):
            # Prefer the non-letter entry (choice text)
            for r in reference:
                s = str(r).strip()
                if len(s) > 1:
                    return s
            # Fall back to the letter
            for r in reference:
                s = str(r).strip()
                if s:
                    return s
            return None
        s = str(reference).strip()
        return s if s else None


# ---------------------------------------------------------------------------
# Answer extraction (regex-only, replaces GPT-based extraction)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?answer\s+is\s*[:\-]?\s*(.+?)(?:(?<!\d)\.|$)", re.IGNORECASE
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer)\s*[:\-]?\s*(.+?)(?:(?<!\d)\.|$)", re.IGNORECASE
)
_PAREN_LETTER_RE = re.compile(r"\(([a-zA-Z])\)")
_OPTION_RE = re.compile(r"\b([A-E])\b")


def _extract_answer(
    response: str,
    question_type: str,
    answer_type: str,
    choices: list[str],
) -> str:
    """Extract the answer from model output (regex-based, no GPT).

    Follows the extraction priority from extract_answer.py:
    1. For multi_choice: check if response is literally one of the choices
    2. For numeric: try direct parse
    3. \\boxed{...}
    4. "The answer is ..." / "Final answer: ..."
    5. Last standalone option letter (for multi_choice)
    6. Last non-empty line
    """
    response = response.strip()

    # For multi_choice: check if response is one of the choices verbatim
    if question_type == "multi_choice" and choices:
        for c in choices:
            if response.strip() == c.strip():
                return response.strip()

    # For numeric types: try direct parse
    if answer_type == "integer":
        try:
            return str(int(float(response)))
        except (ValueError, OverflowError):
            pass
    elif answer_type == "float":
        try:
            float(response)
            return response
        except ValueError:
            pass

    # Quick extract: \boxed{...}
    m = _BOXED_RE.search(response)
    if m:
        return _clean_extraction(m.group(1))

    # "The answer is ..." / "Final answer ..."
    for pattern in (_ANSWER_IS_RE, _FINAL_ANSWER_RE):
        m = pattern.search(response)
        if m:
            return _clean_extraction(m.group(1))

    # (A) pattern for multi_choice
    if question_type == "multi_choice":
        letters = _PAREN_LETTER_RE.findall(response)
        if letters:
            return letters[-1].upper()
        options = _OPTION_RE.findall(response)
        if options:
            return options[-1]

    # Fallback: last non-empty line
    for line in reversed(response.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return _clean_extraction(stripped)

    return _clean_extraction(response.strip())


def _clean_extraction(text: str) -> str:
    """Strip trailing punctuation and whitespace from an extracted answer."""
    return text.strip().rstrip(".,;:!? ")


# ---------------------------------------------------------------------------
# Official normalize_extracted_answer
# ---------------------------------------------------------------------------

def _get_most_similar(prediction: str, choices: list[str]) -> str:
    """Port of official ``get_most_similar`` — Levenshtein closest match."""
    if not choices:
        return prediction
    distances = [levenshtein_distance(prediction, c) for c in choices]
    min_idx = 0
    for i in range(1, len(distances)):
        if distances[i] < distances[min_idx]:
            min_idx = i
    return choices[min_idx]


def _normalize_extracted_answer(
    extraction: str,
    choices: list[str],
    question_type: str,
    answer_type: str,
    precision: int | float | None,
) -> str | None:
    """Port of official ``normalize_extracted_answer``."""
    if question_type == "multi_choice":
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        if not extraction:
            return None

        # Extract "A" from "(A) text"
        letters = re.findall(r"\(([a-zA-Z])\)", extraction)
        if letters:
            extraction = letters[0].upper()

        sequential = [chr(ord("A") + i) for i in range(len(choices))]

        if extraction in sequential:
            option_index = sequential.index(extraction)
            return choices[option_index]
        else:
            return _get_most_similar(extraction, choices)

    elif answer_type == "integer":
        try:
            return str(int(float(extraction)))
        except Exception:
            return None

    elif answer_type == "float":
        try:
            prec = int(precision) if precision is not None else 2
            return str(round(float(extraction), prec))
        except Exception:
            return None

    elif answer_type == "list":
        try:
            return str(extraction)
        except Exception:
            return None

    else:
        # Default: text comparison
        return str(extraction).strip() if extraction else None


# ---------------------------------------------------------------------------
# Official safe_equal
# ---------------------------------------------------------------------------

def _safe_equal(prediction: str | None, answer: str | None) -> bool:
    """Port of official ``safe_equal`` — strict equality."""
    try:
        return prediction == answer
    except Exception:
        return False
