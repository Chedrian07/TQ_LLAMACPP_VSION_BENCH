"""MMMU official-parity evaluator.

Ports the evaluation logic from the official MMMU repository:
https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py

Key differences from the local ``OptionMatchEvaluator``:
- MCQ: uses ``rfind`` (last occurrence) instead of first-match regex
- MCQ: full index2ans content matching against choice texts
- MCQ: random fallback when no letter is extracted (official behaviour)
- Open: ``normalize_str`` rounds floats to 2 decimal places, then exact eq
- Open: substring containment (bidirectional for strings)
"""

from __future__ import annotations

import random
import re
from typing import Any

from .base import BaseEvaluator


class MMMUOfficialEvaluator(BaseEvaluator):
    """MMMU evaluation matching the official ``eval_utils.py`` logic.

    Requires ``metadata`` with:
    - ``question_type``: ``"multiple-choice"`` or ``"open"``
    - ``options``: list of choice texts (for MCQ)
    - ``is_mcq``: bool (convenience flag)

    Parameters
    ----------
    random_fallback:
        If True (default), randomly guess when no letter can be parsed.
        This matches the official scorer and inflates scores slightly
        compared to returning 0.
    seed:
        Base seed for the random fallback RNG.
    """

    metric_name = "mmmu_official"

    def __init__(self, *, random_fallback: bool = True, seed: int = 42) -> None:
        self.random_fallback = random_fallback
        self.seed = seed

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
        is_mcq = metadata.get("is_mcq", True)
        question_type = metadata.get("question_type", "multiple-choice")

        if is_mcq or question_type == "multiple-choice":
            return self._score_mcq(pred_text, reference, metadata)
        else:
            return self._score_open(pred_text, reference, metadata)

    # ------------------------------------------------------------------
    # MCQ scoring — ports parse_multi_choice_response + letter comparison
    # ------------------------------------------------------------------

    def _score_mcq(
        self, prediction: str, reference: Any, metadata: dict[str, Any]
    ) -> float:
        options: list[str] = metadata.get("options", [])

        # Resolve reference to a single gold letter
        gold_letter = self._resolve_gold_letter(reference)
        if gold_letter is None:
            return 0.0

        # Build all_choices and index2ans from options
        all_choices = [chr(ord("A") + i) for i in range(len(options))]
        index2ans = {chr(ord("A") + i): opt for i, opt in enumerate(options)}

        if not all_choices:
            # No options available, fall back to simple letter comparison
            pred_letter = self._simple_letter_extract(prediction)
            return 1.0 if pred_letter == gold_letter else 0.0

        pred_letter = self._parse_multi_choice_response(
            prediction, all_choices, index2ans, metadata
        )
        return 1.0 if pred_letter == gold_letter else 0.0

    def _parse_multi_choice_response(
        self,
        response: str,
        all_choices: list[str],
        index2ans: dict[str, str],
        metadata: dict[str, Any],
    ) -> str:
        """Port of official ``parse_multi_choice_response``.

        Returns a predicted letter (e.g. "A").
        """
        # Strip punctuation from edges (official: strip ,. ! ? ; : ')
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        # Pad with spaces for pattern matching
        response = " " + response + " "

        index_ans = True
        ans_with_brack = False
        candidates: list[str] = []

        # Priority 1: bracketed letters like (A), (B)
        for choice in all_choices:
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True

        # Priority 2: space-surrounded letters like " A "
        if len(candidates) == 0:
            for choice in all_choices:
                if f" {choice} " in response:
                    candidates.append(choice)

        # Priority 3: content matching (response > 5 tokens)
        if len(candidates) == 0 and len(response.split()) > 5:
            for index, ans in index2ans.items():
                if ans.lower() in response.lower():
                    candidates.append(index)
                    index_ans = False

        # Resolve candidates
        if len(candidates) == 0:
            if self.random_fallback:
                # Official behaviour: random guess
                sample_id = metadata.get("id", "")
                rng = random.Random(self.seed + hash(str(sample_id)))
                pred_index = rng.choice(all_choices)
            else:
                # Conservative: return a non-matching sentinel
                pred_index = ""
        elif len(candidates) > 1:
            # Multiple candidates: pick the LAST occurrence (rfind)
            start_indexes: list[int] = []
            if index_ans:
                if ans_with_brack:
                    for can in candidates:
                        idx = response.rfind(f"({can})")
                        start_indexes.append(idx)
                else:
                    for can in candidates:
                        idx = response.rfind(f" {can} ")
                        start_indexes.append(idx)
            else:
                for can in candidates:
                    idx = response.lower().rfind(index2ans[can].lower())
                    start_indexes.append(idx)
            # Pick the candidate with the largest (latest) index
            best_idx = 0
            for i in range(1, len(start_indexes)):
                if start_indexes[i] > start_indexes[best_idx]:
                    best_idx = i
            pred_index = candidates[best_idx]
        else:
            pred_index = candidates[0]

        return pred_index

    # ------------------------------------------------------------------
    # Open-ended scoring — ports eval_open + normalize_str
    # ------------------------------------------------------------------

    def _score_open(
        self, prediction: str, reference: Any, metadata: dict[str, Any]
    ) -> float:
        # Normalize gold answers
        if isinstance(reference, (list, tuple)):
            gold_items = [str(r) for r in reference]
        else:
            gold_items = [str(reference)]

        norm_answers: list[str | float] = []
        for ans in gold_items:
            norm_answers.extend(_normalize_str(ans))

        # Extract candidate predictions from model response
        pred_candidates = self._extract_open_candidates(prediction)

        # Normalize each candidate
        norm_preds: list[str | float] = []
        for cand in pred_candidates:
            norm_preds.extend(_normalize_str(cand))

        # Official eval_open logic: check if any norm_ans appears in any pred
        for pred in norm_preds:
            for norm_ans in norm_answers:
                if isinstance(pred, str) and isinstance(norm_ans, str):
                    if norm_ans in pred:
                        return 1.0
                elif isinstance(pred, (int, float)) and isinstance(norm_ans, (int, float)):
                    if pred == norm_ans:
                        return 1.0
                elif str(norm_ans) == str(pred):
                    return 1.0

        return 0.0

    def _extract_open_candidates(self, response: str) -> list[str]:
        """Extract candidate answer strings from a free-form response.

        Mirrors the MMMU pipeline's response extraction: split on sentence
        boundaries, look for indicator phrases, extract tails.
        """
        candidates: list[str] = []

        # Split on ". " followed by uppercase or newlines
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", response.strip())

        indicators = [
            "could be", "so", "is", "thus", "therefore",
            "final", "answer", "result",
        ]

        for sent in sentences:
            sent_lower = sent.lower().strip()
            for indicator in indicators:
                idx = sent_lower.rfind(indicator)
                if idx >= 0:
                    tail = sent[idx + len(indicator):].strip()
                    # Strip leading punctuation/whitespace
                    tail = tail.lstrip(":;, ")
                    if tail:
                        candidates.append(tail)

        # Also check for "=" in the last sentence
        if sentences:
            last = sentences[-1]
            eq_idx = last.rfind("=")
            if eq_idx >= 0:
                tail = last[eq_idx + 1:].strip()
                if tail:
                    candidates.append(tail)

        # Always include the full response and the last non-empty line
        candidates.append(response.strip())
        for line in reversed(response.strip().splitlines()):
            stripped = line.strip()
            if stripped:
                candidates.append(stripped)
                break

        return candidates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gold_letter(reference: Any) -> str | None:
        """Extract a single uppercase gold letter from the reference."""
        if isinstance(reference, (list, tuple)):
            for r in reference:
                s = str(r).strip().upper()
                if len(s) == 1 and s.isalpha():
                    return s
            return None
        s = str(reference).strip().upper()
        if len(s) == 1 and s.isalpha():
            return s
        return None

    @staticmethod
    def _simple_letter_extract(text: str) -> str:
        """Bare-minimum letter extraction when no options are available."""
        text = text.strip()
        if len(text) == 1 and text.upper().isalpha():
            return text.upper()
        m = re.search(r"\b([A-Z])\b", text)
        return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Official normalize_str + check_is_number (standalone functions)
# ---------------------------------------------------------------------------

def _check_is_number(string: str) -> bool:
    """Port of official ``check_is_number``."""
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(string: str) -> list[str | float]:
    """Port of official ``normalize_str``.

    Returns a list because the official code returns:
    - ``[float_val]`` for numbers (rounded to 2 decimals)
    - ``[" x", "x "]`` for single-char strings (space-padded)
    - ``[lower_str]`` for everything else
    """
    string = string.strip()

    if _check_is_number(string):
        cleaned = string.replace(",", "")
        val = round(float(cleaned), 2)
        return [val]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]
        return [string]
