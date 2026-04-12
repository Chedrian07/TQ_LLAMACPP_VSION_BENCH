from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

from .base import BaseEvaluator

SUPPORTED_VQA_METRICS = (
    "anls",
    "exact_match",
    "mathvista_match",
    "normalized_exact_match",
    "relaxed_accuracy",
)


# ======================================================================
# String utilities shared across evaluators
# ======================================================================

def _levenshtein_distance(s: str, t: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n

    # Use two-row optimisation to keep memory O(min(n, m)).
    if n > m:
        s, t = t, s
        n, m = m, n

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for j in range(1, m + 1):
        curr[0] = j
        for i in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[i] = min(
                prev[i] + 1,      # deletion
                curr[i - 1] + 1,  # insertion
                prev[i - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _try_parse_number(text: str) -> float | None:
    """Try to parse *text* as a number, returning ``None`` on failure.

    Handles commas as thousands separators, percentage signs, and leading
    dollar signs.
    """
    cleaned = text.strip()
    # Remove currency symbols and percent signs.
    cleaned = cleaned.lstrip("$").rstrip("%")
    # Remove thousands separators.
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_text_vqa(text: str) -> str:
    """Normalize text following the TextVQA / VQAv2 convention.

    Steps: lowercase -> strip articles -> strip punctuation -> collapse
    whitespace.
    """
    text = text.lower()
    # Remove articles.
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation.
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace.
    text = " ".join(text.split())
    return text.strip()


def _normalize_basic(text: str) -> str:
    """Lowercase and strip whitespace for basic comparisons."""
    return text.strip().lower()


# ======================================================================
# MathVista answer extraction helpers
# ======================================================================

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?answer\s+is\s*[:\-]?\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer)\s*[:\-]?\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(r"\b([A-E])\b")


def _extract_mathvista_answer(text: str) -> str:
    r"""Extract the final answer from a MathVista-style model response.

    Tries (in order):
    1. ``\\boxed{...}``
    2. "The answer is ..."  /  "Final answer: ..."
    3. Last standalone option letter (A-E)
    4. Last line of the response (stripped)
    """
    # 1. Boxed answer
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    # 2. "The answer is ..." / "Final answer ..."
    for pattern in (_ANSWER_IS_RE, _FINAL_ANSWER_RE):
        m = pattern.search(text)
        if m:
            return m.group(1).strip()

    # 3. Option letter (last one wins)
    options = _OPTION_RE.findall(text)
    if options:
        return options[-1]

    # 4. Fallback: last non-empty line.
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped

    return text.strip()


# ======================================================================
# Concrete evaluators
# ======================================================================

class ANLSEvaluator(BaseEvaluator):
    """Average Normalized Levenshtein Similarity (DocVQA metric)."""

    metric_name = "anls"

    def score(self, prediction: str, reference: str | list[str], *, metadata: dict[str, Any] | None = None) -> float:
        """Return ANLS score.

        *reference* may be a single string or a list of acceptable ground
        truths (max is taken).
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        pred_norm = _normalize_basic(prediction)
        best = 0.0
        for ref in references:
            ref_norm = _normalize_basic(ref)
            max_len = max(len(pred_norm), len(ref_norm))
            if max_len == 0:
                # Both empty -> perfect match.
                best = 1.0
                break
            nld = _levenshtein_distance(pred_norm, ref_norm) / max_len
            score = 1.0 - nld if nld < 0.5 else 0.0
            best = max(best, score)
        return best


class RelaxedAccuracyEvaluator(BaseEvaluator):
    """Relaxed accuracy (ChartQA style).

    Correct if exact string match OR numeric value within 5% tolerance.
    """

    metric_name = "relaxed_accuracy"

    def score(self, prediction: str, reference: str | list[str], *, metadata: dict[str, Any] | None = None) -> float:
        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        pred_norm = _normalize_basic(prediction)
        pred_num = _try_parse_number(pred_norm)

        for ref in references:
            ref_norm = _normalize_basic(ref)

            # Exact string match.
            if pred_norm == ref_norm:
                return 1.0

            # Numeric tolerance (5%).
            ref_num = _try_parse_number(ref_norm)
            if pred_num is not None and ref_num is not None:
                if ref_num == 0:
                    if pred_num == 0:
                        return 1.0
                else:
                    if abs(pred_num - ref_num) / abs(ref_num) <= 0.05:
                        return 1.0

        return 0.0


class ExactMatchEvaluator(BaseEvaluator):
    """Case-insensitive exact match after basic normalization."""

    metric_name = "exact_match"

    def score(self, prediction: str, reference: str | list[str], *, metadata: dict[str, Any] | None = None) -> float:
        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        pred_norm = _normalize_basic(prediction)
        for ref in references:
            if pred_norm == _normalize_basic(ref):
                return 1.0
        return 0.0


class NormalizedExactMatchEvaluator(BaseEvaluator):
    """TextVQA / VQAv2 consensus accuracy.

    Official formula (Goyal et al. 2017):
        acc(ans) = min(#humans_who_said_same / 3, 1)
    Using all 10 human answers. Each reference is normalized the same way
    (lowercase, strip articles/punctuation, collapse whitespace) before
    comparison.

    For a single-reference case, this degrades to a simple 0/1 match.
    """

    metric_name = "normalized_exact_match"

    def score(self, prediction: str, reference: str | list[str], *, metadata: dict[str, Any] | None = None) -> float:
        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        if not references:
            return 0.0

        pred_norm = _normalize_text_vqa(prediction)
        if not pred_norm:
            return 0.0

        # Also extract a concise short-phrase variant (last line, strip prefix)
        short = _extract_short_answer(prediction)
        short_norm = _normalize_text_vqa(short) if short else ""

        ref_norms = [_normalize_text_vqa(r) for r in references]

        # Count number of refs matching prediction (either full or short)
        matches = 0
        for rn in ref_norms:
            if not rn:
                continue
            if rn == pred_norm or (short_norm and rn == short_norm):
                matches += 1

        if matches == 0:
            # Try containment fallback for single-reference case
            if len(ref_norms) == 1 and ref_norms[0] and ref_norms[0] in pred_norm:
                return 1.0
            return 0.0

        # VQA consensus: min(matches / 3, 1)
        if len(references) >= 3:
            return min(matches / 3.0, 1.0)
        # Single/few reference: binary
        return 1.0


def _extract_short_answer(text: str) -> str:
    """Pull a short answer phrase out of a verbose response.

    Strategies: last line, then strip common prefixes like "answer:".
    """
    if not text:
        return ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    # Strip common prefixes
    prefixes = [
        "answer:", "answer is", "the answer is", "final answer:",
        "response:", "result:",
    ]
    lower = candidate.lower()
    for p in prefixes:
        if lower.startswith(p):
            candidate = candidate[len(p):].strip()
            break
    # Strip trailing period
    return candidate.rstrip(".,!?")


class MathVistaMatchEvaluator(BaseEvaluator):
    """MathVista answer matching.

    Extracts the final answer from the model response and compares it to
    the ground truth. Handles both multi_choice (letter or choice text) and
    free_form (numeric with precision/tolerance or text).

    For multi_choice: the *reference* is typically a list containing both
    the letter form ("A") and the choice text ("145°"); either match counts.
    For free_form: numeric answers compared with precision-aware tolerance,
    text answers compared after normalization.
    """

    metric_name = "mathvista_match"

    def score(self, prediction: str, reference: str | list[str], *, metadata: dict[str, Any] | None = None) -> float:
        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        # Pull candidate extractions from the prediction
        extracted = _extract_mathvista_answer(prediction)
        candidates: list[str] = [extracted]

        # Also try the full (cleaned) prediction — sometimes the model
        # emits the correct choice text inline without a preamble.
        candidates.append(prediction.strip())

        # Also try the last line
        last_line = _last_nonempty_line(prediction)
        if last_line and last_line not in candidates:
            candidates.append(last_line)

        # Also try any numeric value found in the response
        any_num = _first_number(prediction)
        if any_num is not None:
            candidates.append(str(any_num))

        # Normalized candidates
        cand_norms = [_normalize_basic(c) for c in candidates]
        cand_nums = [_try_parse_number(c) for c in cand_norms]

        for ref in references:
            ref_norm = _normalize_basic(ref)
            ref_num = _try_parse_number(ref_norm)

            # --- Strategy 1: direct normalized string match against any candidate
            for cn in cand_norms:
                if cn == ref_norm and cn:
                    return 1.0
                # Letter references (e.g. "A") — check first non-whitespace letter of extraction
                if len(ref_norm) == 1 and ref_norm.isalpha():
                    # Look for isolated letter in candidate
                    if re.search(rf"\b{re.escape(ref_norm)}\b", cn):
                        return 1.0

            # --- Strategy 2: substring containment (reference appears in prediction)
            # Useful for choice-text like "145°" appearing in "...so the answer is 145°."
            for cn in cand_norms:
                if ref_norm and ref_norm in cn:
                    return 1.0
                # Also try without unit suffixes for numeric-like refs
                if ref_num is not None:
                    ref_num_str = _normalize_basic(str(ref_num))
                    if ref_num_str and ref_num_str in cn:
                        return 1.0

            # --- Strategy 3: numeric tolerance (5% or within 1 unit-of-precision)
            if ref_num is not None:
                for cnum in cand_nums:
                    if cnum is None:
                        continue
                    if ref_num == 0:
                        if abs(cnum) < 1e-6:
                            return 1.0
                    else:
                        rel_err = abs(cnum - ref_num) / abs(ref_num)
                        if rel_err <= 0.05:
                            return 1.0

        return 0.0


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _first_number(text: str) -> float | None:
    m = _NUMBER_RE.search(text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


# ======================================================================
# VqaEvaluator facade (preserves original API)
# ======================================================================

_METRIC_EVALUATORS: dict[str, type[BaseEvaluator]] = {
    "anls": ANLSEvaluator,
    "relaxed_accuracy": RelaxedAccuracyEvaluator,
    "exact_match": ExactMatchEvaluator,
    "normalized_exact_match": NormalizedExactMatchEvaluator,
    "mathvista_match": MathVistaMatchEvaluator,
}


class VqaEvaluator(BaseEvaluator):
    """Dispatch wrapper that delegates to the appropriate concrete evaluator."""

    def __init__(self, metric_name: str) -> None:
        if metric_name not in _METRIC_EVALUATORS:
            raise ValueError(
                f"Unknown VQA metric {metric_name!r}. "
                f"Supported: {', '.join(SUPPORTED_VQA_METRICS)}"
            )
        self.metric_name = metric_name
        self._delegate = _METRIC_EVALUATORS[metric_name]()

    def score(self, prediction: str, reference: str | list[str], *, metadata: dict[str, Any] | None = None) -> float:
        return self._delegate.score(prediction, reference, metadata=metadata)
