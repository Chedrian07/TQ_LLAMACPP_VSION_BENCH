from __future__ import annotations

import re
import unicodedata
from typing import Any

from .base import BaseEvaluator

SUPPORTED_MCQ_METRICS = ("option_match",)


# ---------------------------------------------------------------------------
# Helpers to extract a single option letter from an arbitrary model response.
# ---------------------------------------------------------------------------

_OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# Unicode subscript/superscript digit translation for robust text matching.
# Model outputs often use "HbO₂" where datasets store "HbO2".
_UNICODE_DIGIT_MAP = str.maketrans({
    # Subscripts
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3",
    "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7",
    "\u2088": "8", "\u2089": "9",
    # Superscripts
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
    "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
    "\u2078": "8", "\u2079": "9",
    # Fancy unicode spaces
    "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
})


def _normalize_for_match(text: str) -> str:
    """Normalize text for robust substring matching.

    - NFKC Unicode normalization (collapses subscript/superscript to ASCII
      where possible, merges full-width forms, ...).
    - Subscript/superscript digit translation.
    - Lowercase + strip.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_UNICODE_DIGIT_MAP)
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# Regexes compiled once at import time.
# These are intentionally case-sensitive: Qwen-VL outputs answer letters as
# uppercase, and lowercase "a"/"i" are pronouns/articles in English text.
_RE_BOXED = re.compile(r"\\boxed\{\s*([A-Z])\s*[^}]*\}")
# Match "answer is X" only when X is followed by a sentence terminator
# (period, comma, semi-colon, end-of-string, or newline) — so "answer is a
# person who ..." does NOT match.
_RE_ANSWER_IS = re.compile(
    r"(?:[Tt]he\s+)?(?:[Cc]orrect\s+)?[Aa]nswer(?:\s+choice)?(?:\s+is|:)\s*"
    r"(?:[Oo]ption\s+)?\*{0,2}\(?\s*([A-Z])\s*[\)\.\,\;\:]?(?:\s*$|\s|\n|$)",
)
_RE_OPTION_IS = re.compile(
    r"[Oo]ption\s*\(?\s*([A-Z])\s*\)?(?:\s*[\.\,\:]|\s*$)",
)
_RE_LEADING_LETTER = re.compile(
    r"^\s*\*{0,2}\(?\s*([A-Z])\s*[\)\.\,\:]",
)
# Match isolated UPPER-CASE letters only. We deliberately do NOT use
# re.IGNORECASE here because lowercase "a" / "i" appear constantly inside
# normal English sentences ("a painting", "I think") and would cause
# extract_option_letter() to return spurious answers.
#
# We further require the letter to be followed by a sentence-terminator
# (punctuation, newline, or end-of-string), so that "I" in "I think this"
# is NOT treated as an option-letter answer. Real MCQ replies always have
# the form "A.", "A)", "A,", "A:", or just "A" at the end.
_RE_ANY_LETTER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])([A-Z])(?=\s*(?:[\.\)\,\;\:]|$))",
    re.MULTILINE,
)


def _strip_markdown(text: str) -> str:
    """Remove common markdown artefacts that can wrap a bare letter."""
    # Strip surrounding ** bold **, *italic*, `code`, quotes, parentheses.
    stripped = text.strip()
    # Remove leading/trailing markdown emphasis markers.
    while stripped and stripped[0] in "*`'\"‘’“”":
        stripped = stripped[1:]
    while stripped and stripped[-1] in "*`'\"‘’“”.":
        stripped = stripped[:-1]
    return stripped.strip()


def extract_option_letter(prediction: str) -> str | None:
    """Best-effort extraction of the chosen MCQ letter from a model response.

    The benchmark runner asks the model for "only the letter of the correct
    option", but small VLMs frequently echo the full option text (e.g.
    ``"B. push particles apart"``) or wrap the letter in phrases such as
    ``"The answer is B."``.  This helper normalises these variants to a
    single uppercase letter so that
    :class:`OptionMatchEvaluator` can compare it against the ground-truth
    label, which is always a single uppercase letter (A-Z).

    Returns ``None`` if no plausible letter can be found.
    """
    if not prediction:
        return None

    text = prediction.strip()
    if not text:
        return None

    # 1. Single-character response (strict letter).
    stripped = _strip_markdown(text)
    if len(stripped) == 1 and stripped.upper() in _OPTION_LETTERS:
        return stripped.upper()

    # 2. \boxed{X} (common for math/cot models).
    m = _RE_BOXED.search(text)
    if m:
        return m.group(1).upper()

    # 3. Leading letter followed by delimiter, e.g. "B. foo", "(C) bar",
    #    "A) baz", "**D**. foo", "A:".
    m = _RE_LEADING_LETTER.match(text)
    if m:
        return m.group(1).upper()

    # 4. "The answer is X", "Correct answer: X", "answer is (B)".
    m = _RE_ANSWER_IS.search(text)
    if m:
        return m.group(1).upper()

    # 5. "option B", "Option (C)".
    m = _RE_OPTION_IS.search(text)
    if m:
        return m.group(1).upper()

    # 6. Single letter at start-of-line / end-of-line, ignoring stray
    #    surrounding whitespace or a trailing period (e.g. "B").
    head = _strip_markdown(text.splitlines()[0])
    if len(head) <= 3 and head:
        for ch in head:
            if ch.upper() in _OPTION_LETTERS:
                return ch.upper()

    # 7. Final fallback: first isolated A-Z token in the first ~80 chars.
    for m in _RE_ANY_LETTER_TOKEN.finditer(text[:80]):
        letter = m.group(1).upper()
        if letter in _OPTION_LETTERS:
            return letter

    return None


class OptionMatchEvaluator(BaseEvaluator):
    """Score MCQ / short-answer predictions.

    Accepts *reference* as either:
    - A single letter ("A"-"Z") → classic MCQ letter match via
      :func:`extract_option_letter`.
    - A list of acceptable answers → tries letter match first, then falls
      back to normalized substring / exact string match. Used for MMMU
      open questions and MathVista multi_choice (where the reference is
      both the letter and the choice text).

    Handles the common tolerance cases:
    - ``"B"``                        -> B
    - ``"B. push particles apart"``  -> B
    - ``"(C)"`` / ``"**C**"``        -> C
    - ``"The answer is D."``         -> D
    - For text refs: substring, normalized match, numeric 5% tolerance
    """

    metric_name = "option_match"

    def score(self, prediction: str, reference, *, metadata: dict[str, Any] | None = None) -> float:
        if prediction is None or reference is None:
            return 0.0

        pred_raw = str(prediction)
        pred_letter = extract_option_letter(pred_raw)
        pred_norm = _normalize_for_match(pred_raw)

        # Normalize references to a list
        if isinstance(reference, (list, tuple)):
            refs = [str(r) for r in reference]
        else:
            refs = [str(reference)]

        # Detect if any reference is a single letter A-Z (the canonical MCQ
        # form). When that's the case, we MUST match the letter; falling
        # back to numeric similarity on a different ref entry would create
        # false positives (e.g. pred="(C) [-431]" vs ref=["D","[-430]"]).
        any_letter_ref = any(
            len(r.strip()) == 1 and r.strip().upper() in _OPTION_LETTERS
            for r in refs
        )

        for ref in refs:
            ref_stripped = ref.strip()
            if not ref_stripped:
                continue
            ref_upper = ref_stripped.upper()
            ref_norm = _normalize_for_match(ref_stripped)

            # --- Strategy 1: letter-form reference (e.g. "A")
            if len(ref_upper) == 1 and ref_upper in _OPTION_LETTERS:
                if pred_letter == ref_upper:
                    return 1.0
                continue  # Letter mismatch; try next ref

            # --- Strategy 2: text-form reference
            # Exact normalized match
            if pred_norm == ref_norm:
                return 1.0
            # Containment: only score as correct if the *full* reference text
            # appears as a meaningful chunk inside the prediction. To avoid
            # spurious matches on tiny refs (e.g. "1") we require the ref
            # to be at least a few characters long.
            if ref_norm and len(ref_norm) >= 3 and ref_norm in pred_norm:
                return 1.0
            # Check last line of prediction for pure text answer
            last_line = pred_raw.strip().splitlines()[-1] if pred_raw.strip() else ""
            if (
                last_line
                and ref_norm
                and len(ref_norm) >= 3
                and ref_norm in _normalize_for_match(last_line)
            ):
                return 1.0
            # Numeric tolerance (5%) — DISABLED when a letter ref exists,
            # because the canonical answer is a letter and any numeric
            # match against the choice text is meaningless.
            if not any_letter_ref:
                try:
                    pred_num = _try_parse_num(pred_raw)
                    ref_num = _try_parse_num(ref_stripped)
                    if pred_num is not None and ref_num is not None:
                        if ref_num == 0:
                            if abs(pred_num) < 1e-6:
                                return 1.0
                        else:
                            if abs(pred_num - ref_num) / abs(ref_num) <= 0.05:
                                return 1.0
                except Exception:
                    pass

        return 0.0


def _try_parse_num(text: str) -> float | None:
    """Extract first number from text and parse as float, or None."""
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", str(text))
    if not m:
        return None
    s = m.group(0)
    if "/" in s:
        # Handle fractional like "24/7"
        try:
            num, den = s.split("/")
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None
