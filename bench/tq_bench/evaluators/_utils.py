"""Shared utilities for benchmark evaluators."""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------

def levenshtein_distance(s: str, t: str) -> int:
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


def normalized_edit_distance(s: str, t: str) -> float:
    """Return edit distance normalized by max(len(s), len(t)).

    Returns 0.0 for two empty strings.
    """
    max_len = max(len(s), len(t))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s, t) / max_len


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

def try_parse_number(text: str) -> float | None:
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


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_basic(text: str) -> str:
    """Lowercase and strip whitespace for basic comparisons."""
    return text.strip().lower()


# Unicode subscript/superscript digit translation for robust text matching.
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


def normalize_for_match(text: str) -> str:
    """Normalize text for robust substring matching.

    NFKC Unicode normalization, subscript/superscript digit translation,
    lowercase, strip, collapse whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_UNICODE_DIGIT_MAP)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
