"""MMMU official-parity evaluators.

This module keeps two distinct MMMU evaluation paths:

``mmmu_official_raw``
    Mirrors the official ``main_parse_and_eval.py`` flow: free-form model
    responses are parsed with benchmark-specific MMMU logic, then scored.

``mmmu_eval_only``
    Mirrors the official ``main_eval_only.py`` flow: a single final answer is
    extracted first, then the official eval-only scorer is applied.

The benchmark parity path should use the raw parse-and-eval flow.  The
``mmmu_official`` metric name is therefore an alias of
``mmmu_official_raw``.  ``mmmu_eval_only`` remains available for
controlled comparisons and debugging.
"""

from __future__ import annotations

import re
import zlib
from typing import Any

from .base import BaseEvaluator
from .mcq import extract_option_letter


_INDICATORS_OF_KEYS = [
    "could be ",
    "so ",
    "is ",
    "thus ",
    "therefore ",
    "final ",
    "answer ",
    "result ",
]

_OPTION_ONLY_RE = re.compile(
    r"^(?:answer\s*[:：-]?\s*)?(?:\*{0,2})?\(?\s*[A-Za-z]\s*\)?(?:[.,:;])?$",
    re.IGNORECASE,
)
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:final\s+answer|answer|result)\s*[:：-]?\s*",
    re.IGNORECASE,
)
_CAPITALIZED_SPAN_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9'/-]*(?:\s+[A-Z][A-Za-z0-9'/-]*)*)\b"
)


# ---------------------------------------------------------------------------
# Exact official eval_utils.py ports
# ---------------------------------------------------------------------------

def _official_parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    *,
    fallback_index: str = "",
) -> str:
    """Port of official ``parse_multi_choice_response``.

    The official code uses a mutable global RNG for the random fallback.  That
    is not reproducible under this repo's parallel per-sample scoring, so the
    caller must inject a deterministic ``fallback_index`` instead.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return fallback_index

    if len(candidates) > 1:
        start_indexes: list[int] = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))

        best_idx = 0
        for i in range(1, len(start_indexes)):
            if start_indexes[i] > start_indexes[best_idx]:
                best_idx = i
        return candidates[best_idx]

    return candidates[0]


def _check_is_number(string: str) -> bool:
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(string: str) -> list[str | float]:
    string = string.strip()

    if _check_is_number(string):
        string = string.replace(",", "")
        string = float(string)
        string = round(string, 2)
        return [string]

    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]
    return [string]


def _extract_numbers(string: str) -> list[str]:
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)

    return numbers_with_commas + numbers_scientific + numbers_simple


def _official_parse_open_response(response: str) -> list[str | float]:
    """Port of official ``parse_open_response``."""

    def get_key_subresponses(text: str) -> list[str]:
        text = text.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", text)
        indicators = list(_INDICATORS_OF_KEYS)
        key_responses: list[str] = []

        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators.extend(["="])

            shortest_key_response: str | None = None
            for indicator in indicators:
                if indicator in resp:
                    candidate = resp.split(indicator)[-1].strip()
                    if not shortest_key_response or len(candidate) < len(shortest_key_response):
                        shortest_key_response = candidate

            if shortest_key_response and shortest_key_response.strip() not in [
                ":",
                ",",
                ".",
                "!",
                "?",
                ";",
                ":",
                "'",
            ]:
                key_responses.append(shortest_key_response)

        if len(key_responses) == 0:
            return [text]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(_extract_numbers(resp))

    tmp_pred_list: list[str | float] = []
    for pred in pred_list:
        tmp_pred_list.extend(_normalize_str(str(pred)))

    deduped: list[str | float] = []
    seen: set[str] = set()
    for pred in tmp_pred_list:
        key = repr(pred)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pred)

    return deduped


def _eval_multi_choice(gold_i: Any, pred_i: str) -> bool:
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                return True
        return False
    return gold_i == pred_i


def _eval_open(gold_i: Any, pred_i: list[str | float]) -> bool:
    if isinstance(gold_i, list):
        norm_answers: list[str | float] = []
        for answer in gold_i:
            norm_answers.extend(_normalize_str(answer))
    else:
        norm_answers = _normalize_str(str(gold_i))

    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        else:
            if pred in norm_answers:
                return True
    return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_gold_letter(reference: Any) -> str | None:
    if isinstance(reference, (list, tuple)):
        for item in reference:
            token = str(item).strip().upper()
            if len(token) == 1 and token.isalpha():
                return token
        return None
    token = str(reference).strip().upper()
    if len(token) == 1 and token.isalpha():
        return token
    return None


def _build_choice_maps(options: list[str] | None) -> tuple[list[str], dict[str, str]]:
    choices = [chr(ord("A") + i) for i in range(len(options or []))]
    index2ans = {
        chr(ord("A") + i): str(opt)
        for i, opt in enumerate(options or [])
    }
    return choices, index2ans


def _fallback_index_for_sample(all_choices: list[str], sample_id: str, seed: int) -> str:
    if not all_choices:
        return ""
    checksum = zlib.crc32(sample_id.encode("utf-8")) & 0xFFFFFFFF
    return all_choices[(seed + checksum) % len(all_choices)]


def _strip_answer_prefix(text: str) -> str:
    return _ANSWER_PREFIX_RE.sub("", text.strip()).strip()


def _looks_like_option_only(text: str) -> bool:
    return bool(_OPTION_ONLY_RE.fullmatch(text.strip()))


def _latest_choice_text_match(text: str, options: list[str] | None) -> str:
    if not text or not options:
        return ""

    lowered = text.lower()
    best_index = -1
    best_letter = ""
    for idx, option in enumerate(options):
        position = lowered.rfind(str(option).lower())
        if position > best_index:
            best_index = position
            best_letter = chr(ord("A") + idx)
    return best_letter if best_index >= 0 else ""


def _extract_eval_only_mcq_answer(response: str, options: list[str] | None) -> str:
    text = str(response).strip()
    if not text:
        return ""

    letter = extract_option_letter(text)
    if letter:
        return letter

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for candidate in (reversed(lines[-3:]) if lines else []):
        candidate = _strip_answer_prefix(candidate)
        if not candidate:
            continue
        letter = extract_option_letter(candidate)
        if letter:
            return letter
        mapped = _latest_choice_text_match(candidate, options)
        if mapped:
            return mapped

    mapped = _latest_choice_text_match(text, options)
    if mapped:
        return mapped

    all_choices, index2ans = _build_choice_maps(options)
    return _official_parse_multi_choice_response(
        text,
        all_choices,
        index2ans,
        fallback_index="",
    )


def _extract_shortest_key_tail(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    candidates: list[str] = []
    lowered = stripped.lower()
    for indicator in _INDICATORS_OF_KEYS:
        if indicator in lowered:
            tail = stripped[lowered.rfind(indicator) + len(indicator):].strip()
            tail = tail.lstrip(":;, ").strip()
            if tail:
                candidates.append(tail)

    if "=" in stripped:
        tail = stripped[stripped.rfind("=") + 1:].strip()
        if tail:
            candidates.append(tail)

    if not candidates:
        return ""

    return min(candidates, key=len)


def _extract_last_capitalized_span(text: str) -> str:
    matches = _CAPITALIZED_SPAN_RE.findall(text)
    for match in reversed(matches):
        token = match.strip()
        if token.lower() in {"the", "answer", "result", "final"}:
            continue
        return token
    return ""


def _extract_eval_only_open_answer(response: str) -> str:
    text = str(response).strip()
    if not text:
        return ""

    boxed = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
    if boxed:
        return boxed[-1].group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        cleaned = _strip_answer_prefix(line)
        if not cleaned:
            continue
        if _looks_like_option_only(cleaned):
            continue

        numbers = _extract_numbers(cleaned)
        if numbers:
            tail = _extract_shortest_key_tail(cleaned)
            if tail and not _looks_like_option_only(tail):
                tail_numbers = _extract_numbers(tail)
                if "/" in tail:
                    return tail
                if tail_numbers:
                    return tail_numbers[-1]
            if "/" in cleaned:
                return cleaned
            return numbers[-1]

        capitalized = _extract_last_capitalized_span(line)
        if capitalized:
            return capitalized

        tail = _extract_shortest_key_tail(cleaned)
        if tail and not _looks_like_option_only(tail):
            return tail
        return cleaned

    tail = _extract_shortest_key_tail(text)
    if tail and not _looks_like_option_only(tail):
        numbers = _extract_numbers(tail)
        if numbers:
            return numbers[-1]
        return tail

    numbers = _extract_numbers(text)
    if numbers:
        return numbers[-1]

    return _strip_answer_prefix(text)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class MMMUOfficialRawEvaluator(BaseEvaluator):
    """Official MMMU parse-and-eval scorer.

    This follows the official parser and scorer as closely as possible inside a
    per-sample evaluator.  The one intentional deviation is the random fallback:
    the official script mutates a single global RNG, which is not reproducible
    under this repo's parallel evaluation.  Here the fallback is stabilized per
    sample via ``seed`` + ``sample_id``.
    """

    metric_name = "mmmu_official_raw"

    def __init__(self, *, seed: int = 42) -> None:
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
        question_type = metadata.get("question_type", "multiple-choice")
        is_mcq = metadata.get("is_mcq", question_type == "multiple-choice")

        if is_mcq or question_type == "multiple-choice":
            options = metadata.get("options") or []
            all_choices, index2ans = _build_choice_maps(options)

            if not all_choices:
                gold_letter = _resolve_gold_letter(reference)
                pred_letter = extract_option_letter(pred_text) or ""
                return 1.0 if gold_letter and pred_letter == gold_letter else 0.0

            sample_id = str(metadata.get("id", ""))
            pred_letter = _official_parse_multi_choice_response(
                pred_text,
                all_choices,
                index2ans,
                fallback_index=_fallback_index_for_sample(all_choices, sample_id, self.seed),
            )
            return 1.0 if _eval_multi_choice(reference, pred_letter) else 0.0

        parsed_pred = _official_parse_open_response(pred_text)
        return 1.0 if _eval_open(reference, parsed_pred) else 0.0


class MMMUEvalOnlyEvaluator(BaseEvaluator):
    """MMMU eval-only scorer with local final-answer extraction.

    The official ``main_eval_only.py`` path expects each prediction file entry
    to already contain a single final answer.  This evaluator performs that
    extraction from the model's free-form response, then runs the official
    eval-only scoring logic.
    """

    metric_name = "mmmu_eval_only"

    def extract_prediction(
        self,
        prediction: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        text = "" if prediction is None else str(prediction).strip()
        if not text:
            return ""

        metadata = metadata or {}
        question_type = metadata.get("question_type", "multiple-choice")
        is_mcq = metadata.get("is_mcq", question_type == "multiple-choice")

        if is_mcq or question_type == "multiple-choice":
            return _extract_eval_only_mcq_answer(text, metadata.get("options"))

        return _extract_eval_only_open_answer(text)

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
        final_prediction = self.extract_prediction(prediction, metadata=metadata)
        if not final_prediction:
            return 0.0

        question_type = metadata.get("question_type", "multiple-choice")
        is_mcq = metadata.get("is_mcq", question_type == "multiple-choice")

        if is_mcq or question_type == "multiple-choice":
            return 1.0 if _eval_multi_choice(reference, final_prediction) else 0.0

        parsed_pred = _official_parse_open_response(final_prediction)
        return 1.0 if _eval_open(reference, parsed_pred) else 0.0


class MMMUOfficialEvaluator(MMMUOfficialRawEvaluator):
    """Backward-compatible alias for the raw official MMMU scorer."""

    metric_name = "mmmu_official"
