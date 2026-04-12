"""TextVQA official-parity evaluator.

Ports the evaluation logic from the official MMF repository:
https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py

Key differences from the local ``NormalizedExactMatchEvaluator``:
- Normalization: full EvalAIAnswerProcessor (contractions, number words,
  digit-adjacent punctuation rules) instead of simple article/punct removal
- Scoring: leave-one-out VQA consensus (official formula)
- No short-answer extraction fallback
- No containment fallback
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseEvaluator


# ---------------------------------------------------------------------------
# Port of EvalAIAnswerProcessor
# ---------------------------------------------------------------------------

# fmt: off
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
    "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2",
    "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
# fmt: on

_ARTICLES = {"a", "an", "the"}

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
_PUNCTUATIONS = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")",
    "=", "+", "\\", "_", "-", ">", "<", "@", "`",
    ",", "?", "!",
]


def _word_tokenize(word: str) -> str:
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()


def _process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in _PUNCTUATIONS:
        if (p + " " in in_text or " " + p in in_text) or (
            re.search(_COMMA_STRIP, in_text) is not None
        ):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def _process_digit_article(in_text: str) -> str:
    out_text: list[str] = []
    temp_text = in_text.lower().split()
    for word in temp_text:
        word = _NUMBER_MAP.get(word, word)
        if word not in _ARTICLES:
            out_text.append(word)
    for word_id, word in enumerate(out_text):
        if word in _CONTRACTIONS:
            out_text[word_id] = _CONTRACTIONS[word]
    return " ".join(out_text)


def evalai_answer_processor(item: str) -> str:
    """Full port of ``EvalAIAnswerProcessor.__call__``."""
    item = _word_tokenize(item)
    item = item.replace("\n", " ").replace("\t", " ").strip()
    item = _process_punctuation(item)
    item = _process_digit_article(item)
    return item


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TextVQAOfficialEvaluator(BaseEvaluator):
    """TextVQA evaluation matching the official MMF evaluator.

    Uses the full ``EvalAIAnswerProcessor`` normalization and the
    leave-one-out VQA consensus scoring formula.

    Expects *reference* to be a list of 10 human answer strings.
    Falls back gracefully for fewer references.
    """

    metric_name = "textvqa_official"

    def score(
        self,
        prediction: Any,
        reference: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        if prediction is None or reference is None:
            return 0.0

        pred_norm = evalai_answer_processor(str(prediction))
        if not pred_norm:
            return 0.0

        if isinstance(reference, str):
            references = [reference]
        else:
            references = list(reference)

        if not references:
            return 0.0

        # Normalize all human answers
        gt_answers = [evalai_answer_processor(str(r)) for r in references]

        # Compute leave-one-out consensus score (official formula)
        return self._compute_vqa_accuracy(pred_norm, gt_answers)

    @staticmethod
    def _compute_vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
        """Official VQA consensus accuracy.

        For each of the N ground-truth answers, compute:
            acc_i = min(#others_matching_pred / 3, 1.0)
        Return mean(acc_i).

        When N < 3, falls back to simple binary match.
        """
        n = len(gt_answers)
        if n == 0:
            return 0.0

        if n < 3:
            # Too few answers for the consensus formula; binary match
            return 1.0 if pred in gt_answers else 0.0

        # Leave-one-out: for each GT slot, count how many OTHER GTs match pred
        accs: list[float] = []
        for i in range(n):
            others = [gt_answers[j] for j in range(n) if j != i]
            matching = sum(1 for o in others if o == pred)
            accs.append(min(matching / 3.0, 1.0))
        return sum(accs) / len(accs)
