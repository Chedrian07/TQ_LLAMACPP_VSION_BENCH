"""Text benchmark dataset loaders.

Each class loads a HuggingFace dataset, samples deterministically, and
yields normalised dicts with ``question``, ``answer``, and ``options``
keys.  All three text benchmarks are multiple-choice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset, concatenate_datasets

from .base import BaseBenchmarkDataset

SUPPORTED_TEXT_BENCHMARKS = (
    "mmlu",
    "commonsenseqa",
    "hellaswag",
)


# ======================================================================
# 9. MMLU
# ======================================================================


class MMLUDataset(BaseBenchmarkDataset):
    """MMLU — Massive Multitask Language Understanding.

    HuggingFace: ``cais/mmlu`` (all subjects)
    Split: ``test``
    Metric: ``option_match``
    """

    benchmark_id = "mmlu"

    def load(self, n_samples: int = 1000, seed: int = 42) -> None:
        # cais/mmlu stores all subjects under the "all" config
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            question_text: str = row["question"]
            choices: list[str] = row["choices"]
            answer_idx: int = row["answer"]
            answer_label = self._index_to_label(answer_idx)

            options_block = self._format_mcq_options(choices)
            prompt = (
                f"{question_text}\n\n"
                f"{options_block}\n\n"
                "Answer with the letter of the correct option."
            )
            self._samples.append(
                {
                    "id": f"mmlu_{idx}",
                    "question": prompt,
                    "answer": answer_label,
                    "options": choices,
                    "subject": row.get("subject", ""),
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 10. CommonsenseQA
# ======================================================================


class CommonsenseQADataset(BaseBenchmarkDataset):
    """CommonsenseQA — commonsense reasoning MCQ.

    HuggingFace: ``tau/commonsense_qa``
    Split: ``validation``  (test labels not public)
    Metric: ``option_match``
    """

    benchmark_id = "commonsenseqa"

    def load(self, n_samples: int = 3000, seed: int = 42) -> None:
        ds = load_dataset("tau/commonsense_qa", split="validation")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            question_text: str = row["question"]
            # tau/commonsense_qa stores choices as a dict with "label" and "text"
            choice_labels: list[str] = row["choices"]["label"]
            choice_texts: list[str] = row["choices"]["text"]
            answer_key: str = row["answerKey"]  # "A", "B", ...

            options_block = self._format_mcq_options(choice_texts)
            prompt = (
                f"{question_text}\n\n"
                f"{options_block}\n\n"
                "Answer with the letter of the correct option."
            )
            self._samples.append(
                {
                    "id": f"commonsenseqa_{idx}",
                    "question": prompt,
                    "answer": answer_key.upper(),
                    "options": choice_texts,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 11. HellaSwag
# ======================================================================


class HellaSwagDataset(BaseBenchmarkDataset):
    """HellaSwag — grounded commonsense inference.

    HuggingFace: ``Rowan/hellaswag``
    Split: ``validation``  (test labels not public)
    Metric: ``option_match``
    """

    benchmark_id = "hellaswag"

    def load(self, n_samples: int = 3000, seed: int = 42) -> None:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            # HellaSwag: context/activity_label + endings
            ctx_a: str = row.get("ctx_a", "")
            ctx_b: str = row.get("ctx_b", "")
            context = f"{ctx_a} {ctx_b}".strip() if ctx_b else ctx_a
            activity: str = row.get("activity_label", "")
            endings: list[str] = row["endings"]
            answer_idx: int = int(row["label"])
            answer_label = self._index_to_label(answer_idx)

            options_block = self._format_mcq_options(endings)
            prompt = (
                f"{activity}: {context}\n\n"
                "Which ending best completes the text?\n\n"
                f"{options_block}\n\n"
                "Answer with the letter of the correct option."
            )
            self._samples.append(
                {
                    "id": f"hellaswag_{idx}",
                    "question": prompt,
                    "answer": answer_label,
                    "options": endings,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples
