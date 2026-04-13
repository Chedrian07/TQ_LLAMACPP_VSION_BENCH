"""VLM benchmark dataset loaders.

Each class loads a HuggingFace dataset, samples deterministically, and
yields normalised dicts with ``image``, ``question``, ``answer``, and
(for MCQ) ``options`` keys.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset
from PIL import Image

from .base import BaseBenchmarkDataset

SUPPORTED_VLM_BENCHMARKS = (
    "ai2d",
    "chartqa",
    "chartqapro",
    "docvqa",
    "mathvista",
    "mmmu",
    "ocrbench_v2",
    "textvqa",
)


# ======================================================================
# 1. AI2D
# ======================================================================


class AI2DDataset(BaseBenchmarkDataset):
    """AI2D — multiple-choice diagram understanding.

    HuggingFace: ``lmms-lab/ai2d``
    Split: ``test``
    Metric: ``option_match``
    """

    benchmark_id = "ai2d"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        ds = load_dataset("lmms-lab/ai2d", split="test")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            image: Image.Image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            options: list[str] = row["options"]
            answer_idx: int = int(row["answer"])
            answer_label = self._index_to_label(answer_idx)
            question_text = row["question"]

            options_block = self._format_mcq_options(options)
            prompt = (
                f"{question_text}\n\n"
                f"{options_block}\n\n"
                "Answer with the letter of the correct option."
            )
            self._samples.append(
                {
                    "id": f"ai2d_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": answer_label,
                    "options": options,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 2. ChartQA
# ======================================================================


class ChartQADataset(BaseBenchmarkDataset):
    """ChartQA — chart understanding free-form QA.

    HuggingFace: ``HuggingFaceM4/ChartQA``
    Split: ``test``
    Metric: ``relaxed_accuracy``
    """

    benchmark_id = "chartqa"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            image: Image.Image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            question_text: str = row["query"]
            # label may be a list or a single string depending on the row;
            # keep ALL labels so the relaxed_accuracy evaluator can match any.
            raw_label = row["label"]
            if isinstance(raw_label, list):
                answers: list[str] = [str(l) for l in raw_label if l is not None]
                if not answers:
                    answers = [""]
            else:
                answers = [str(raw_label)]

            prompt = (
                f"{question_text}\n\n"
                "Answer concisely with just the numeric value or short phrase."
            )
            self._samples.append(
                {
                    "id": f"chartqa_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": answers if len(answers) > 1 else answers[0],
                    "human_or_machine": row.get("human_or_machine"),
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 3. ChartQA-Pro
# ======================================================================


class ChartQAProDataset(BaseBenchmarkDataset):
    """ChartQA-Pro — advanced chart understanding QA.

    HuggingFace: ``ahmed-masry/ChartQAPro``
    Split: ``test``
    Metric: ``relaxed_accuracy``

    Notes: Question/Answer are stored as lists (one QA pair per chart).
    Image is stored as raw JPEG bytes and must be decoded.
    """

    benchmark_id = "chartqapro"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        import io

        ds = load_dataset("ahmed-masry/ChartQAPro", split="test")

        # Each row may contain multiple Q/A pairs.  Flatten to individual
        # samples so each sample has exactly one question.
        flat: list[dict[str, Any]] = []
        for row in ds:
            questions: list[str] = row["Question"]
            answers: list[str] = row["Answer"]
            img_data = row["image"]  # raw bytes
            q_type = row.get("Type", "")
            year_col = row.get("Year", [])
            for i, (q, a) in enumerate(zip(questions, answers)):
                yr_flag = year_col[i] if isinstance(year_col, list) and i < len(year_col) else None
                flat.append({
                    "question": q, "answer": a, "image_bytes": img_data,
                    "question_type": q_type,
                    "year_flags": [yr_flag] if yr_flag is not None else None,
                })

        # Deterministic sampling from the flattened list
        import random as _random
        rng = _random.Random(seed)
        if n_samples < len(flat):
            flat = rng.sample(flat, n_samples)
        else:
            rng.shuffle(flat)

        self._samples = []
        for idx, item in enumerate(flat):
            img_bytes = item["image_bytes"]
            if isinstance(img_bytes, bytes):
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                image = img_bytes
                if hasattr(image, "mode") and image.mode != "RGB":
                    image = image.convert("RGB")

            prompt = (
                f"{item['question']}\n\n"
                "Answer concisely."
            )
            self._samples.append(
                {
                    "id": f"chartqapro_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": item["answer"],
                    "question_type": item.get("question_type", ""),
                    "year_flags": item.get("year_flags"),
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 4. DocVQA
# ======================================================================


class DocVQADataset(BaseBenchmarkDataset):
    """DocVQA — document visual question answering.

    HuggingFace: ``lmms-lab/DocVQA``
    Split: ``test``  (validation also available)
    Metric: ``anls``
    """

    benchmark_id = "docvqa"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        # lmms-lab/DocVQA test split has no answers; use validation.
        ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            image: Image.Image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            question_text: str = row["question"]
            # Answers are stored as a list of acceptable strings
            answers = row.get("answers", row.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]

            prompt = (
                f"{question_text}\n\n"
                "Answer briefly."
            )
            self._samples.append(
                {
                    "id": f"docvqa_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": answers,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 5. MathVista
# ======================================================================


class MathVistaDataset(BaseBenchmarkDataset):
    """MathVista — mathematical reasoning with visual context.

    HuggingFace: ``AI4Math/MathVista``
    Split: ``testmini``
    Metric: ``mathvista_match``
    """

    benchmark_id = "mathvista"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            image: Image.Image | None = row.get("decoded_image") or row.get("image")
            if image is not None and image.mode != "RGB":
                image = image.convert("RGB")

            question_type = row.get("question_type", "")  # "multi_choice" | "free_form"
            answer_type = row.get("answer_type", "")      # "text" | "integer" | "float" | ...
            answer: str = str(row["answer"])
            choices = row.get("choices") or []
            precision = row.get("precision")

            # Use the official MathVista `query` field if present — it is
            # the pre-formatted prompt the MathVista authors use for eval.
            # Format (for multi_choice):
            #   Hint: Please answer the question and provide the correct
            #         option letter, e.g., A, B, C, D, at the end.
            #   Question: <q>
            #   Choices:
            #   (A) <c0>
            #   (B) <c1>
            # For free_form:
            #   Hint: Please answer the question requiring a floating-point
            #         number with one decimal place ...
            #   Question: <q>
            official_query = row.get("query")

            if official_query and isinstance(official_query, str) and official_query.strip():
                prompt = official_query.strip()
            elif question_type == "multi_choice" and choices:
                options_block = self._format_mcq_options(list(choices))
                prompt = (
                    f"Hint: Please answer the question and provide the correct option "
                    f"letter, e.g., A, B, C, D, at the end.\n"
                    f"Question: {row['question']}\n"
                    f"Choices:\n{options_block}"
                )
            else:
                prec_hint = ""
                if answer_type in ("integer",):
                    prec_hint = " requiring an integer answer"
                elif answer_type in ("float",) and precision:
                    prec_hint = f" requiring a floating-point number with {int(precision)} decimal place(s)"
                prompt = (
                    f"Hint: Please answer the question{prec_hint}.\n"
                    f"Question: {row['question']}"
                )

            # Store the resolved answer: for multi_choice we want the choice text
            # (since model output after letter extraction is compared to it),
            # but ALSO keep the letter-form for option-letter matching.
            resolved_answer: str | list[str] = answer
            if question_type == "multi_choice" and choices:
                try:
                    correct_idx = list(choices).index(answer)
                    letter = chr(ord('A') + correct_idx)
                    # Provide BOTH letter and text as acceptable references.
                    resolved_answer = [letter, str(answer)]
                except ValueError:
                    resolved_answer = answer

            self._samples.append(
                {
                    "id": f"mathvista_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": resolved_answer,
                    "choices": list(choices) if choices else None,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "precision": precision,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 6. MMMU
# ======================================================================


class MMMUDataset(BaseBenchmarkDataset):
    """MMMU — Massive Multi-discipline Multimodal Understanding.

    HuggingFace: ``MMMU/MMMU``
    Split: ``validation``  (test answers not public)
    Metric: ``option_match``
    """

    benchmark_id = "mmmu"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        ds = load_dataset("MMMU/MMMU", "Accounting", split="validation")
        # MMMU is split by subject.  Load all available validation subjects
        # and merge into one dataset, then sample.
        _MMMU_SUBJECTS = [
            "Accounting", "Agriculture", "Architecture_and_Engineering",
            "Art", "Art_Theory", "Basic_Medical_Science",
            "Biology", "Chemistry", "Clinical_Medicine",
            "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
            "Economics", "Electronics", "Energy_and_Power",
            "Finance", "Geography", "History",
            "Literature", "Manage", "Marketing",
            "Materials", "Math", "Mechanical_Engineering",
            "Music", "Pharmacy", "Physics",
            "Psychology", "Public_Health", "Sociology",
        ]
        parts = []
        for subject in _MMMU_SUBJECTS:
            try:
                part = load_dataset("MMMU/MMMU", subject, split="validation")
                parts.append(part)
            except Exception:
                continue

        from datasets import concatenate_datasets
        if parts:
            ds = concatenate_datasets(parts)
        # else: fall back to the Accounting-only ds we already loaded

        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        import ast
        for idx, row in enumerate(ds):
            # MMMU stores images as image_1 .. image_7 columns.
            images: list[Image.Image] = []
            for img_key in ("image_1", "image_2", "image_3", "image_4",
                            "image_5", "image_6", "image_7"):
                candidate = row.get(img_key)
                if candidate is not None:
                    if candidate.mode != "RGB":
                        candidate = candidate.convert("RGB")
                    images.append(candidate)
            image: Image.Image | None = images[0] if images else None

            question_text: str = row["question"]
            raw_answer = row.get("answer", "")
            question_type = row.get("question_type", "multiple-choice")

            # Parse options — stored as a stringified Python list, e.g. "['$6', '$7']"
            options: list[str] = []
            raw_options = row.get("options", "")
            if raw_options and isinstance(raw_options, str) and raw_options.strip().startswith("["):
                try:
                    parsed = ast.literal_eval(raw_options)
                    if isinstance(parsed, list):
                        options = [str(o) for o in parsed]
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(raw_options, list):
                options = [str(o) for o in raw_options]

            # Keep explicit image numbering for multi-image MMMU questions.
            clean_question = re.sub(
                r"<image\s*(\d+)\s*>",
                lambda m: f"[image {m.group(1)}]",
                question_text,
            )

            # Route by question type (MMMU has "multiple-choice" and "open")
            is_mcq = question_type == "multiple-choice" and bool(options)

            if is_mcq:
                options_block = self._format_mcq_options(options)
                prompt = (
                    f"{clean_question}\n"
                    f"{options_block}\n"
                    "Answer the preceding multiple choice question. "
                    "The last line of your response should be of the following "
                    "format: 'Answer: $LETTER' (without quotes) where LETTER "
                    "is one of the options. Think step by step before "
                    "answering."
                )
                # answer is a single letter like "C"; also provide text form
                answer_str = str(raw_answer).strip()
                resolved_answer: str | list[str] = answer_str
                # If it's a letter, also add the corresponding choice text
                if len(answer_str) == 1 and answer_str.isalpha():
                    try:
                        correct_idx = ord(answer_str.upper()) - ord('A')
                        if 0 <= correct_idx < len(options):
                            resolved_answer = [answer_str, options[correct_idx]]
                    except Exception:
                        pass
            else:
                # Open-ended question: answer is either a plain string
                # or a stringified list of acceptable answers like "['24/7', '3.429']"
                prompt = (
                    f"{clean_question}\n"
                    "Answer the preceding open-ended question. "
                    "The last line of your response should be of the following "
                    "format: 'Answer: $FINAL_ANSWER' (without quotes) where "
                    "FINAL_ANSWER is a short answer. Think step by step "
                    "before answering."
                )
                # Parse answer
                answer_list: list[str] = []
                if isinstance(raw_answer, str) and raw_answer.strip().startswith("["):
                    try:
                        parsed = ast.literal_eval(raw_answer)
                        if isinstance(parsed, list):
                            answer_list = [str(a) for a in parsed]
                    except (ValueError, SyntaxError):
                        pass
                if not answer_list:
                    answer_list = [str(raw_answer)]
                resolved_answer = answer_list

            self._samples.append(
                {
                    "id": f"mmmu_{idx}",
                    "image": image,
                    "images": images if images else None,
                    "question": prompt,
                    "answer": resolved_answer,
                    "options": options if options else None,
                    "question_type": question_type,
                    "is_mcq": is_mcq,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 7. OCRBench-v2
# ======================================================================


class OCRBenchV2Dataset(BaseBenchmarkDataset):
    """OCRBench-v2 — OCR-centric VLM benchmark.

    HuggingFace: ``echo840/OCRBench_v2``
    Split: varies; typically the full dataset has a single split.
    Metric: ``exact_match``
    """

    benchmark_id = "ocrbench_v2"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        # OCRBench v2 may be hosted under several names; try common ones.
        ds = None
        for repo_id in ("echo840/OCRBench_v2", "echo840/OCRBench-v2", "lmms-lab/OCRBench_v2"):
            try:
                ds = load_dataset(repo_id, split="test")
                break
            except Exception:
                try:
                    ds = load_dataset(repo_id, split="train")
                    break
                except Exception:
                    continue

        if ds is None:
            raise RuntimeError(
                "Could not load OCRBench-v2 from HuggingFace. "
                "Tried: echo840/OCRBench_v2, echo840/OCRBench-v2, lmms-lab/OCRBench_v2"
            )

        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            image: Image.Image | None = row.get("image")
            if image is not None and image.mode != "RGB":
                image = image.convert("RGB")

            question_text: str = row.get("question", row.get("query", ""))
            answer = row.get("answer", row.get("label", ""))
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            else:
                answer = str(answer)

            prompt = f"{question_text}\n\nAnswer concisely."

            self._samples.append(
                {
                    "id": f"ocrbench_v2_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": answer,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples


# ======================================================================
# 8. TextVQA
# ======================================================================


class TextVQADataset(BaseBenchmarkDataset):
    """TextVQA — reading text in images.

    HuggingFace: ``lmms-lab/textvqa``
    Split: ``validation``
    Metric: ``normalized_exact_match``
    """

    benchmark_id = "textvqa"

    def load(self, n_samples: int = 500, seed: int = 42) -> None:
        ds = load_dataset("lmms-lab/textvqa", split="validation")
        ds = self._deterministic_sample(ds, n_samples, seed)
        self._samples = []
        for idx, row in enumerate(ds):
            image: Image.Image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            question_text: str = row["question"]
            # TextVQA has a list of 10 human answers; store all for evaluation
            answers: list[str] = row.get("answers", [])
            if not answers:
                answers = [str(row.get("answer", ""))]

            prompt = (
                f"{question_text}\n\n"
                "Answer with a short phrase."
            )
            self._samples.append(
                {
                    "id": f"textvqa_{idx}",
                    "image": image,
                    "question": prompt,
                    "answer": answers,
                }
            )
        self._loaded = True

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        yield from self._samples
