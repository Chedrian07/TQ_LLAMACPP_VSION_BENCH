"""Evaluator abstractions for benchmark metrics."""

from .base import BaseEvaluator
from .mcq import SUPPORTED_MCQ_METRICS, OptionMatchEvaluator
from .vqa import (
    SUPPORTED_VQA_METRICS,
    ANLSEvaluator,
    ExactMatchEvaluator,
    MathVistaMatchEvaluator,
    NormalizedExactMatchEvaluator,
    RelaxedAccuracyEvaluator,
    VqaEvaluator,
)
from .mmmu_official import (
    MMMUEvalOnlyEvaluator,
    MMMUOfficialEvaluator,
    MMMUOfficialRawEvaluator,
)
from .mathvista_official import MathVistaOfficialEvaluator
from .textvqa_official import TextVQAOfficialEvaluator
from .chartqapro_official import ChartQAProOfficialEvaluator

__all__ = [
    "BaseEvaluator",
    "SUPPORTED_MCQ_METRICS",
    "SUPPORTED_VQA_METRICS",
    "get_evaluator",
]

_REGISTRY: dict[str, type[BaseEvaluator]] = {
    # MCQ
    "option_match": OptionMatchEvaluator,
    # VQA
    "anls": ANLSEvaluator,
    "relaxed_accuracy": RelaxedAccuracyEvaluator,
    "exact_match": ExactMatchEvaluator,
    "normalized_exact_match": NormalizedExactMatchEvaluator,
    "mathvista_match": MathVistaMatchEvaluator,
    # Official parity evaluators
    "mmmu_official": MMMUOfficialEvaluator,
    "mmmu_eval_only": MMMUEvalOnlyEvaluator,
    "mmmu_official_raw": MMMUOfficialRawEvaluator,
    "mathvista_official": MathVistaOfficialEvaluator,
    "textvqa_official": TextVQAOfficialEvaluator,
    "chartqapro_official": ChartQAProOfficialEvaluator,
}


def get_evaluator(metric_name: str) -> BaseEvaluator:
    """Return an evaluator instance for the given metric name.

    Args:
        metric_name: One of the supported metric names (see
            ``SUPPORTED_MCQ_METRICS`` and ``SUPPORTED_VQA_METRICS``).

    Returns:
        A concrete :class:`BaseEvaluator` subclass instance.

    Raises:
        ValueError: If *metric_name* is not recognised.
    """
    cls = _REGISTRY.get(metric_name)
    if cls is None:
        raise ValueError(
            f"Unknown metric {metric_name!r}. "
            f"Supported: {sorted(_REGISTRY.keys())}"
        )
    return cls()
