"""Dataset abstractions and registry for all benchmark loaders."""

from __future__ import annotations

from .base import BaseBenchmarkDataset
from .text import (
    SUPPORTED_TEXT_BENCHMARKS,
    CommonsenseQADataset,
    HellaSwagDataset,
    MMLUDataset,
)
from .vlm import (
    SUPPORTED_VLM_BENCHMARKS,
    AI2DDataset,
    ChartQADataset,
    ChartQAProDataset,
    DocVQADataset,
    MathVistaDataset,
    MMMUDataset,
    OCRBenchV2Dataset,
    TextVQADataset,
)

# ----- Registry: benchmark_id -> dataset class ---------------------------

_REGISTRY: dict[str, type[BaseBenchmarkDataset]] = {
    "ai2d": AI2DDataset,
    "chartqa": ChartQADataset,
    "chartqapro": ChartQAProDataset,
    "docvqa": DocVQADataset,
    "mathvista": MathVistaDataset,
    "mmmu": MMMUDataset,
    "ocrbench_v2": OCRBenchV2Dataset,
    "textvqa": TextVQADataset,
    "mmlu": MMLUDataset,
    "commonsenseqa": CommonsenseQADataset,
    "hellaswag": HellaSwagDataset,
}


def get_dataset(benchmark_id: str) -> BaseBenchmarkDataset:
    """Instantiate the dataset loader for the given benchmark ID.

    Parameters
    ----------
    benchmark_id:
        One of the 11 supported benchmark IDs (see ``benchmarks.yaml``).

    Returns
    -------
    BaseBenchmarkDataset
        An uninitialised loader. Call ``.load(n_samples, seed)`` before
        iterating.

    Raises
    ------
    KeyError
        If *benchmark_id* is not in the registry.
    """
    cls = _REGISTRY.get(benchmark_id)
    if cls is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown benchmark_id {benchmark_id!r}. "
            f"Supported: {supported}"
        )
    return cls()


def list_benchmarks() -> list[str]:
    """Return all registered benchmark IDs."""
    return sorted(_REGISTRY)


__all__ = [
    # ABC
    "BaseBenchmarkDataset",
    # VLM loaders
    "AI2DDataset",
    "ChartQADataset",
    "ChartQAProDataset",
    "DocVQADataset",
    "MathVistaDataset",
    "MMMUDataset",
    "OCRBenchV2Dataset",
    "TextVQADataset",
    # Text loaders
    "CommonsenseQADataset",
    "HellaSwagDataset",
    "MMLUDataset",
    # Registry
    "get_dataset",
    "list_benchmarks",
    # Constants
    "SUPPORTED_TEXT_BENCHMARKS",
    "SUPPORTED_VLM_BENCHMARKS",
]
