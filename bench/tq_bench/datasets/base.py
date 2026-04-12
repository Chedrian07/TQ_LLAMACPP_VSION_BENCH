from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset


class BaseBenchmarkDataset(ABC):
    """Abstract base for all benchmark dataset loaders.

    Subclasses must set ``benchmark_id`` and implement ``load()`` /
    ``iter_samples()``.  The ``load()`` method downloads from HuggingFace
    and stores ``n_samples`` items deterministically via *seed*.
    """

    benchmark_id: str

    def __init__(self) -> None:
        self._samples: list[dict[str, Any]] = []
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, n_samples: int, seed: int = 42) -> None:
        """Download / cache the dataset and sample *n_samples* items.

        After this method returns, ``iter_samples()`` must be usable.
        """

    @abstractmethod
    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        """Yield normalised sample dicts for the runner."""

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_sample(dataset: Dataset, n: int, seed: int) -> Dataset:
        """Return a reproducible random subset of *dataset*.

        If *n* is -1 or >= len(dataset) the full dataset is returned
        (shuffled).  ``n=-1`` is the sentinel for parity-mode "use the
        full split".
        """
        shuffled = dataset.shuffle(seed=seed)
        if n < 0 or n >= len(shuffled):
            return shuffled
        return shuffled.select(range(n))

    @staticmethod
    def _format_mcq_options(options: list[str]) -> str:
        """Format a list of option strings as ``A. ... \\nB. ...``."""
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines: list[str] = []
        for idx, opt in enumerate(options):
            lines.append(f"{labels[idx]}. {opt}")
        return "\n".join(lines)

    @staticmethod
    def _index_to_label(index: int) -> str:
        """Convert a 0-based index to an uppercase letter label."""
        return chr(ord("A") + index)
