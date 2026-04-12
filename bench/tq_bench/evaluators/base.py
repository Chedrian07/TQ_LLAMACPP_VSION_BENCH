from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    metric_name: str

    @abstractmethod
    def score(
        self,
        prediction: Any,
        reference: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """Return a scalar score in [0, 1].

        Parameters
        ----------
        prediction:
            Model output text.
        reference:
            Ground-truth answer (string, list of strings, etc.).
        metadata:
            Optional per-sample metadata dict from the dataset loader.
            Parity evaluators use this for question_type, options, precision,
            etc.  Existing evaluators ignore it.
        """

