from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    metric_name: str

    @abstractmethod
    def score(self, prediction: Any, reference: Any) -> float:
        """Return a scalar score in [0, 1]."""

