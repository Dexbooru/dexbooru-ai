"""Abstract base for skops-backed sklearn estimators used by the API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from services.sklearn_model import load_sklearn_estimator


class BasePredictor(ABC):
    """Loads a single estimator from ``.skops``; subclasses define input transform and prediction."""

    _estimator: Any

    def __init__(self, skops_path: Path) -> None:
        self._skops_path = Path(skops_path)
        self._estimator = load_sklearn_estimator(self._skops_path)

    @abstractmethod
    def transform_data(self, tag_string: str) -> str:
        """Normalize input into the exact representation passed to the sklearn ``predict`` call."""

    @abstractmethod
    def predict(self, tag_string: str) -> tuple[str, str, dict[str, float]]:
        """Return ``(transformed_input, predicted_class, class_name -> percent)``."""
