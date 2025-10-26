"""Utilities for loading and caching the inference model bundle."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

from src.infer import load_model, predict


DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "ml" / "models" / "model_v3.pth"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    labels: Tuple[str, ...]


@lru_cache(maxsize=2)
def _load_bundle(weights_path: str) -> ModelBundle:
    model, classes = load_model(weights_path)
    return ModelBundle(model=model, labels=tuple(classes))


class ModelPredictor:
    """Wraps the training package helpers with caching and simple helpers."""

    def __init__(self, weights_path: Path | str | None = None) -> None:
        self._weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    @property
    def bundle(self) -> ModelBundle:
        return _load_bundle(str(self._weights_path))

    def predict(self, image_path: str) -> Dict[str, float]:
        bundle = self.bundle
        return predict(bundle.model, bundle.labels, image_path)

    def model(self) -> Any:
        return self.bundle.model

    def labels(self) -> Tuple[str, ...]:
        return self.bundle.labels

    def label_index(self, label: str) -> int | None:
        try:
            return self.labels().index(label)
        except ValueError:
            return None
