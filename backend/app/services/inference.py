"""Backwards-compatible prediction entry point for the API layer."""
from __future__ import annotations

from .prediction_service import PredictionService

_service = PredictionService.create_default()


def predict_image(image_path: str):
    """Predict disease metadata for the provided image path."""
    return _service.predict(image_path).as_dict()
