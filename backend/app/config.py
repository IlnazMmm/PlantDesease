"""Application configuration helpers."""
from __future__ import annotations

import os


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


CONFIDENCE_THRESHOLD = _get_float("CONFIDENCE_THRESHOLD", 0.70)

REVIEW_STATUS_NOT_REQUIRED = "not_required"
REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_CONFIRMED = "confirmed"
REVIEW_STATUS_CORRECTED = "corrected"
REVIEW_REQUIRED_WARNING = "Уверенность модели ниже порога. Требуется подтверждение агрономом."
