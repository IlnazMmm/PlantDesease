"""Utilities for loading and caching the inference model bundle."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import pathlib
import sys
import types
from typing import Any, Dict, Tuple

from src.infer import load_model, predict


DEFAULT_WEIGHTS_PATH = "./ml/models/model_v3.pth" #Path(__file__).resolve().parents[3] / "ml" / "models" / "model_v3.pth"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    labels: Tuple[str, ...]


@lru_cache(maxsize=2)
def _load_bundle(weights_path: str) -> ModelBundle:
    _ensure_pathlib_local_compat()
    model, classes = load_model(weights_path)
    return ModelBundle(model=model, labels=tuple(classes))


def _ensure_pathlib_local_compat() -> None:
    """
    Backward-compatibility shim for checkpoints pickled on newer Python versions.

    Some model artifacts can contain ``pathlib.Path`` internals that resolve to the
    ``pathlib._local`` module when serialized on Python 3.12+. On Python 3.10 this
    private module does not exist, causing ``torch.load`` to fail during unpickling.
    """
    if "pathlib._local" in sys.modules:
        return

    local_module = types.ModuleType("pathlib._local")
    # Use "pure" variants when the concrete OS-specific path cannot be created.
    local_module.Path = pathlib.Path
    local_module.PurePath = pathlib.PurePath
    local_module.PosixPath = pathlib.PosixPath if os.name != "nt" else pathlib.PurePosixPath
    local_module.PurePosixPath = pathlib.PurePosixPath
    local_module.WindowsPath = pathlib.WindowsPath if os.name == "nt" else pathlib.PureWindowsPath
    local_module.PureWindowsPath = pathlib.PureWindowsPath
    sys.modules["pathlib._local"] = local_module


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
