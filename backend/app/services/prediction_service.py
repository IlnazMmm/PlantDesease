"""High level prediction workflow orchestrator."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

from PIL import Image

from ..models.knowledge_base import AdviceRepository, PlantDiseaseAdvice, get_repository
from .gradcam import GradCamGenerator
from .model_loader import ModelPredictor

logger = logging.getLogger(__name__)


def _normalize_label_part(raw: str) -> str:
    replaced = raw.replace("_", " ").replace("  ", " ")
    replaced = replaced.replace(" ,", ", ").strip()
    return " ".join(replaced.split())


@dataclass
class PredictionPayload:
    label: str
    plant: str
    disease: str
    confidence: float
    description: str
    treatment: str
    pathogen: str
    prevention: str
    gradcam_image: Image.Image | None

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "plant": self.plant,
            "disease": self.disease,
            "confidence": self.confidence,
            "description": self.description,
            "treatment": self.treatment,
            "pathogen": self.pathogen,
            "prevention": self.prevention,
            "gradcam_image": self.gradcam_image,
        }


class PredictionService:
    """Coordinates the model predictor, Grad-CAM, and domain advice."""

    def __init__(
        self,
        predictor: ModelPredictor,
        advice_repository: AdviceRepository | None = None,
        gradcam_generator: GradCamGenerator | None = None,
    ) -> None:
        self._predictor = predictor
        self._advice_repository = advice_repository or get_repository()
        self._gradcam = gradcam_generator or GradCamGenerator(predictor)

    @classmethod
    def create_default(cls) -> "PredictionService":
        predictor = ModelPredictor()
        return cls(predictor=predictor, advice_repository=get_repository(), gradcam_generator=GradCamGenerator(predictor))

    def predict(self, image_path: str) -> PredictionPayload:
        raw_prediction = self._predictor.predict(image_path)
        label = str(raw_prediction.get("class", "Unknown"))
        confidence = float(raw_prediction.get("confidence", 0.0))

        if "___" in label:
            plant_raw, disease_raw = label.split("___", maxsplit=1)
        else:
            plant_raw, disease_raw = "Unknown", label

        advice: PlantDiseaseAdvice = self._advice_repository.get(label)
        plant_name = advice.get("plant_name") or _normalize_label_part(plant_raw)
        disease_name = advice.get("title") or _normalize_label_part(disease_raw)

        gradcam_image = self._gradcam.generate(image_path, label)

        return PredictionPayload(
            label=label,
            plant=plant_name,
            disease=disease_name,
            confidence=confidence,
            description=advice.get("description", ""),
            treatment=advice.get("treatment", ""),
            pathogen=advice.get("pathogen", ""),
            prevention=advice.get("prevention", ""),
            gradcam_image=gradcam_image,
        )
