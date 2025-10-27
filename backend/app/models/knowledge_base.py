"""Plant disease advice repository backed by static JSON data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Mapping, MutableMapping


class PlantDiseaseAdvice(dict):
    """Typed mapping describing symptoms and care tips for a label."""

    title: str
    pathogen: str
    plant_name: str
    description: str
    treatment: str
    prevention: str


DEFAULT_ADVICE: PlantDiseaseAdvice = PlantDiseaseAdvice(
    title="Информация уточняется",
    plant_name="Неизвестное растение",
    description=(
        "Мы пока не собрали описание для этого класса заболевания. "
        "Пожалуйста, сверяйтесь с агрономом для уточнения деталей."
    ),
    treatment="Соберите больше данных и проконсультируйтесь со специалистом.",
    pathogen="",
    prevention="",
)


def _canonical_label(label: str) -> str:
    return label.strip().lower()


def _open_resource(package: str, resource_name: str):
    try:
        return resources.files(package).joinpath(resource_name).open("r", encoding="utf-8")
    except AttributeError:
        return resources.open_text(package, resource_name, encoding="utf-8")


@dataclass(frozen=True)
class AdviceRepository:
    """Provides read-only access to curated plant disease advice."""

    _advice_map: Mapping[str, PlantDiseaseAdvice]
    _default: PlantDiseaseAdvice = field(
        default_factory=lambda: PlantDiseaseAdvice(DEFAULT_ADVICE)
    )

    @classmethod
    def from_json_resource(cls, package: str, resource_name: str) -> "AdviceRepository":
        data = cls._load_resource(package, resource_name)
        prepared: MutableMapping[str, PlantDiseaseAdvice] = {}
        for raw_label, payload in data.items():
            canonical = _canonical_label(raw_label)
            merged = PlantDiseaseAdvice(DEFAULT_ADVICE)
            merged.update(payload)
            prepared[canonical] = merged
        return cls(prepared)

    @staticmethod
    def _load_resource(package: str, resource_name: str) -> Mapping[str, PlantDiseaseAdvice]:
        with _open_resource(package, resource_name) as handle:
            return json.load(handle)

    def get(self, label: str) -> PlantDiseaseAdvice:
        return self._advice_map.get(_canonical_label(label), self._default)


@lru_cache(maxsize=1)
def get_repository() -> AdviceRepository:
    return AdviceRepository.from_json_resource("app.data", "plant_advice.json")


def get_advice_for_label(label: str) -> PlantDiseaseAdvice:
    return get_repository().get(label)
