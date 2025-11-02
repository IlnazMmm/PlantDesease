"""Общие вспомогательные утилиты для интеграционных тестов бекенда."""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import pytest
import requests
from PIL import Image
from dotenv import find_dotenv, load_dotenv


# Загружаем переменные окружения из локального файла .env, чтобы можно было
# настроить BACKEND_BASE_URL без ручного экспорта.
load_dotenv(find_dotenv(usecwd=True), override=False)


@dataclass
class ApiClient:
    """Небольшая обёртка вокруг :mod:`requests` для обращений к бекенду FastAPI."""

    session: requests.Session
    base_url: str

    def _url(self, path: str) -> str:
        """Построить абсолютный URL для переданного пути API."""

        if path.startswith("http://") or path.startswith("https://"):
            return path

        normalized_base = self.base_url.rstrip("/")
        normalized_path = path if path.startswith("/") else f"/{path}"

        return f"{normalized_base}{normalized_path}"

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        if "timeout" not in kwargs:
            kwargs["timeout"] = 60.0
        return self.session.post(self._url(path), **kwargs)

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        if "timeout" not in kwargs:
            kwargs["timeout"] = 60.0
        return self.session.get(self._url(path), **kwargs)


def require_backend_base_url() -> str:
    """Вернуть базовый URL API либо пропустить набор тестов, если он не задан."""

    base_url = os.getenv("BACKEND_BASE_URL")
    if not base_url:
        pytest.skip("BACKEND_BASE_URL must be provided to run backend API tests")
    return base_url.rstrip("/")


def generate_test_image() -> bytes:
    """Создать JPEG в памяти для загрузки через API."""

    image = Image.new("RGB", (224, 224), color=(64, 160, 84))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def wait_for_job_completion(api_client: ApiClient, job_id: str, timeout: float = 90.0) -> Dict[str, Any]:
    """Опросить эндпоинт статуса до завершения задачи или истечения тайм-аута."""

    deadline = time.time() + timeout
    last_payload: Dict[str, Any] | None = None

    while time.time() < deadline:
        response = api_client.get(f"/api/v1/status/{job_id}")
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        if payload.get("status") in {"done", "error"}:
            return payload
        time.sleep(0.5)

    raise AssertionError(
        f"Job {job_id} did not finish within the timeout. Last payload: {last_payload}"
    )
