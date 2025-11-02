"""Общие вспомогательные утилиты для интеграционных тестов бекенда."""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict
from urllib.parse import quote_plus

import pytest
import requests
from PIL import Image
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


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


def _build_database_url() -> str:
    """Собрать URL базы данных SQLAlchemy из переменных окружения."""

    if url := os.getenv("DATABASE_URL"):
        return url

    user = os.getenv("POSTGRES_USER", "postgres")
    password = quote_plus(os.getenv("POSTGRES_PASSWORD", "postgres"))
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "postgres")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    """Создать (и закэшировать) движок SQLAlchemy для прямого доступа к базе."""

    database_url = _build_database_url()
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)

    # Рано падаем, если база недоступна, чтобы сразу увидеть проблемы конфигурации.
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return engine


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker:
    """Вернуть лениво инициализируемую фабрику сессий, привязанную к тестовой БД."""

    engine = _get_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def persist_result_record(job_id: str, file_id: str, result_payload: Dict[str, Any]) -> None:
    """Гарантировать наличие строки в таблице результатов для указанного job_id.

    В продакшене бекенд сохраняет результаты инференса фоновой задачей.
    В CI эта задача может быть отключена или заглушена, поэтому тесты создают
    минимальную запись самостоятельно, чтобы проверить позитивный сценарий
    эндпоинта «result».
    """

    session_factory = _get_session_factory()
    with session_factory() as session:
        existing = session.execute(
            text("SELECT 1 FROM results WHERE job_id = :job_id"),
            {"job_id": job_id},
        ).first()
        if existing:
            return

        upload_row = session.execute(
            text("SELECT path FROM uploads WHERE file_id = :file_id"),
            {"file_id": file_id},
        ).mappings().first()

        file_path = ""
        if upload_row and "path" in upload_row:
            file_path = str(upload_row["path"])

        normalized_result = result_payload or {}
        created_at_value = normalized_result.get("created_at")
        if isinstance(created_at_value, str):
            try:
                created_at_value = datetime.fromisoformat(
                    created_at_value.replace("Z", "+00:00")
                )
            except ValueError:
                created_at_value = datetime.utcnow()
        elif not isinstance(created_at_value, datetime):
            created_at_value = datetime.utcnow()

        record_data = {
            "job_id": job_id,
            "file_path": file_path,
            "plant": normalized_result.get("plant") or "Unknown",
            "disease": normalized_result.get("disease") or "Unknown",
            "confidence": float(normalized_result.get("confidence") or 0.0),
            "gradcam_path": None,
            "label": normalized_result.get("label"),
            "description": normalized_result.get("description") or "",
            "treatment": normalized_result.get("treatment") or "",
            "prevention": normalized_result.get("prevention") or "",
            "pathogen": normalized_result.get("pathogen") or "",
            "created_at": created_at_value,
        }

        session.execute(
            text(
                """
                INSERT INTO results (
                    job_id, file_path, plant, disease, confidence, gradcam_path,
                    label, description, treatment, prevention, pathogen, created_at
                ) VALUES (
                    :job_id, :file_path, :plant, :disease, :confidence, :gradcam_path,
                    :label, :description, :treatment, :prevention, :pathogen, :created_at
                )
                """
            ),
            record_data,
        )
        session.commit()

