"""HTTP-интеграционные тесты для сервиса FastAPI бекенда."""
from __future__ import annotations

from typing import Dict

import pytest

from .helpers import ApiClient, generate_test_image, wait_for_job_completion


def test_upload_endpoint_returns_file_id(api_client: ApiClient) -> None:
    response = api_client.post(
        "/api/v1/upload",
        files={"file": ("leaf.jpg", generate_test_image(), "image/jpeg")},
    )
    response.raise_for_status()
    payload = response.json()

    assert response.status_code == 200
    assert "file_id" in payload
    assert isinstance(payload["file_id"], str)


def test_predict_endpoint_returns_job_id(api_client: ApiClient, uploaded_file_id: str) -> None:
    response = api_client.post("/api/v1/predict", json={"file_id": uploaded_file_id})
    response.raise_for_status()
    payload = response.json()

    assert response.status_code == 200
    assert "job_id" in payload
    assert isinstance(payload["job_id"], str)


def test_status_endpoint_reports_completion(api_client: ApiClient, prediction_job: Dict[str, str]) -> None:
    payload = wait_for_job_completion(api_client, prediction_job["job_id"])

    assert payload.get("status") == "done"
    assert "result" in payload


def test_result_endpoint_returns_prediction_details(
    api_client: ApiClient, completed_job: Dict[str, Dict[str, str]]
) -> None:
    response = api_client.get(f"/api/v1/result/{completed_job['job_id']}")
    response.raise_for_status()
    payload = response.json()

    assert payload.get("job_id") == completed_job["job_id"]
    assert payload.get("status") in {None, "done"}
    assert isinstance(payload.get("plant"), str)
    assert isinstance(payload.get("disease"), str)


def test_history_endpoint_lists_recent_jobs(api_client: ApiClient, completed_job: Dict[str, Dict[str, str]]) -> None:
    response = api_client.get("/api/v1/history?limit=10")
    response.raise_for_status()
    payload = response.json()

    assert isinstance(payload, list)
    assert any(item.get("job_id") == completed_job["job_id"] for item in payload)


def test_feedback_endpoint_accepts_submission(api_client: ApiClient, completed_job: Dict[str, Dict[str, str]]) -> None:
    label = completed_job["result"].get("label") or "unknown"
    response = api_client.post(
        "/api/v1/feedback",
        json={"job_id": completed_job["job_id"], "correct_label": label},
    )
    response.raise_for_status()
    payload = response.json()

    assert payload.get("status") == "saved"


def test_gradcam_endpoint_serves_image(api_client: ApiClient, completed_job: Dict[str, Dict[str, str]]) -> None:
    gradcam_url = completed_job["result"].get("gradcam_url")
    if not gradcam_url:
        pytest.skip("Grad-CAM not enabled for this deployment")

    response = api_client.get(gradcam_url)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    assert "image" in content_type


def test_predict_missing_file_returns_404(api_client: ApiClient) -> None:
    response = api_client.post("/api/v1/predict", json={"file_id": "missing"})

    assert response.status_code == 404
    assert response.json().get("detail") == "file_id not found"


def test_upload_without_file_returns_422(api_client: ApiClient) -> None:
    """Проверка, что отсутствие файла в запросе даёт ошибку валидации."""

    response = api_client.post("/api/v1/upload")

    assert response.status_code == 422


def test_predict_without_file_id_returns_422(api_client: ApiClient) -> None:
    """Проверка, что эндпоинт /predict требует обязательный параметр file_id."""

    response = api_client.post("/api/v1/predict", json={})

    assert response.status_code == 422


def test_status_unknown_job_returns_404(api_client: ApiClient) -> None:
    """Проверка, что запрос статуса для несуществующей задачи завершаетcя 404."""

    response = api_client.get("/api/v1/status/unknown-job")

    assert response.status_code == 404
    assert response.json().get("detail") == "job_id not found"


def test_result_unknown_job_returns_404(api_client: ApiClient) -> None:
    """Проверка, что эндпоинт результата возвращает 404 для неизвестного job_id."""

    response = api_client.get("/api/v1/result/unknown-job")

    assert response.status_code == 404
    assert response.json().get("detail") == "job_id not found"


def test_feedback_without_payload_returns_422(api_client: ApiClient) -> None:
    """Проверка, что отправка пустого тела фидбека приводит к ошибке 422."""

    response = api_client.post("/api/v1/feedback", json={})

    assert response.status_code == 422


def test_gradcam_unknown_image_returns_404(api_client: ApiClient) -> None:
    """Проверка, что запрос несуществующего Grad-CAM изображения даёт 404."""

    response = api_client.get("/static/gradcam/missing.png")

    assert response.status_code == 404
    assert response.json().get("detail") == "not found"

