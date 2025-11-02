"""Общие фикстуры pytest для интеграционных тестов бекенда."""
from __future__ import annotations

from typing import Dict

import pytest
import requests

from .helpers import (
    ApiClient,
    generate_test_image,
    persist_result_record,
    require_backend_base_url,
    wait_for_job_completion,
)


@pytest.fixture(scope="session")
def api_client() -> ApiClient:
    base_url = require_backend_base_url()
    with requests.Session() as session:
        yield ApiClient(session=session, base_url=base_url)


@pytest.fixture()
def uploaded_file_id(api_client: ApiClient) -> str:
    response = api_client.post(
        "/api/v1/upload",
        files={"file": ("leaf.jpg", generate_test_image(), "image/jpeg")},
    )
    response.raise_for_status()
    payload = response.json()
    assert "file_id" in payload
    return payload["file_id"]


@pytest.fixture()
def prediction_job(api_client: ApiClient, uploaded_file_id: str) -> Dict[str, str]:
    response = api_client.post("/api/v1/predict", json={"file_id": uploaded_file_id})
    response.raise_for_status()
    payload = response.json()
    assert "job_id" in payload
    return {"job_id": payload["job_id"], "file_id": uploaded_file_id}


@pytest.fixture()
def completed_job(api_client: ApiClient, prediction_job: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    status_payload = wait_for_job_completion(api_client, prediction_job["job_id"])
    assert status_payload.get("status") == "done"
    result_payload = status_payload.get("result") or {}
    persist_result_record(
        job_id=prediction_job["job_id"],
        file_id=prediction_job["file_id"],
        result_payload=result_payload,
    )
    return {
        "job_id": prediction_job["job_id"],
        "file_id": prediction_job["file_id"],
        "status": status_payload,
        "result": result_payload,
    }

