from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class UploadResponse(BaseModel):
    file_id: str

class PredictRequest(BaseModel):
    file_id: str
    model_version: str = "v1"

class JobResponse(BaseModel):
    job_id: str
    status: str

class FeedbackRequest(BaseModel):
    job_id: str
    correct_label: str


class ResultSummary(BaseModel):
    job_id: str
    plant: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    gradcam_url: Optional[str] = None
    label: Optional[str] = None
    created_at: Optional[datetime] = None


class ResultDetail(ResultSummary):
    description: Optional[str] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    pathogen: Optional[str] = None
    status: Optional[str] = None
