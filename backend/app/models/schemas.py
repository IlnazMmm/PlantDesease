from datetime import datetime
from typing import Literal, Optional

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


ReviewStatus = Literal["not_required", "pending", "confirmed", "corrected"]


class ReviewRequest(BaseModel):
    confirmed: bool
    expert_label: Optional[str] = None
    expert_comment: Optional[str] = None


class ReviewResponse(BaseModel):
    status: str = "saved"
    job_id: str
    review_required: bool
    review_status: ReviewStatus
    expert_label: Optional[str] = None
    expert_comment: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class ResultSummary(BaseModel):
    job_id: str
    plant: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    gradcam_url: Optional[str] = None
    label: Optional[str] = None
    created_at: Optional[datetime] = None
    review_required: bool = False
    review_status: ReviewStatus = "not_required"
    expert_label: Optional[str] = None
    expert_comment: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class ResultDetail(ResultSummary):
    description: Optional[str] = None
    treatment: Optional[str] = None
    prevention: Optional[str] = None
    pathogen: Optional[str] = None
    status: Optional[str] = None
    review_warning: Optional[str] = None
