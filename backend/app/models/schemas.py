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
