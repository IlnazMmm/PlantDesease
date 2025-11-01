import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .db import get_session
from .models import db_models, schemas
from .services import inference as infer_service
from .services.job_store import JobStatus, job_store

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
GRADCAM_DIR = BASE_DIR / "static" / "gradcam"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/upload", response_model=schemas.UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    file_id = str(uuid4())
    out_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # store minimal metadata in the relational database
    with get_session() as db:
        meta = db_models.Upload(file_id=file_id, path=str(out_path), filename=file.filename)
        db.add(meta)
    return {"file_id": file_id}

@app.post("/api/v1/predict", response_model=schemas.JobResponse)
def predict(payload: schemas.PredictRequest, background_tasks: BackgroundTasks):
    # find file path
    with get_session() as db:
        upload = db.query(db_models.Upload).filter_by(file_id=payload.file_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="file_id not found")
    job_id = str(uuid4())
    job_store.create(job_id)
    background_tasks.add_task(_process_job, job_id, upload.path)
    return {"job_id": job_id, "status": JobStatus.PENDING.value}

def _process_job(job_id: str, path: str):
    try:
        job_store.mark_running(job_id)
        result = infer_service.predict_image(path)
        gradcam_image = result.pop("gradcam_image", None)
        gradcam_url = None
        completed_at = datetime.utcnow()
        if gradcam_image is not None:
            gradcam_path = GRADCAM_DIR / f"{job_id}.png"
            gradcam_image.save(gradcam_path)
            gradcam_url = f"/static/gradcam/{job_id}.png"

        with get_session() as db:
            record = db_models.Result(
                job_id=job_id,
                file_path=path,
                plant=result.get("plant", ""),
                disease=result.get("disease", ""),
                confidence=float(result.get("confidence", 0.0)),
                gradcam_path=str(GRADCAM_DIR / f"{job_id}.png") if gradcam_image is not None else None,
                label=result.get("label"),
                description=result.get("description"),
                treatment=result.get("treatment"),
                prevention=result.get("prevention"),
                pathogen=result.get("pathogen"),
                created_at=completed_at,
            )
            db.add(record)

        payload: Dict[str, object] = {
            **result,
            "gradcam_url": gradcam_url,
            "job_id": job_id,
            "created_at": completed_at.isoformat(),
        }
        job_store.mark_done(job_id, payload)
        logger.info("Job %s finished", job_id)
    except Exception as exc:
        logger.exception("Error processing job %s", job_id)
        job_store.mark_error(job_id, str(exc))

@app.get("/api/v1/status/{job_id}")
def get_status(job_id: str):
    try:
        record = job_store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")
    return record.as_dict()

@app.get("/api/v1/result/{job_id}", response_model=schemas.ResultDetail)
def get_result(job_id: str):
    record = None
    try:
        record = job_store.get(job_id)
    except KeyError:
        record = None

    if record is not None and record.status != JobStatus.DONE:
        return {"status": record.status.value, "job_id": job_id}

    if record is not None and record.result is not None:
        payload = dict(record.result)
        payload.setdefault("job_id", job_id)
    else:
        with get_session() as db:
            db_result = (
                db.query(db_models.Result).filter_by(job_id=job_id).first()
            )
        if db_result is None:
            raise HTTPException(status_code=404, detail="job_id not found")

        gradcam_url = None
        if db_result.gradcam_path:
            gradcam_path = Path(db_result.gradcam_path)
            if gradcam_path.exists():
                gradcam_url = f"/static/gradcam/{gradcam_path.name}"

        payload = {
            "job_id": job_id,
            "plant": db_result.plant or "Unknown",
            "disease": db_result.disease or "Unknown",
            "confidence": float(db_result.confidence or 0.0),
            "gradcam_url": gradcam_url,
            "description": db_result.description or "",
            "treatment": db_result.treatment or "",
            "prevention": db_result.prevention or "",
            "pathogen": db_result.pathogen or "",
            "label": db_result.label or None,
            "created_at": db_result.created_at.isoformat() if db_result.created_at else None,
        }

    return payload


@app.get("/api/v1/history", response_model=List[schemas.ResultSummary])
def list_history(limit: int = 10):
    limit = max(1, min(limit, 50))
    with get_session() as db:
        query = (
            db.query(db_models.Result)
            .order_by(db_models.Result.created_at.desc(), db_models.Result.id.desc())
            .limit(limit)
        )
        rows = list(query)

    items: List[schemas.ResultSummary] = []
    for row in rows:
        gradcam_url = None
        if row.gradcam_path:
            gradcam_path = Path(row.gradcam_path)
            if gradcam_path.exists():
                gradcam_url = f"/static/gradcam/{gradcam_path.name}"

        items.append(
            schemas.ResultSummary(
                job_id=row.job_id,
                plant=row.plant or "Unknown",
                disease=row.disease or "Unknown",
                confidence=float(row.confidence or 0.0),
                gradcam_url=gradcam_url,
                label=row.label or None,
                created_at=row.created_at,
            )
        )

    return items

@app.post("/api/v1/feedback")
def feedback(payload: schemas.FeedbackRequest):
    # store feedback to DB (simple)
    with get_session() as db:
        rec = db_models.Feedback(job_id=payload.job_id, correct_label=payload.correct_label)
        db.add(rec)
    return {"status": "saved"}

@app.get("/static/gradcam/{name}")
def serve_gradcam(name: str):
    p = GRADCAM_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p)
