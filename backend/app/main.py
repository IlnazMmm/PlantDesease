import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import db_models, schemas
from .services import inference as infer_service
from .services.job_store import JobStatus, job_store

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
GRADCAM_DIR = BASE_DIR / "static" / "gradcam"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# Simple SQLite DB (metadata)
DATABASE_URL = f"sqlite:///{BASE_DIR / 'app.db'}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
db_models.Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def get_session() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@app.post("/api/v1/upload", response_model=schemas.UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    file_id = str(uuid4())
    out_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # store minimal metadata in sqlite
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
            )
            db.add(record)

        payload: Dict[str, object] = {**result, "gradcam_url": gradcam_url}
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

@app.get("/api/v1/result/{job_id}")
def get_result(job_id: str):
    try:
        record = job_store.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job_id not found")

    if record.status != JobStatus.DONE:
        return {"status": record.status.value}

    assert record.result is not None
    response = {
        "plant": record.result.get("plant", "Unknown"),
        "disease": record.result.get("disease", "Unknown"),
        "confidence": record.result.get("confidence", 0.0),
        "gradcam_url": record.result.get("gradcam_url"),
        "description": record.result.get("description", ""),
        "treatment": record.result.get("treatment", ""),
    }
    return JSONResponse(response)

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
