import traceback

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
import shutil
import os
from .services import inference as infer_service
# from .services import storage as storage_service
from .models import schemas, db_models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
GRADCAM_DIR = BASE_DIR / "static" / "gradcam"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# Simple SQLite DB (metadata)
DATABASE_URL = f"sqlite:///{BASE_DIR / 'app.db'}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
db_models.Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (for demo)
JOBS = {}

@app.post("/api/v1/upload", response_model=schemas.UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    file_id = str(uuid4())
    out_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # store minimal metadata in sqlite
    db = SessionLocal()
    meta = db_models.Upload(file_id=file_id, path=str(out_path), filename=file.filename)
    db.add(meta)
    db.commit()
    db.close()
    return {"file_id": file_id}

@app.post("/api/v1/predict", response_model=schemas.JobResponse)
def predict(payload: schemas.PredictRequest, background_tasks: BackgroundTasks):
    # find file path
    db = SessionLocal()
    upload = db.query(db_models.Upload).filter_by(file_id=payload.file_id).first()
    db.close()
    if not upload:
        raise HTTPException(status_code=404, detail="file_id not found")
    job_id = str(uuid4())
    JOBS[job_id] = {"status": "done"}
    # background processing
    background_tasks.add_task(_process_job, job_id, upload.path)
    return {"job_id": job_id, "status": "done"}

def _process_job(job_id: str, path: str):
    try:
        JOBS[job_id]["status"] = "running"
        result = infer_service.predict_image(path)
        # save gradcam image
        gradcam_path = GRADCAM_DIR / f"{job_id}.png"
        # if "gradcam_image" in result:
        #     result["gradcam_image"].save(gradcam_path)
        # save into simple sqlite table
        db = SessionLocal()
        print(result)
        # print(result.get("confidence",0.0))
        # record = db_models.Result(job_id=job_id, file_path=path, plant=result.get("plant", ""), disease=result.get("disease",""), confidence=float(result.get("confidence",0.0)), gradcam_path=str(gradcam_path))
        # db.add(record)
        # db.commit()
        db.close()
        JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        print("❌ Error in _process_job:", e)
        traceback.print_exc()
        JOBS[job_id] = {"status": "error", "error": str(e)}

@app.get("/api/v1/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JOBS[job_id]

@app.get("/api/v1/result/{job_id}")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job["status"] != "done":
        return {"status": job["status"]}
    res = job["result"]
    gradcam_url = f"/static/gradcam/{job_id}.png" if (GRADCAM_DIR / f"{job_id}.png").exists() else None
    response = {
        "plant": res.get("plant", "Unknown"),
        "disease": res.get("disease", "Unknown"),
        "confidence": res.get("confidence", 0.0),
        "gradcam_url": gradcam_url,
        "description": res.get("description", ""),
        "treatment": res.get("treatment", "")
    }
    return JSONResponse(response)

@app.post("/api/v1/feedback")
def feedback(payload: schemas.FeedbackRequest):
    # store feedback to DB (simple)
    db = SessionLocal()
    rec = db_models.Feedback(job_id=payload.job_id, correct_label=payload.correct_label)
    db.add(rec)
    db.commit()
    db.close()
    return {"status": "saved"}

@app.get("/static/gradcam/{name}")
def serve_gradcam(name: str):
    p = GRADCAM_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p)
