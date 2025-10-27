from datetime import datetime

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, DateTime

Base = declarative_base()

class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True)
    path = Column(String)
    filename = Column(String)

class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    file_path = Column(String)
    plant = Column(String)
    disease = Column(String)
    confidence = Column(Float)
    gradcam_path = Column(String)
    label = Column(String)
    description = Column(Text)
    treatment = Column(Text)
    prevention = Column(Text)
    pathogen = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String)
    correct_label = Column(String)
