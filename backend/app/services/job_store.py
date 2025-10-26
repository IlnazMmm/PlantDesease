from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class JobRecord:
    status: JobStatus = JobStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[dict]]:
        payload: Dict[str, Optional[dict]] = {"status": self.status.value}
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


class JobStore:
    """Thread-safe in-memory job registry used while a persistent queue is absent."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, job_id: str) -> JobRecord:
        with self._lock:
            record = JobRecord()
            self._jobs[job_id] = record
            return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def mark_running(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.RUNNING
            record.error = None
            return record

    def mark_done(self, job_id: str, result: dict) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.DONE
            record.result = result
            record.error = None
            return record

    def mark_error(self, job_id: str, message: str) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.ERROR
            record.error = message
            record.result = None
            return record


job_store = JobStore()
