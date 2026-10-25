from __future__ import annotations

import logging
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import time
from typing import Callable
from uuid import uuid4

from engimind.observability import get_logger, log_event

logger = get_logger("jobs")


@dataclass
class IndexingJob:
    job_id: str
    document_name: str
    file_bytes: bytes
    version: str
    active: bool = True
    status: str = "queued"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    result: dict | None = None
    error: str | None = None


class IndexingQueue:
    """In-memory background indexer with job polling and worker-thread execution."""

    def __init__(self, processor: Callable[[IndexingJob], dict] | None = None) -> None:
        self.jobs: dict[str, IndexingJob] = {}
        self._lock = Lock()
        self._processor = processor
        self._queue: Queue[IndexingJob] = Queue()
        self._stop = Event()
        self._worker = Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                self.mark_running(job.job_id)
                if self._processor is not None:
                    payload = self._processor(job)
                    self.mark_done(job.job_id, payload)
                else:
                    self.mark_done(job.job_id)
            except Exception as exc:  # pragma: no cover - defensive guard
                log_event(
                    logger,
                    "job.failed",
                    level=logging.ERROR,
                    job_id=job.job_id,
                    document=job.document_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                self.mark_failed(job.job_id, str(exc))
            finally:
                self._queue.task_done()

    def enqueue(self, *, document_name: str, file_bytes: bytes, version: str, active: bool = True) -> dict:
        job = IndexingJob(
            job_id=uuid4().hex,
            document_name=document_name,
            file_bytes=file_bytes,
            version=version,
            active=active,
        )
        with self._lock:
            self.jobs[job.job_id] = job
        self._queue.put(job)
        return {
            "job_id": job.job_id,
            "document_name": document_name,
            "version": version,
            "status": job.status,
            "created_at": job.created_at,
        }

    def get_status(self, job_id: str) -> dict:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown job id: {job_id}")
        payload = {
            "job_id": job.job_id,
            "document_name": job.document_name,
            "version": job.version,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        if job.result is not None:
            payload["result"] = job.result
        if job.error is not None:
            payload["error"] = job.error
        return payload

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "running"
            job.updated_at = time()

    def mark_done(self, job_id: str, result: dict | None = None) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "done"
            job.updated_at = time()
            job.result = result
            job.error = None

    def mark_failed(self, job_id: str, reason: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "failed"
            job.updated_at = time()
            job.error = reason

    def shutdown(self) -> None:
        self._stop.set()
        self._worker.join(timeout=1)
