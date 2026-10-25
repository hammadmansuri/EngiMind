from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from openai import OpenAI

from engimind.core.config import load_settings
from engimind.documents.lifecycle import DocumentVersionRegistry
from engimind.jobs.indexing_queue import IndexingJob, IndexingQueue
from engimind.application.rag import answer_question, ingest_document
from engimind.observability import (
    configure_logging,
    current_trace_id,
    get_logger,
    log_event,
    request_context,
)

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

registry = DocumentVersionRegistry()


def process_indexing_job(job: IndexingJob) -> dict:
    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    result = ingest_document(job.file_bytes, job.document_name, settings, client)
    record = registry.register_version(
        document_name=job.document_name,
        file_hash=result.file_hash,
        collection_name=result.collection,
        pages=result.pages,
        chunks=result.chunks,
        version=job.version,
        active=job.active,
    )
    return {
        "document_id": record["document_id"],
        "document_name": job.document_name,
        "version": record["version"],
        "collection": result.collection,
        "pages": result.pages,
        "chunks": result.chunks,
        "active": record["active"],
    }


queue = IndexingQueue(processor=process_indexing_job)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def get_api_key(request: Request, provided_key: str | None = Depends(api_key_header)) -> None:
    configured_key = os.getenv("ENGIMIND_API_KEY", "").strip()
    if configured_key:
        if provided_key != configured_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    yield


app = FastAPI(
    title="EngiMind RAG API",
    version="1.0.0",
    description="Production-style PDF and CSV retrieval and answer API with document version tracking.",
    lifespan=lifespan,
)
logger = get_logger("api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if _SAFE_REQUEST_ID.fullmatch(supplied) else None
    supplied_trace = request.headers.get("x-trace-id", "")
    trace_id = supplied_trace if _SAFE_REQUEST_ID.fullmatch(supplied_trace) else None
    with request_context(request_id, trace_id) as active_id:
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            log_event(
                logger,
                "http.request.failed",
                level=logging.ERROR,
                method=request.method,
                path=request.url.path,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        log_event(
            logger,
            "http.request.completed",
            level=logging.WARNING if response.status_code >= 400 else logging.INFO,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        response.headers["X-Request-ID"] = active_id
        trace = current_trace_id()
        if trace:
            response.headers["X-Trace-ID"] = trace
        return response


class AskRequest(BaseModel):
    question: str
    collection: str | list[str]
    chat_history: list[list[str] | tuple[str, str]] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "engimind-rag", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/documents")
def list_documents(_: None = Depends(get_api_key)) -> dict:
    return {"documents": registry.list_documents()}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str = Path(..., description="Job identifier"), _: None = Depends(get_api_key)) -> dict:
    return queue.get_status(job_id)


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    version: str | None = Form(None),
    activate: bool = Form(True),
    _: None = Depends(get_api_key),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A PDF or CSV filename is required.")

    content = await file.read()
    suffix = file.filename.rsplit(".", 1)[-1].lower()
    if suffix not in {"pdf", "csv"}:
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")
    if suffix == "pdf" and content and b"%PDF" not in content[:1024]:
        raise HTTPException(status_code=400, detail="File content is not a valid PDF.")

    version_name = version or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    job = queue.enqueue(
        document_name=file.filename,
        file_bytes=content,
        version=version_name,
        active=activate,
    )

    return {
        "message": "Document indexing queued.",
        "document_name": file.filename,
        "version": version_name,
        "job": job,
        "active": activate,
    }


@app.post("/documents/upload-batch")
async def upload_documents(
    files: list[UploadFile] = File(...),
    _: None = Depends(get_api_key),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    file_payloads = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Every file must have a filename.")
        suffix = file.filename.rsplit(".", 1)[-1].lower()
        if suffix not in {"pdf", "csv"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Only PDF and CSV are supported.",
            )
        content = await file.read()
        if suffix == "pdf" and content and b"%PDF" not in content[:1024]:
            raise HTTPException(
                status_code=400,
                detail=f"File content is not a valid PDF: {file.filename}.",
            )
        file_payloads.append((file.filename, content))

    queued = []
    for filename, content in file_payloads:
        version_name = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid4().hex[:8]}"
        )
        job = queue.enqueue(
            document_name=filename,
            file_bytes=content,
            version=version_name,
            active=True,
        )
        queued.append(
            {
                "document_name": filename,
                "version": version_name,
                "job": job,
            }
        )
    return {"message": "Document indexing queued.", "documents": queued}


@app.post("/documents/reindex")
async def reindex_document(
    file: UploadFile = File(...),
    version: str | None = Form(None),
    _: None = Depends(get_api_key),
) -> dict:
    return await upload_document(file=file, version=version, activate=True)


@app.post("/ask")
def ask_question(payload: AskRequest, _: None = Depends(get_api_key)) -> dict:
    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    history: list[tuple[str, str]] = []
    for item in payload.chat_history or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            history.append((str(item[0]), str(item[1])))

    result = answer_question(
        payload.question,
        payload.collection,
        settings,
        client,
        chat_history=history,
    )

    return {
        "answer": result.answer,
        "refused": result.refused,
        "chunks": [
            {
                "text": chunk.text,
                "page": chunk.page,
                "row": chunk.page if chunk.source.lower().endswith(".csv") else None,
                "page_end": chunk.end_page,
                "row_end": (
                    chunk.end_page if chunk.source.lower().endswith(".csv") else None
                ),
                "source": chunk.source,
                "section": chunk.section,
                "distance": chunk.distance,
            }
            for chunk in result.chunks
        ],
        "prompt": result.prompt,
    }
