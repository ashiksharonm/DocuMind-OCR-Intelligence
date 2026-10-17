"""Async job + human review API (mounted under /v1)."""

import hashlib
import json
import os
import re
import uuid
from typing import Any, Dict, Iterator, Literal, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.review.processing import load_pages
from app.review.store import FIELD_NAMES, Store

router = APIRouter(prefix="/v1", tags=["review"])
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def auth(x_api_key: Optional[str] = Header(None)) -> None:
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key")


def store_of(request: Request) -> Store:
    return request.app.state.store  # type: ignore[no-any-return]


class Decision(BaseModel):
    action: Literal["confirm", "correct"]
    reviewer: str = Field(min_length=1, max_length=80)
    value: Optional[str] = Field(None, max_length=200)


def _validate_value(name: str, value: Optional[str]) -> str:
    if value is None or not value.strip():
        raise HTTPException(422, "'value' is required when action is 'correct'")
    value = value.strip()
    if name == "invoice_date" and not _DATE.match(value):
        raise HTTPException(422, "invoice_date must be YYYY-MM-DD")
    if name == "total_amount":
        try:
            float(value.replace(",", ""))
        except ValueError:
            raise HTTPException(422, "total_amount must be a number") from None
    return value


@router.post("/jobs", status_code=202, dependencies=[Depends(auth)])
async def submit_job(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Queue a document. Returns immediately; poll ``GET /v1/jobs/{id}``."""
    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {settings.MAX_UPLOAD_BYTES} bytes")
    if data[:5] != b"%PDF-" and not load_pages(data):
        raise HTTPException(400, "not a decodable image or PDF")
    jid = uuid.uuid4().hex
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, jid)  # never derived from the client file name
    with open(path, "wb") as fh:
        fh.write(data)
    store = store_of(request)
    store.create_job(
        os.path.basename(file.filename or "upload")[:120],
        file.content_type or "",
        hashlib.sha256(data).hexdigest(),
        path,
        settings.REVIEW_THRESHOLD,
        jid,
    )
    request.app.state.dispatcher.publish(jid)
    return {"job_id": jid, "status": "queued", "status_url": f"/v1/jobs/{jid}"}


@router.get("/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str, request: Request) -> Dict[str, Any]:
    job = store_of(request).get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    for k in ("file_path", "preview_path"):
        job.pop(k, None)
    return job


@router.get("/jobs/{job_id}/image", dependencies=[Depends(auth)])
def job_image(job_id: str, request: Request) -> FileResponse:
    job = store_of(request).get_job(job_id)
    if job is None or not job.get("preview_path") or not os.path.exists(job["preview_path"]):
        raise HTTPException(404, "no preview available")
    return FileResponse(job["preview_path"], media_type="image/png")


@router.get("/reviews/queue", dependencies=[Depends(auth)])
def review_queue(
    request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """Fields waiting for a human, lowest confidence first."""
    items = store_of(request).review_queue(limit, offset)
    return {"count": len(items), "items": items}


@router.post("/reviews/{job_id}/fields/{field}", dependencies=[Depends(auth)])
def review_field(job_id: str, field: str, decision: Decision, request: Request) -> Dict[str, Any]:
    if field not in FIELD_NAMES:
        raise HTTPException(404, f"unknown field; one of {list(FIELD_NAMES)}")
    value = _validate_value(field, decision.value) if decision.action == "correct" else None
    job = store_of(request).review_field(job_id, field, decision.reviewer, value, decision.action)
    if job is None:
        raise HTTPException(404, "unknown job or field")
    job.pop("file_path", None)
    job.pop("preview_path", None)
    return job


@router.get("/dataset/export", dependencies=[Depends(auth)])
def export_dataset(request: Request, include: Literal["reviewed", "all"] = "reviewed") -> StreamingResponse:
    """JSONL of labelled documents. ``reviewed`` (default) holds only human-decided labels."""

    def lines() -> Iterator[str]:
        for rec in store_of(request).export_dataset(include):
            yield json.dumps(rec, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@router.get("/stats", dependencies=[Depends(auth)])
def stats(request: Request) -> Dict[str, Any]:
    return store_of(request).stats()
