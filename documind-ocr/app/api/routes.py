"""
routes.py — DocuMind OCR API Endpoints
========================================

Endpoints
---------
POST /extract        — Single document extraction (PDF or image)
POST /batch          — Batch extraction from a ZIP archive
GET  /metrics        — Real-time latency and SLA compliance stats
GET  /health         — Health check (defined in main.py)
"""

import asyncio
import uuid
import time
import os
import json
import logging
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2

from app.pipelines.ingest import process_upload
from app.pipelines.preprocess import preprocess_pipeline
from app.pipelines.ocr import run_ocr
from app.pipelines.classify import classify_document
from app.pipelines.extract_fields import extract_fields
from app.pipelines.summarize import summarize_document
from app.pipelines.batch_processor import BatchProcessor
from app.schemas.response_models import ExtractionResponse, ExtractedFields, DebugInfo
from app.core.config import settings
from app.middleware.perf import get_metrics

router = APIRouter()
logger = logging.getLogger("documind")
logging.basicConfig(level=logging.INFO)

# Shared batch processor (reuses thread pool across requests)
_batch_processor = BatchProcessor(max_workers=4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_artifacts(
    doc_id: str,
    original_file_name: str,
    images: list,
    ocr_result: dict,
    extraction_result: dict,
):
    """Save debugging artifacts to disk (runs in background task)."""
    base_path = os.path.join(settings.ARTIFACTS_DIR, doc_id)
    os.makedirs(base_path, exist_ok=True)

    for idx, img in enumerate(images):
        cv2.imwrite(os.path.join(base_path, f"page_{idx}.png"), img)

    with open(os.path.join(base_path, "ocr.json"), "w") as f:
        json.dump(ocr_result, f, indent=2)

    # extraction_result may contain non-serialisable values; guard with default
    with open(os.path.join(base_path, "extraction.json"), "w") as f:
        json.dump(extraction_result, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# POST /extract — single document
# ---------------------------------------------------------------------------

@router.post(
    "/extract",
    response_model=ExtractionResponse,
    tags=["extraction"],
    summary="Extract fields from a single invoice or receipt",
    description=(
        "Accepts a PDF or image file (JPG/PNG). Runs the full OCR pipeline: "
        "ingest → preprocess → OCR → classify → field extraction → summarise. "
        "Returns structured JSON with vendor, date, amount, doc type, and "
        "confidence scores. Target latency: < 2 000 ms."
    ),
)
async def extract_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    start_time = time.time()
    doc_id = str(uuid.uuid4())

    logger.info(f"Processing document {doc_id} — {file.filename}")

    # 1. Ingest
    try:
        images = await process_upload(file)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format or corrupted file")

    if not images:
        raise HTTPException(status_code=400, detail="Could not extract images from file")

    # Limit to 5 pages for performance (food-tech docs rarely exceed this)
    images = images[:5]

    ocr_results = []
    full_text_pages = []
    total_tokens = []
    preprocessing_ms_list = []

    # 2. Process each page
    for idx, img in enumerate(images):
        clean_img, preprocess_debug = preprocess_pipeline(img)
        preprocessing_ms_list.append(preprocess_debug.get("total_preprocessing_ms", 0))

        page_result = run_ocr(clean_img, engine="easyocr")
        ocr_results.append(page_result)
        full_text_pages.append(page_result["full_text"])
        total_tokens.extend(page_result["tokens"])

    full_text = "\n".join(full_text_pages)
    avg_preprocess_ms = sum(preprocessing_ms_list) / len(preprocessing_ms_list) if preprocessing_ms_list else 0

    # 3. Classify
    classification = classify_document(full_text, total_tokens)

    # 4. Extract fields
    fields_data = extract_fields(full_text, total_tokens)

    # 5. Summarise
    summary = summarize_document(full_text)

    processing_time_ms = (time.time() - start_time) * 1000

    debug_info = DebugInfo(
        ocr_engine=ocr_results[0]["engine"] if ocr_results else "easyocr",
        pages=len(images),
        processing_time_ms=processing_time_ms,
    )

    extraction_response = {
        "doc_id": doc_id,
        "doc_type": classification["doc_type"],
        "classification_confidence": classification["confidence"],
        "fields": fields_data,
        "summary": summary,
        "debug": debug_info,
        "errors": [],
    }

    # Background: save artifacts
    background_tasks.add_task(
        save_artifacts,
        doc_id,
        file.filename,
        images,
        {"pages": ocr_results},
        extraction_response,
    )

    logger.info(
        f"Document {doc_id} processed in {processing_time_ms:.1f} ms "
        f"(preprocessing avg: {avg_preprocess_ms:.1f} ms/page)"
    )

    return extraction_response


# ---------------------------------------------------------------------------
# POST /batch — batch extraction from a ZIP archive
# ---------------------------------------------------------------------------

@router.post(
    "/batch",
    tags=["extraction"],
    summary="Batch-extract fields from a ZIP archive of documents",
    description=(
        "Accepts a ZIP file containing images (JPG/PNG). All documents are "
        "processed concurrently using an async thread pool. Returns a batch "
        "report including per-document results, aggregate throughput, and "
        "SLA compliance statistics."
    ),
)
async def batch_extract(
    file: UploadFile = File(..., description="ZIP archive of invoice/receipt images"),
):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP archives are accepted for batch processing")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        report = await _batch_processor.process_zip(content)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {e}")

    return JSONResponse(content=report)


# ---------------------------------------------------------------------------
# GET /metrics — latency and SLA statistics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    tags=["ops"],
    summary="Real-time API latency metrics",
    description=(
        "Returns p50/p95/p99 latency percentiles, average, max, and "
        "SLA violation percentage (requests exceeding 2 000 ms). "
        "Metrics are computed from the last 1 000 requests."
    ),
)
def metrics():
    return get_metrics()
