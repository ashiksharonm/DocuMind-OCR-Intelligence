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
from app.schemas.response_models import ExtractionResponse, ExtractedFields, DebugInfo
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger("documind")
logging.basicConfig(level=logging.INFO)

def save_artifacts(doc_id: str, original_file_name:str, images: list, ocr_result: dict, extraction_result: dict):
    """Save debugging artifacts to disk."""
    base_path = os.path.join(settings.ARTIFACTS_DIR, doc_id)
    os.makedirs(base_path, exist_ok=True)
    
    # Save processed images
    for idx, img in enumerate(images):
        cv2.imwrite(os.path.join(base_path, f"page_{idx}.png"), img)
        
    # Save JSONs
    with open(os.path.join(base_path, "ocr.json"), "w") as f:
        json.dump(ocr_result, f, indent=2)
        
    with open(os.path.join(base_path, "extraction.json"), "w") as f:
        # Convert pydantic to dict if needed, or just dump the dict
        json.dump(extraction_result, f, indent=2)

@router.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    start_time = time.time()
    doc_id = str(uuid.uuid4())
    
    logger.info(f"Processing document {doc_id} - {file.filename}")
    
    # 1. Ingest
    try:
        images = await process_upload(file)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format or corrupted file")
        
    if not images:
        raise HTTPException(status_code=400, detail="Could not extract images from file")
        
    ocr_results = []
    full_text_pages = []
    
    # 2. Process each page
    # Limit to 5 pages for demo performance
    images = images[:5] 
    
    total_tokens = []
    
    for idx, img in enumerate(images):
        # Preprocess
        clean_img, debug_imgs = preprocess_pipeline(img)
        
        # OCR
        # Use EasyOCR
        page_result = run_ocr(clean_img, engine="easyocr")
        ocr_results.append(page_result)
        
        full_text_pages.append(page_result['full_text'])
        total_tokens.extend(page_result['tokens'])
        
    full_text = "\n".join(full_text_pages)
    
    # 3. Classify
    classification = classify_document(full_text, total_tokens)
    
    # 4. Extract
    fields_data = extract_fields(full_text, total_tokens)
    
    # 5. Summarize
    summary = summarize_document(full_text)
    
    # Build Response
    debug_info = DebugInfo(
        ocr_engine="easyocr", # simplification, could vary per page
        pages=len(images),
        processing_time_ms=(time.time() - start_time) * 1000
    )
    
    extraction_response = {
        "doc_id": doc_id,
        "doc_type": classification["doc_type"],
        "classification_confidence": classification["confidence"],
        "fields": fields_data, # Pydantic will validate this against ExtractedFields
        "summary": summary,
        "debug": debug_info,
        "errors": []
    }
    
    # Background save artifacts
    background_tasks.add_task(
        save_artifacts, 
        doc_id, 
        file.filename, 
        images, 
        {"pages": ocr_results}, 
        extraction_response
    )
    
    return extraction_response
