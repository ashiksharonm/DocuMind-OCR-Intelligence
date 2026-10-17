"""Document -> routed field results. Shared by the worker, the Space demo and the evaluation."""

import os
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.pipelines.classify import classify_document
from app.pipelines.extract_fields import extract_fields
from app.pipelines.ingest import load_image_from_bytes, load_pdf_from_bytes
from app.pipelines.ocr import run_ocr
from app.pipelines.preprocess import preprocess_pipeline
from app.review.confidence import score_fields

OcrFn = Callable[[np.ndarray], Dict[str, Any]]
MAX_PAGES = 3


def load_pages(data: bytes) -> List[np.ndarray]:
    """Decode an upload (PDF or image) into BGR pages; an empty list means undecodable."""
    if data[:5] == b"%PDF-":
        return load_pdf_from_bytes(data)[:MAX_PAGES]
    img = load_image_from_bytes(data)
    return [] if img is None else [img]


def process_pages(pages: List[np.ndarray], ocr: Optional[OcrFn] = None) -> Dict[str, Any]:
    """Run preprocess -> OCR -> classify -> extract -> confidence on already-decoded pages."""
    ocr = ocr or run_ocr
    texts, tokens, engine = [], [], ""
    for img in pages:
        clean, _ = preprocess_pipeline(img)
        res = ocr(clean)
        engine = res["engine"]
        texts.append(res["full_text"])
        tokens.extend(res["tokens"])
    text = "\n".join(texts)
    return {
        "engine": engine,
        "ocr_text": text,
        "doc_type": classify_document(text, tokens)["doc_type"],
        "fields": score_fields(extract_fields(text, tokens), tokens),
    }


def process_bytes(data: bytes, ocr: Optional[OcrFn] = None) -> Dict[str, Any]:
    pages = load_pages(data)
    if not pages:
        raise ValueError("could not decode file as image or PDF")
    return process_pages(pages, ocr)


def write_preview(page: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, page)
    return path
