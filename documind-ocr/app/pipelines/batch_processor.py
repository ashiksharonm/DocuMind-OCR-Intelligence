"""
batch_processor.py — Scalable Async Batch Processing Pipeline
=============================================================

Processes large unstructured datasets of images and videos concurrently.
Uses ``asyncio`` for I/O and a ``ThreadPoolExecutor`` for CPU-bound OCR
workloads, enabling sub-2-second per-document throughput at scale.

Supports
--------
* Image files  (.jpg, .jpeg, .png, .bmp, .tiff)
* Video files  (.mp4, .avi, .mov, .mkv) — key-frame extraction via OpenCV
* ZIP archives — extracted automatically before processing

Usage
-----
>>> from app.pipelines.batch_processor import BatchProcessor
>>> processor = BatchProcessor(max_workers=4)
>>> report = await processor.process_directory("dataset/sample_docs/")
>>> print(report["throughput_docs_per_sec"])
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import uuid
import zipfile
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

from app.pipelines.ingest import load_image_from_bytes
from app.pipelines.preprocess import preprocess_pipeline
from app.pipelines.ocr import run_ocr
from app.pipelines.extract_fields import extract_fields
from app.pipelines.classify import classify_document

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


# ---------------------------------------------------------------------------
# Frame extraction from video
# ---------------------------------------------------------------------------

def extract_key_frames(
    video_path: str,
    max_frames: int = 10,
    frame_interval: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract representative frames from a video file.

    Uses uniform sampling at ``frame_interval`` steps (or auto-calculated from
    total frame count and ``max_frames``).  This converts video datasets into
    per-frame image datasets suitable for OCR.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    max_frames : int
        Maximum number of frames to extract.
    frame_interval : int, optional
        Sample every Nth frame.  Auto-calculated if None.

    Returns
    -------
    List[np.ndarray]
        List of BGR numpy arrays (OpenCV format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    if frame_interval is None:
        frame_interval = max(1, total_frames // max_frames)

    frames: List[np.ndarray] = []
    frame_idx = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += frame_interval
        if frame_idx >= total_frames:
            break

    cap.release()
    logger.info(f"Extracted {len(frames)} key frames from {video_path}")
    return frames


# ---------------------------------------------------------------------------
# Single-document processing (runs in thread pool)
# ---------------------------------------------------------------------------

def _process_single_image(img: np.ndarray, source_path: str) -> Dict[str, Any]:
    """
    Run the full OCR pipeline on a single image (thread-safe, blocking).

    Returns a result dict suitable for aggregation.
    """
    doc_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    try:
        clean_img, preprocess_debug = preprocess_pipeline(img)
        ocr_result = run_ocr(clean_img, engine="easyocr")
        full_text = ocr_result["full_text"]
        tokens = ocr_result["tokens"]
        fields = extract_fields(full_text, tokens)
        classification = classify_document(full_text, tokens)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "doc_id": doc_id,
            "source_path": source_path,
            "doc_type": classification["doc_type"],
            "classification_confidence": classification["confidence"],
            "fields": fields,
            "ocr_engine": ocr_result["engine"],
            "preprocessing_ms": preprocess_debug.get("total_preprocessing_ms", 0),
            "overhead_reduction_pct": preprocess_debug.get("overhead_reduction_pct", 0),
            "total_ms": round(elapsed_ms, 2),
            "success": True,
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.error(f"Failed to process {source_path}: {exc}")
        return {
            "doc_id": doc_id,
            "source_path": source_path,
            "success": False,
            "error": str(exc),
            "total_ms": round(elapsed_ms, 2),
        }


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """
    Scalable asynchronous batch document processor.

    Architecture
    ------------
    * A ``ThreadPoolExecutor`` handles CPU-bound OCR workloads without blocking
      the asyncio event loop.
    * ``asyncio.gather`` submits all tasks concurrently, capped by
      ``max_workers`` threads.
    * Throughput is measured end-to-end and reported in the summary.

    Parameters
    ----------
    max_workers : int
        Number of parallel OCR threads.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def _run_in_thread(self, img: np.ndarray, source_path: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            _process_single_image,
            img,
            source_path,
        )

    async def process_images(
        self,
        images: List[np.ndarray],
        source_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a list of in-memory images concurrently.

        Parameters
        ----------
        images : list of np.ndarray
            Images to process.
        source_paths : list of str, optional
            Corresponding source file paths for reporting.

        Returns
        -------
        dict
            Batch report with per-document results and aggregate metrics.
        """
        if source_paths is None:
            source_paths = [f"image_{i}" for i in range(len(images))]

        batch_start = time.perf_counter()

        tasks = [
            self._run_in_thread(img, path)
            for img, path in zip(images, source_paths)
        ]
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - batch_start
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        latencies = [r["total_ms"] for r in successful]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        below_2s = sum(1 for ms in latencies if ms < 2000)

        return {
            "total_documents": len(images),
            "successful": len(successful),
            "failed": len(failed),
            "total_elapsed_sec": round(elapsed, 3),
            "throughput_docs_per_sec": round(len(images) / elapsed, 2) if elapsed > 0 else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "sub_2s_transactions_pct": round(below_2s / len(images) * 100, 1) if images else 0,
            "results": results,
        }

    async def process_directory(
        self,
        directory: str,
        max_video_frames: int = 5,
    ) -> Dict[str, Any]:
        """
        Discover and process all image and video files in a directory.

        Parameters
        ----------
        directory : str
            Path to the dataset directory.
        max_video_frames : int
            Maximum frames to extract from each video file.

        Returns
        -------
        dict
            Batch report (same structure as ``process_images``).
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Dataset directory not found: {directory}")

        images: List[np.ndarray] = []
        paths: List[str] = []

        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                fpath = os.path.join(root, fname)

                if ext in IMAGE_EXTENSIONS:
                    img = cv2.imread(fpath)
                    if img is not None:
                        images.append(img)
                        paths.append(fpath)

                elif ext in VIDEO_EXTENSIONS:
                    frames = extract_key_frames(fpath, max_frames=max_video_frames)
                    for idx, frame in enumerate(frames):
                        images.append(frame)
                        paths.append(f"{fpath}::frame_{idx}")

        if not images:
            return {
                "total_documents": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "error": "No processable files found in directory",
            }

        logger.info(f"Batch: discovered {len(images)} items in {directory}")
        return await self.process_images(images, paths)

    async def process_zip(self, zip_bytes: bytes) -> Dict[str, Any]:
        """
        Extract a ZIP archive of documents and process all contents.

        Parameters
        ----------
        zip_bytes : bytes
            Raw bytes of the ZIP file (e.g., from an HTTP upload).

        Returns
        -------
        dict
            Batch report.
        """
        images: List[np.ndarray] = []
        paths: List[str] = []

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in sorted(zf.namelist()):
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                try:
                    data = zf.read(name)
                    img = load_image_from_bytes(data)
                    if img is not None:
                        images.append(img)
                        paths.append(name)
                except Exception as e:
                    logger.warning(f"Skipping {name} in ZIP: {e}")

        if not images:
            return {
                "total_documents": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "error": "No valid images found in ZIP archive",
            }

        logger.info(f"Batch ZIP: processing {len(images)} images")
        return await self.process_images(images, paths)

    def shutdown(self):
        """Gracefully shut down the thread pool."""
        self._executor.shutdown(wait=True)
