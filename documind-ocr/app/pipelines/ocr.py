"""OCR front end. Tesseract is the default; PaddleOCR and EasyOCR are optional, lazily loaded."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

BBox = List[Tuple[int, int]]
_readers: Dict[str, Any] = {}


class OCRResult:
    def __init__(self, text: str, confidence: float, bbox: BBox):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "bbox": self.bbox}


def _rect(x: int, y: int, w: int, h: int) -> BBox:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def run_tesseract(img: np.ndarray) -> List[OCRResult]:
    """Word-level Tesseract output (confidence scaled to 0..1)."""
    import pytesseract

    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:  # missing binary, corrupt image
        logger.error("Tesseract failed: %s", e)
        return []
    outputs = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        conf = float(data["conf"][i])
        if text and conf >= 0:
            x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
            outputs.append(OCRResult(text, conf / 100.0, _rect(x, y, w, h)))
    return outputs


def run_easyocr(img: np.ndarray) -> List[OCRResult]:
    try:
        if "easyocr" not in _readers:
            import easyocr

            _readers["easyocr"] = easyocr.Reader(["en"], gpu=False)
        results = _readers["easyocr"].readtext(img)
        return [OCRResult(t, float(p), [(int(pt[0]), int(pt[1])) for pt in b]) for b, t, p in results]
    except Exception as e:
        logger.error("EasyOCR failed: %s", e)
        return []


def run_paddleocr(img: np.ndarray) -> List[OCRResult]:
    try:
        if "paddleocr" not in _readers:
            from paddleocr import PaddleOCR

            _readers["paddleocr"] = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        out = _readers["paddleocr"].ocr(img, cls=False) or [[]]
        return [
            OCRResult(t, float(c), [(int(p[0]), int(p[1])) for p in box]) for box, (t, c) in (out[0] or [])
        ]
    except Exception as e:
        logger.error("PaddleOCR failed: %s", e)
        return []


ENGINES = {"tesseract": run_tesseract, "easyocr": run_easyocr, "paddleocr": run_paddleocr}


def group_lines(results: List[OCRResult]) -> List[List[OCRResult]]:
    """Cluster tokens into reading-order lines by vertical overlap."""
    if not results:
        return []
    heights = sorted(abs(r.bbox[2][1] - r.bbox[0][1]) or 1 for r in results)
    tol = 0.6 * heights[len(heights) // 2]
    lines: List[List[OCRResult]] = []
    for r in sorted(results, key=lambda r: (r.bbox[0][1], r.bbox[0][0])):
        cy = (r.bbox[0][1] + r.bbox[2][1]) / 2
        for line in lines:
            ly = sum((t.bbox[0][1] + t.bbox[2][1]) / 2 for t in line) / len(line)
            if abs(cy - ly) <= tol:
                line.append(r)
                break
        else:
            lines.append([r])
    lines.sort(key=lambda ln: min(t.bbox[0][1] for t in ln))
    return [sorted(ln, key=lambda t: t.bbox[0][0]) for ln in lines]


def run_ocr(img: np.ndarray, engine: Optional[str] = None) -> Dict[str, Any]:
    """Run OCR and return ``{engine, tokens, full_text}``; ``full_text`` has one text line per row."""
    engine = engine or settings.OCR_ENGINE
    if engine not in ENGINES:
        raise ValueError(f"unknown OCR engine {engine!r}; choose from {sorted(ENGINES)}")
    results = ENGINES[engine](img)
    if not results and engine != "tesseract":
        logger.warning("%s returned nothing, falling back to Tesseract", engine)
        results = run_tesseract(img)
        engine = "tesseract_fallback"
    lines = group_lines(results)
    ordered = [t for ln in lines for t in ln]
    return {
        "engine": engine,
        "tokens": [t.to_dict() for t in ordered],
        "full_text": "\n".join(" ".join(t.text for t in ln) for ln in lines),
    }
