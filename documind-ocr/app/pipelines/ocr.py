import easyocr
import pytesseract
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader once to avoid reloading model
# NOTE: In production with multiple workers, this might consume memory per worker.
reader = easyocr.Reader(['en'], gpu=False) # GPU=False for general compatibility

class OCRResult:
    def __init__(self, text: str, confidence: float, bbox: List[Tuple[int, int]]):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox
        }

def run_easyocr(img: np.ndarray) -> List[OCRResult]:
    """Run EasyOCR on image."""
    try:
        # EasyOCR returns list of (bbox, text, prob)
        results = reader.readtext(img)
        outputs = []
        for bbox, text, prob in results:
            # bbox is list of 4 points [[x,y], [x,y], [x,y], [x,y]]
            # Convert to standard python ints
            clean_bbox = [(int(p[0]), int(p[1])) for p in bbox]
            outputs.append(OCRResult(text, float(prob), clean_bbox))
        return outputs
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        return []

def run_tesseract(img: np.ndarray) -> List[OCRResult]:
    """Run Tesseract on image (fallback)."""
    try:
        # Output dict: 'text', 'conf', 'left', 'top', 'width', 'height'
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        outputs = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0: # Filter empty results
                text = data['text'][i].strip()
                if not text:
                    continue
                
                conf = float(data['conf'][i]) / 100.0
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                bbox = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                outputs.append(OCRResult(text, conf, bbox))
        return outputs
    except Exception as e:
        logger.error(f"Tesseract failed: {e}")
        return []

def run_ocr(img: np.ndarray, engine: str = "easyocr") -> Dict[str, Any]:
    """Main OCR entry point."""
    if engine == "easyocr":
        results = run_easyocr(img)
        if not results:
             logger.warning("EasyOCR returned empty, falling back to Tesseract")
             results = run_tesseract(img)
             engine = "tesseract_fallback"
    else:
        results = run_tesseract(img)
        
    # Sort results by vertical position (Y), then horizontal (X) for reading order
    # Heuristic: Sort by Top-Left Y, then Top-Left X. 
    # To handle slight skew, bin Y values? For now simple sort.
    results.sort(key=lambda r: (r.bbox[0][1], r.bbox[0][0]))
    
    full_text = " ".join([r.text for r in results])
    
    return {
        "engine": engine,
        "tokens": [r.to_dict() for r in results],
        "full_text": full_text
    }
