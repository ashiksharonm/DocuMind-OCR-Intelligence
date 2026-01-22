import cv2
import numpy as np
from typing import Tuple

def grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

def threshold_adaptive(img: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to get binary image."""
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

def correct_skew(img: np.ndarray) -> np.ndarray:
    """Detect and correct text skew."""
    # Convert to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    coords = np.column_stack(np.where(gray > 0)) # Assuming white text on black, wait. Invoices are black on white. 
    # Use Canny or Threshold to find text structure
    
    # Simple projection profile or Hough lines is better for docs.
    # Let's use a simpler minAreaRect approach on inverted image
    
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img
        
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def preprocess_pipeline(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Run full preprocessing pipeline."""
    debug_info = {}
    
    # ensure BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    # 1. Deskew (Optional, can be expensive or risky)
    # deskewed = correct_skew(img)
    # debug_info['deskewed'] = deskewed.shape
    
    # For now, let's stick to basic cleaning for OCR
    gray = grayscale(img)
    
    # Denoise is slow, use selectively. 
    # gray = denoise(gray) 
    
    # Binarization helpful for Tesseract, EasyOCR handles generic input well.
    # But clean binary is good for layout analysis.
    thresh = threshold_adaptive(gray)
    
    return img, {"gray": gray, "thresh": thresh}
