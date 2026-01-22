import os
from typing import List, Union
import numpy as np
import cv2
from pdf2image import convert_from_path, convert_from_bytes
from fastapi import UploadFile

def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load image from bytes into numpy array (OpenCV format)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_pdf_from_bytes(file_bytes: bytes) -> List[np.ndarray]:
    """Convert PDF bytes to list of numpy images."""
    try:
        pages = convert_from_bytes(file_bytes)
        images = []
        for page in pages:
            # Convert PIL image to numpy array (RGB)
            img = np.array(page)
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return []

async def process_upload(file: UploadFile) -> List[np.ndarray]:
    """Router to handle both PDF and Image uploads."""
    content = await file.read()
    
    if file.content_type == "application/pdf":
        return load_pdf_from_bytes(content)
    elif file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
        img = load_image_from_bytes(content)
        if img is not None:
            return [img]
        return []
    else:
        # Fallback based on extension if content-type is missing/wrong
        if file.filename.lower().endswith(".pdf"):
             return load_pdf_from_bytes(content)
        else:
             img = load_image_from_bytes(content)
             return [img] if img is not None else []
