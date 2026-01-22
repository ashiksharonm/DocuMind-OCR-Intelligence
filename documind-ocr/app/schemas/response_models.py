from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class FieldValue(BaseModel):
    value: Optional[str | float] = None
    currency: Optional[str] = None # Only for amount
    confidence: float = 0.0

class ExtractedFields(BaseModel):
    vendor_name: FieldValue
    invoice_date: FieldValue
    total_amount: FieldValue

class DebugInfo(BaseModel):
    ocr_engine: str
    pages: int
    processing_time_ms: float

class ExtractionResponse(BaseModel):
    doc_id: str
    doc_type: str
    classification_confidence: float
    fields: ExtractedFields
    summary: List[str]
    debug: DebugInfo
    errors: List[str] = []
