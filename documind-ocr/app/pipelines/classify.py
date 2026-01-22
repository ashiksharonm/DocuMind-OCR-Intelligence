from typing import List, Dict, Any

def classify_document(text: str, tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Classify document type based on keywords and layout.
    Types: 'invoice', 'receipt', 'unknown'
    """
    text_lower = text.lower()
    
    # Simple Keyword score
    invoice_keywords = ['invoice', 'bill to', 'due date', 'invoice number', 'inv#']
    receipt_keywords = ['receipt', 'ticket', 'change due', 'cash', 'terminal', 'pos']
    
    invoice_score = sum(1 for k in invoice_keywords if k in text_lower)
    receipt_score = sum(1 for k in receipt_keywords if k in text_lower)
    
    # Layout heuristics (optional)
    # Receipts are often narrow (width/height ratio) - need image dims for that.
    # For now, text based is enough.
    
    doc_type = "unknown"
    confidence = 0.0
    
    if invoice_score > receipt_score:
        doc_type = "invoice"
        confidence = min(0.5 + (invoice_score * 0.1), 0.99)
    elif receipt_score > invoice_score:
        doc_type = "receipt"
        confidence = min(0.5 + (receipt_score * 0.1), 0.99)
    else:
        # Default fallback
        if invoice_score > 0:
            doc_type = "invoice"
            confidence = 0.5
        elif receipt_score > 0:
            doc_type = "receipt"
            confidence = 0.5
            
    return {
        "doc_type": doc_type,
        "confidence": confidence
    }
