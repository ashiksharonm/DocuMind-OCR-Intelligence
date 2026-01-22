import pytest
import numpy as np
from app.pipelines.preprocess import grayscale, threshold_adaptive
from app.pipelines.classify import classify_document

def test_preprocess_shapes():
    # Create dummy image 100x100 RGB
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    gray = grayscale(img)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)
    
    thresh = threshold_adaptive(gray)
    assert thresh.shape == (100, 100)

def test_classification():
    text_invoice = "INVOICE #12345 Due Date: 2023-01-01 Total: $500"
    res = classify_document(text_invoice, [])
    assert res['doc_type'] == 'invoice'
    
    text_receipt = "WALMART RECEIPT Total: $50.00 Change Due: $2.00"
    res2 = classify_document(text_receipt, [])
    assert res2['doc_type'] == 'receipt'

def test_sanity():
    assert 1 == 1
