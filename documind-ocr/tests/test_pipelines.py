"""Tests for core pipelines — preprocessing shapes, classification, and extraction."""

import pytest
import numpy as np
from app.pipelines.preprocess import grayscale, threshold_adaptive
from app.pipelines.classify import classify_document
from app.pipelines.extract_fields import extract_fields, extract_amount, extract_date


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_shapes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    gray = grayscale(img)
    assert len(gray.shape) == 2
    assert gray.shape == (100, 100)

    thresh = threshold_adaptive(gray)
    assert thresh.shape == (100, 100)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classification_invoice():
    text = "INVOICE #12345 Due Date: 2023-01-01 Bill To: Acme Corp Total: $500"
    res = classify_document(text, [])
    assert res["doc_type"] == "invoice"
    assert res["confidence"] > 0.5


def test_classification_receipt():
    text = "WALMART RECEIPT Total: $50.00 Change Due: $2.00 Cash Terminal POS"
    res = classify_document(text, [])
    assert res["doc_type"] == "receipt"
    assert res["confidence"] > 0.5


def test_classification_unknown():
    text = "Hello world, this is a random document with no keywords."
    res = classify_document(text, [])
    assert res["doc_type"] == "unknown"
    assert res["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def test_extract_amount_basic():
    text = "Subtotal: $80.00\nTax: $8.00\nTotal: $88.00"
    result = extract_amount(text, [])
    assert result["value"] is not None
    assert result["confidence"] > 0.5
    # Should pick the grand total, not subtotal
    assert abs(float(result["value"]) - 88.0) < 1.0


def test_extract_date_iso():
    text = "Invoice Date: 2024-03-15\nDue: 2024-04-15"
    result = extract_date(text, [])
    assert result["value"] is not None
    assert "2024" in result["value"]


def test_extract_fields_returns_all_keys():
    text = "ACME Corp\nInvoice Date: 2024-01-20\nTotal: $1,200.00"
    result = extract_fields(text, [])
    assert "vendor_name" in result
    assert "invoice_date" in result
    assert "total_amount" in result


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------

def test_sanity():
    assert 1 == 1
