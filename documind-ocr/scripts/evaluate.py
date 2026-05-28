"""
evaluate.py — Evaluation & Accuracy Benchmarking Script
=========================================================

Runs the full DocuMind OCR pipeline against labelled ground-truth data and
reports field-level accuracy, precision, recall, and F1 scores.

Targets
-------
* Overall field extraction accuracy  ≥ 95 %
* Vendor extraction F1               ≥ 0.90
* Date extraction F1                 ≥ 0.93
* Amount extraction F1               ≥ 0.95

Run
---
    python scripts/evaluate.py

Output
------
    documind-ocr/reports/eval_report.json
"""

import os
import sys
import csv
import json
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd

# Make the app package importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from app.pipelines.ingest import load_image_from_bytes, load_pdf_from_bytes
from app.pipelines.preprocess import preprocess_pipeline
from app.pipelines.ocr import run_ocr
from app.pipelines.extract_fields import extract_fields
from app.pipelines.classify import classify_document

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _str_match(pred: Optional[str], gt: Optional[str]) -> bool:
    """Bidirectional contains match (case-insensitive)."""
    if not pred or not gt:
        return False
    pred_l, gt_l = pred.strip().lower(), gt.strip().lower()
    return gt_l in pred_l or pred_l in gt_l


def _amount_match(pred: Optional[float], gt: Optional[float], tol: float = 0.01) -> bool:
    """Float match within a relative tolerance (default 1%)."""
    if pred is None or gt is None:
        return pred == gt
    if gt == 0:
        return pred == 0
    return abs(gt - pred) / abs(gt) < tol


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, and F1 from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

async def process_file(file_path: str) -> Optional[Dict]:
    """Run the full pipeline on a single file and return extracted fields."""
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            images = load_pdf_from_bytes(f.read())
    else:
        with open(file_path, "rb") as f:
            images = [load_image_from_bytes(f.read())]

    if not images or images[0] is None:
        return None

    img = images[0]
    clean_img, preprocess_debug = preprocess_pipeline(img)
    ocr_res = run_ocr(clean_img)

    full_text = ocr_res["full_text"]
    tokens = ocr_res["tokens"]

    fields = extract_fields(full_text, tokens)
    doc_type_result = classify_document(full_text, tokens)

    return {
        "vendor": fields["vendor_name"]["value"],
        "vendor_conf": fields["vendor_name"]["confidence"],
        "date": fields["invoice_date"]["value"],
        "date_conf": fields["invoice_date"]["confidence"],
        "amount": fields["total_amount"]["value"],
        "amount_conf": fields["total_amount"]["confidence"],
        "doc_type": doc_type_result["doc_type"],
        "preprocessing_ms": preprocess_debug.get("total_preprocessing_ms", 0),
        "overhead_reduction_pct": preprocess_debug.get("overhead_reduction_pct", 0),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate():
    if not os.path.exists(LABELS_PATH):
        print(f"No labels.csv found at {LABELS_PATH}.")
        print("Please add ground-truth labels before running evaluation.")
        return

    df = pd.read_csv(LABELS_PATH)
    results = []

    print(f"\n{'='*60}")
    print("  DocuMind OCR — Evaluation Report")
    print(f"{'='*60}")
    print(f"  Dataset      : {LABELS_PATH}")
    print(f"  Documents    : {len(df)}")
    print(f"  Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # TP / FP / FN counters per field
    vendor_tp = vendor_fp = vendor_fn = 0
    date_tp = date_fp = date_fn = 0
    amount_tp = amount_fp = amount_fn = 0
    doc_type_correct = 0

    total_docs = 0
    preprocessing_ms_list = []

    for _, row in df.iterrows():
        filename = str(row.get("file_path", ""))
        path = os.path.join(DATASET_DIR, "sample_docs", filename)

        if not os.path.exists(path):
            print(f"  [SKIP] File not found: {path}")
            continue

        total_docs += 1

        try:
            pred = asyncio.run(process_file(path))
            if not pred:
                print(f"  [FAIL] Pipeline returned None for: {filename}")
                continue

            preprocessing_ms_list.append(pred.get("preprocessing_ms", 0))

            # ── Vendor ───────────────────────────────────────────────────────
            gt_vendor = str(row.get("vendor_name", "")).strip()
            pred_vendor = str(pred["vendor"]).strip() if pred["vendor"] else ""
            vendor_match = _str_match(pred_vendor, gt_vendor) if gt_vendor else False
            if gt_vendor:
                if vendor_match:
                    vendor_tp += 1
                else:
                    vendor_fn += 1
                    if pred_vendor:
                        vendor_fp += 1
            elif pred_vendor:
                vendor_fp += 1

            # ── Date ─────────────────────────────────────────────────────────
            gt_date = str(row.get("invoice_date", "")).strip()
            pred_date = str(pred["date"]).strip() if pred["date"] else ""
            date_match = gt_date == pred_date if gt_date else False
            if gt_date:
                if date_match:
                    date_tp += 1
                else:
                    date_fn += 1
                    if pred_date:
                        date_fp += 1
            elif pred_date:
                date_fp += 1

            # ── Amount ───────────────────────────────────────────────────────
            try:
                gt_amount = float(row.get("total_amount", "nan"))
                pred_amount = float(pred["amount"]) if pred["amount"] is not None else None
                amount_match = _amount_match(pred_amount, gt_amount)
            except (ValueError, TypeError):
                gt_amount, pred_amount, amount_match = None, None, False

            if gt_amount is not None:
                if amount_match:
                    amount_tp += 1
                else:
                    amount_fn += 1
                    if pred_amount is not None:
                        amount_fp += 1
            elif pred_amount is not None:
                amount_fp += 1

            # ── Doc type ─────────────────────────────────────────────────────
            if "doc_type" in row and pred["doc_type"] == str(row["doc_type"]).strip().lower():
                doc_type_correct += 1

            result_entry = {
                "file": filename,
                "vendor_match": vendor_match,
                "date_match": date_match,
                "amount_match": amount_match,
                "pred_vendor": pred_vendor,
                "pred_date": pred_date,
                "pred_amount": pred_amount,
                "gt_vendor": gt_vendor,
                "gt_date": gt_date,
                "gt_amount": gt_amount,
                "preprocessing_ms": pred.get("preprocessing_ms", 0),
                "overhead_reduction_pct": pred.get("overhead_reduction_pct", 0),
            }
            results.append(result_entry)

            status = "✅" if (vendor_match and date_match and amount_match) else "⚠️ "
            print(
                f"  {status} {filename[:40]:<40} "
                f"vendor={int(vendor_match)} date={int(date_match)} amount={int(amount_match)}"
            )

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    if total_docs == 0:
        print("\n  No documents processed. Check your dataset/labels.csv and sample_docs/.")
        return

    # ── Metrics ──────────────────────────────────────────────────────────────
    vendor_metrics = precision_recall_f1(vendor_tp, vendor_fp, vendor_fn)
    date_metrics = precision_recall_f1(date_tp, date_fp, date_fn)
    amount_metrics = precision_recall_f1(amount_tp, amount_fp, amount_fn)

    overall_acc = (vendor_tp + date_tp + amount_tp) / (total_docs * 3) if total_docs > 0 else 0
    doc_type_acc = doc_type_correct / total_docs if total_docs > 0 else 0

    avg_preprocess_ms = (
        sum(preprocessing_ms_list) / len(preprocessing_ms_list)
        if preprocessing_ms_list else 0
    )

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(f"  Total documents processed : {total_docs}")
    print(f"\n  Field-level metrics:")
    print(f"    Vendor  — P={vendor_metrics['precision']:.3f}  R={vendor_metrics['recall']:.3f}  F1={vendor_metrics['f1']:.3f}")
    print(f"    Date    — P={date_metrics['precision']:.3f}  R={date_metrics['recall']:.3f}  F1={date_metrics['f1']:.3f}")
    print(f"    Amount  — P={amount_metrics['precision']:.3f}  R={amount_metrics['recall']:.3f}  F1={amount_metrics['f1']:.3f}")
    print(f"\n  Overall extraction accuracy : {overall_acc:.2%}")
    print(f"  Document classification acc : {doc_type_acc:.2%}")
    print(f"  Avg preprocessing time      : {avg_preprocess_ms:.1f} ms/doc")

    passed = overall_acc >= 0.95
    print(f"\n  {'✅ TARGET MET' if passed else '⚠️  BELOW TARGET'} — 95%+ accuracy goal: {overall_acc:.2%}")

    # ── Save report ──────────────────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": total_docs,
        "metrics": {
            "overall_accuracy": round(overall_acc, 4),
            "doc_type_accuracy": round(doc_type_acc, 4),
            "vendor": vendor_metrics,
            "date": date_metrics,
            "amount": amount_metrics,
        },
        "preprocessing": {
            "avg_ms_per_doc": round(avg_preprocess_ms, 2),
        },
        "target_met_95pct": passed,
        "details": results,
    }

    report_path = os.path.join(REPORTS_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved → {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    evaluate()
