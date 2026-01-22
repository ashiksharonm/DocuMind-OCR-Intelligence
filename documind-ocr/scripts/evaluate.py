import os
import csv
import json
import asyncio
import pandas as pd
from typing import List, Dict
from datetime import datetime
import tabulate

# Determine project root and add to path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_context := os.path.join(project_root, 'documind-ocr'))

from app.pipelines.ingest import load_image_from_bytes, load_pdf_from_bytes
from app.pipelines.preprocess import preprocess_pipeline
from app.pipelines.ocr import run_ocr
from app.pipelines.extract_fields import extract_fields
from app.pipelines.classify import classify_document

DATASET_DIR = os.path.join(project_root, 'documind-ocr/dataset')
LABELS_PATH = os.path.join(DATASET_DIR, 'labels.csv')
REPORTS_DIR = os.path.join(project_root, 'documind-ocr/reports')

async def process_file(file_path: str):
    """Run full pipeline on a file."""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            images = load_pdf_from_bytes(f.read())
    else:
        with open(file_path, 'rb') as f:
            images = [load_image_from_bytes(f.read())]
            
    if not images:
        return None
        
    # Process first page only for eval simplicity or all?
    # Usually invoices have key info on page 1.
    img = images[0]
    clean_img, _ = preprocess_pipeline(img)
    ocr_res = run_ocr(clean_img)
    
    full_text = ocr_res['full_text']
    tokens = ocr_res['tokens']
    
    fields = extract_fields(full_text, tokens)
    doc_type = classify_document(full_text, tokens)
    
    return {
        "vendor": fields['vendor_name']['value'],
        "date": fields['invoice_date']['value'],
        "amount": fields['total_amount']['value'],
        "currency": fields['total_amount']['currency'],
        "doc_type": doc_type['doc_type']
    }

def evaluate():
    if not os.path.exists(LABELS_PATH):
        print("No labels.csv found.")
        return

    df = pd.read_csv(LABELS_PATH)
    results = []
    
    print(f"Starting evaluation on {len(df)} documents...")
    
    correct_vendor = 0
    correct_date = 0
    correct_amount = 0
    total_docs = 0
    
    for _, row in df.iterrows():
        doc_id = row['doc_id']
        filename = row['file_path']
        path = os.path.join(DATASET_DIR, 'sample_docs', filename)
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        total_docs += 1
        
        try:
            # We need to run async function in sync loop
            pred = asyncio.run(process_file(path))
            if not pred:
                continue
                
            # Compare
            # Vendor: fuzzy match or exact? Let's do simple cleaning
            gt_vendor = str(row['vendor_name']).strip().lower()
            pred_vendor = str(pred['vendor']).strip().lower() if pred['vendor'] else ""
            
            # Simple contains check or exact
            vendor_match = gt_vendor in pred_vendor or pred_vendor in gt_vendor
            if vendor_match: correct_vendor += 1
            
            # Date: match extracted YYYY-MM-DD
            gt_date = str(row['invoice_date']).strip()
            pred_date = str(pred['date']).strip() if pred['date'] else ""
            date_match = gt_date == pred_date
            if date_match: correct_date += 1
            
            # Amount: Text match or float match with tolerance
            gt_amount = float(row['total_amount'])
            pred_amount = float(pred['amount']) if pred['amount'] else 0.0
            
            # 1% tolerance
            amount_match = False
            if gt_amount > 0:
                diff = abs(gt_amount - pred_amount) / gt_amount
                amount_match = diff < 0.01
            elif pred_amount == 0:
                amount_match = True
                
            if amount_match: correct_amount += 1
            
            results.append({
                "doc_id": doc_id,
                "vendor_match": vendor_match,
                "date_match": date_match,
                "amount_match": amount_match,
                "gt_amount": gt_amount,
                "pred_amount": pred_amount
            })
            
        except Exception as e:
            print(f"Error processing {doc_id}: {e}")
            
    # Report
    if total_docs == 0:
        print("No documents processed.")
        return

    acc_vendor = correct_vendor / total_docs
    acc_date = correct_date / total_docs
    acc_amount = correct_amount / total_docs
    
    print("\n=== Evaluation Report ===")
    print(f"Total Documents: {total_docs}")
    print(f"Vendor Accuracy: {acc_vendor:.2%}")
    print(f"Date Accuracy:   {acc_date:.2%}")
    print(f"Amount Accuracy: {acc_amount:.2%}")
    
    # Save Report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "metrics": {
                "vendor_acc": acc_vendor,
                "date_acc": acc_date,
                "amount_acc": acc_amount
            },
            "details": results
        }, f, indent=2)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()
