"""Gradio demo: upload a document, see per-field confidence, review flagged fields, export labels.

Runs the same pipeline as the API (Tesseract, CPU) and the same SQLite store, synchronously
in-process. Data lives in a temp file and disappears when the Space restarts.
"""

import hashlib
import json
import os
import sys
import tempfile
import uuid

import cv2
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "app")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.review.processing import load_pages, process_pages, write_preview  # noqa: E402
from app.review.store import FIELD_NAMES, Store  # noqa: E402
from app.review.synthetic import generate  # noqa: E402

THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.75"))
WORK = tempfile.mkdtemp(prefix="documind_")
store = Store(os.path.join(WORK, "demo.db"))


def _rows(job):
    return [[n, f["predicted"], f["confidence"], f["status"]] for n, f in job["fields"].items()]


def extract(path):
    if path is None:
        raise gr.Error("Upload an image or PDF first")
    with open(path, "rb") as fh:
        data = fh.read()
    pages = load_pages(data)
    if not pages:
        raise gr.Error("Could not read that file as an image or PDF")
    jid = uuid.uuid4().hex
    store.create_job(os.path.basename(path), "", hashlib.sha256(data).hexdigest(), path, THRESHOLD, jid)
    store.claim(jid)
    res = process_pages(pages)
    preview = write_preview(pages[0], os.path.join(WORK, f"{jid}.png"))
    store.complete(jid, res["doc_type"], res["engine"], res["ocr_text"], res["fields"], THRESHOLD, preview)
    job = store.get_job(jid)
    msg = (
        f"Job `{jid[:8]}`: **{job['status']}** (document type: {job['doc_type']}). "
        f"Fields below {THRESHOLD} confidence are sent to the review queue."
    )
    return preview, _rows(job), msg


def sample():
    label, level, img = generate(5, int.from_bytes(os.urandom(2), "big"))[4]  # the 'severe' level
    path = os.path.join(WORK, f"sample_{uuid.uuid4().hex[:6]}.png")
    cv2.imwrite(path, img)
    return path


def queue():
    items = store.review_queue(50)
    return [[i["job_id"], i["name"], i["predicted"], i["confidence"]] for i in items]


def decide(job_id, field, action, value, reviewer):
    if field not in FIELD_NAMES:
        raise gr.Error(f"field must be one of {list(FIELD_NAMES)}")
    if action == "correct" and not (value or "").strip():
        raise gr.Error("Enter the corrected value")
    job = store.review_field(job_id.strip(), field, reviewer or "demo", (value or "").strip() or None, action)
    if job is None:
        raise gr.Error("Unknown job id or field")
    return queue(), f"Saved. Job status: {job['status']}"


def export():
    path = os.path.join(WORK, "labels.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for rec in store.export_dataset("reviewed"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def build():
    with gr.Blocks(title="DocuMind review") as demo:
        gr.Markdown(
            "# DocuMind: confidence-routed human review\nUpload an invoice or receipt. Tesseract reads it, "
            "each field gets a heuristic confidence, and uncertain fields go to a review queue. "
            "Corrections are exported as a labelled dataset. Confidence is a ranking signal, not a "
            "calibrated probability; no accuracy claim is made for your documents."
        )
        with gr.Tab("Extract"):
            up = gr.File(label="Invoice / receipt (image or PDF)", type="filepath")
            with gr.Row():
                gr.Button("Use a degraded synthetic sample").click(sample, None, up)
                go = gr.Button("Extract", variant="primary")
            img, table, msg = (
                gr.Image(label="Page 1"),
                gr.Dataframe(headers=["field", "value", "confidence", "status"], label="Fields"),
                gr.Markdown(),
            )
            go.click(extract, up, [img, table, msg])
        with gr.Tab("Review queue"):
            q = gr.Dataframe(headers=["job_id", "field", "predicted", "confidence"], label="Pending")
            gr.Button("Refresh").click(queue, None, q)
            with gr.Row():
                jid, fld = gr.Textbox(label="job_id"), gr.Dropdown(list(FIELD_NAMES), label="field")
                val, who = (
                    gr.Textbox(label="Corrected value (YYYY-MM-DD for dates)"),
                    gr.Textbox(label="Reviewer"),
                )
            note = gr.Markdown()
            with gr.Row():
                gr.Button("Confirm prediction").click(
                    lambda j, f, v, w: decide(j, f, "confirm", v, w), [jid, fld, val, who], [q, note]
                )
                gr.Button("Save correction", variant="primary").click(
                    lambda j, f, v, w: decide(j, f, "correct", v, w), [jid, fld, val, who], [q, note]
                )
        with gr.Tab("Export"):
            out = gr.File(label="labels.jsonl")
            gr.Button("Export reviewed labels").click(export, None, out)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860)
