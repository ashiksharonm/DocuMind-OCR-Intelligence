"""Measure extraction accuracy and confidence routing on seeded synthetic documents.

    python scripts/evaluate_synthetic.py --n 60 --seed 0

Writes reports/eval_synthetic.json and reports/eval_synthetic.md. The documents are rendered
text with controlled degradation, so the numbers are optimistic for real scans. They are meant
to show how the confidence score separates right from wrong fields, and to catch regressions.
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.review.processing import process_pages  # noqa: E402
from app.review.synthetic import generate  # noqa: E402

FIELDS = ("vendor_name", "invoice_date", "total_amount")
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85)


def correct(name: str, pred: Any, gold: str) -> bool:
    if pred is None:
        return False
    if name == "total_amount":
        try:
            return abs(float(pred) - float(gold)) < 0.005
        except (TypeError, ValueError):
            return False
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()  # noqa: E731
    return norm(pred) == norm(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports"))
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for label, level, img in generate(args.n, args.seed):
        res = process_pages([img])
        for name in FIELDS:
            f = res["fields"][name]
            rows.append(
                {
                    "level": level,
                    "field": name,
                    "conf": f["confidence"],
                    "ok": correct(name, f["value"], label[name]),
                    "doc": id(label),
                }
            )
    per_doc_ms = (time.perf_counter() - t0) * 1000 / args.n

    def acc(sel: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"n": len(sel), "accuracy": round(sum(r["ok"] for r in sel) / len(sel), 4) if sel else None}

    report: Dict[str, Any] = {
        "n_docs": args.n,
        "seed": args.seed,
        "threshold": args.threshold,
        "ocr_engine": os.environ.get("OCR_ENGINE", "tesseract"),
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "ms_per_doc_pipeline": round(per_doc_ms, 1),
        "overall": acc(rows),
        "by_field": {f: acc([r for r in rows if r["field"] == f]) for f in FIELDS},
        "by_level": {
            lv: acc([r for r in rows if r["level"] == lv]) for lv in dict.fromkeys(r["level"] for r in rows)
        },
    }
    sweep = []
    for th in THRESHOLDS:
        auto = [r for r in rows if r["conf"] >= th]
        flagged = [r for r in rows if r["conf"] < th]
        by_doc: Dict[int, List[bool]] = defaultdict(list)
        silent = 0
        docs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            docs[r["doc"]].append(r)
        for rs in docs.values():
            if all(r["conf"] >= th for r in rs) and not all(r["ok"] for r in rs):
                silent += 1
        sweep.append(
            {
                "threshold": th,
                "auto_fields": len(auto),
                "flagged_fields": len(flagged),
                "share_flagged": round(len(flagged) / len(rows), 4),
                "auto_accuracy": acc(auto)["accuracy"],
                "flagged_accuracy": acc(flagged)["accuracy"],
                "docs_fully_auto_with_an_error": silent,
                "n_docs": len(docs),
            }
        )
        del by_doc
    report["threshold_sweep"] = sweep

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval_synthetic.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    md = [
        f"# Synthetic evaluation ({args.n} docs, seed {args.seed}, OCR {report['ocr_engine']})",
        "",
        f"Overall field accuracy: **{report['overall']['accuracy']}** over {report['overall']['n']} fields; "
        f"{report['ms_per_doc_pipeline']} ms/doc end to end on {report['environment']['machine']}.",
        "",
        "| field | n | accuracy |",
        "|---|---|---|",
    ]
    md += [f"| {k} | {v['n']} | {v['accuracy']} |" for k, v in report["by_field"].items()]
    md += ["", "| degradation | n fields | accuracy |", "|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['accuracy']} |" for k, v in report["by_level"].items()]
    md += [
        "",
        "| threshold | flagged fields | share flagged | accuracy auto | accuracy flagged | docs fully auto with an error |",
        "|---|---|---|---|---|---|",
    ]
    md += [
        f"| {s['threshold']} | {s['flagged_fields']} | {s['share_flagged']} | {s['auto_accuracy']} | "
        f"{s['flagged_accuracy']} | {s['docs_fully_auto_with_an_error']}/{s['n_docs']} |"
        for s in sweep
    ]
    with open(os.path.join(args.out, "eval_synthetic.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
