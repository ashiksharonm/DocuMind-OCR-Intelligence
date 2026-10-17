"""Per-field confidence: extractor heuristic score combined with OCR token confidence.

The result is a heuristic ranking signal for routing, not a calibrated probability. Use
``scripts/evaluate_synthetic.py`` to see how it relates to accuracy on a labelled set.
"""

import re
from typing import Any, Dict, List, Optional


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def locate_tokens(raw: Optional[str], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OCR tokens that make up ``raw`` (matched on alphanumerics, in order of appearance)."""
    if not raw:
        return []
    target = _norm(raw)
    if not target:
        return []
    for i in range(len(tokens)):  # find a run of consecutive tokens spelling the target
        acc = ""
        run: List[Dict[str, Any]] = []
        for t in tokens[i:]:
            n = _norm(t["text"])
            if not n or not target.startswith(acc + n):
                break
            acc, run = acc + n, [*run, t]
            if acc == target:
                return run
    return []


def combine(extractor_conf: float, token_confs: List[float]) -> float:
    """``extractor_conf`` times the mean OCR confidence; halved when no token could be located."""
    if not token_confs:
        return round(extractor_conf * 0.5, 4)
    return round(extractor_conf * (sum(token_confs) / len(token_confs)), 4)


def score_fields(
    fields: Dict[str, Dict[str, Any]], tokens: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Return ``{name: {value, confidence, extractor_confidence, ocr_confidence, bbox, raw}}``."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, f in fields.items():
        located = locate_tokens(f.get("raw"), tokens)
        confs = [float(t["confidence"]) for t in located]
        xs = [p[0] for t in located for p in t["bbox"]]
        ys = [p[1] for t in located for p in t["bbox"]]
        value = f.get("value")
        conf = 0.0 if value is None else combine(float(f.get("confidence", 0.0)), confs)
        out[name] = {
            "value": value,
            "confidence": conf,
            "extractor_confidence": float(f.get("confidence", 0.0)),
            "ocr_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "raw": f.get("raw"),
            "bbox": [min(xs), min(ys), max(xs), max(ys)] if located else None,
        }
    return out
