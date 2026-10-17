"""Confidence-routed human review: scoring, store, queue, worker, API and OCR integration."""

import json
import shutil
import threading
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.main import create_app
from app.pipelines.extract_fields import _sum_check, extract_amount, extract_vendor
from app.pipelines.ocr import OCRResult, group_lines, run_ocr
from app.review import store as st
from app.review.confidence import combine, locate_tokens, score_fields
from app.review.processing import load_pages, process_bytes, process_pages
from app.review.queue import REDIS_KEY, RedisDispatcher, SqliteDispatcher, make_dispatcher
from app.review.store import Store
from app.review.synthetic import generate, make_label, render
from app.review.worker import Worker

HAS_TESSERACT = shutil.which("tesseract") is not None


# ----------------------------------------------------------------- helpers
def fake_ocr(lines: List[str], conf: float = 0.95):
    """An OCR stand-in: one token per word, laid out on rows, constant confidence."""

    def ocr(_img: np.ndarray) -> Dict[str, Any]:
        tokens = []
        for r, line in enumerate(lines):
            for c, word in enumerate(line.split()):
                x, y = 10 + c * 120, 10 + r * 40
                tokens.append(
                    {
                        "text": word,
                        "confidence": conf,
                        "bbox": [(x, y), (x + 100, y), (x + 100, y + 30), (x, y + 30)],
                    }
                )
        return {"engine": "fake", "tokens": tokens, "full_text": "\n".join(lines)}

    return ocr


GOOD = [
    "Green Garden Restaurant",
    "INVOICE",
    "Invoice Date: 2024-01-15",
    "Subtotal: $100.00",
    "Tax: $5.00",
    "Total: $105.00",
]
BLANK = np.full((120, 300, 3), 255, np.uint8)


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setattr(settings, "API_KEY", None)
    monkeypatch.setattr(settings, "REVIEW_THRESHOLD", 0.6)
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10_000_000)
    return Settings(
        DB_PATH=str(tmp_path / "db.sqlite"), UPLOAD_DIR=str(tmp_path / "up"), REVIEW_THRESHOLD=0.6
    )


@pytest.fixture()
def client(cfg):
    app = create_app(cfg)
    return TestClient(app)


# --------------------------------------------------------------- confidence
TOKENS = [
    {"text": "Total:", "confidence": 0.9, "bbox": [(0, 0), (10, 0), (10, 5), (0, 5)]},
    {"text": "$105.00", "confidence": 0.8, "bbox": [(20, 0), (60, 0), (60, 5), (20, 5)]},
    {"text": "Green", "confidence": 0.5, "bbox": [(0, 10), (9, 10), (9, 15), (0, 15)]},
    {"text": "Garden", "confidence": 0.7, "bbox": [(12, 10), (30, 10), (30, 15), (12, 15)]},
]


def test_locate_tokens_matches_runs_and_amounts():
    assert [t["text"] for t in locate_tokens("105.00", TOKENS)] == ["$105.00"]
    assert [t["text"] for t in locate_tokens("Green Garden", TOKENS)] == ["Green", "Garden"]
    assert locate_tokens("Nowhere", TOKENS) == [] and locate_tokens(None, TOKENS) == []
    assert locate_tokens("!!", TOKENS) == []


def test_combine_and_score_fields():
    assert combine(0.9, [0.8, 1.0]) == 0.81
    assert combine(0.9, []) == 0.45
    fields = {
        "vendor_name": {"value": "Green Garden", "confidence": 0.8, "raw": "Green Garden"},
        "total_amount": {"value": 105.0, "confidence": 0.9, "raw": "105.00"},
        "invoice_date": {"value": None, "confidence": 0.0},
    }
    s = score_fields(fields, TOKENS)
    assert s["vendor_name"]["confidence"] == 0.48 and s["vendor_name"]["bbox"] == [0, 10, 30, 15]
    assert s["total_amount"]["ocr_confidence"] == 0.8
    assert s["invoice_date"]["confidence"] == 0.0 and s["invoice_date"]["bbox"] is None


# ----------------------------------------------------------------- extraction
def test_sum_check_boosts_and_penalises_total():
    ok = "Subtotal: $100.00\nTax: $5.00\nTotal: $105.00"
    bad = "Subtotal: $100.00\nTax: $5.00\nTotal: $185.00"
    assert _sum_check(ok, 105.0) is True and _sum_check(bad, 185.0) is False
    assert _sum_check("Total: $5.00", 5.0) is None
    assert extract_amount(ok, [])["confidence"] >= 0.95
    assert extract_amount(bad, [])["confidence"] <= 0.4


def test_vendor_skips_label_lines():
    v = extract_vendor("INVOICE\nInvoice Date: 2024-01-01\nSpice Route Kitchen\nTotal: 5.00", [])
    assert v["value"] == "Spice Route Kitchen" and v["raw"] == v["value"]
    assert extract_vendor("12345\n#99", [])["confidence"] == 0.3
    assert extract_vendor("", [])["value"] is None


# ------------------------------------------------------------------------ ocr
def test_group_lines_orders_rows_and_columns():
    def tok(t, x, y):
        return OCRResult(t, 0.9, [(x, y), (x + 40, y), (x + 40, y + 20), (x, y + 20)])

    lines = group_lines([tok("b", 100, 5), tok("c", 0, 60), tok("a", 0, 0)])
    assert [[t.text for t in ln] for ln in lines] == [["a", "b"], ["c"]]
    assert group_lines([]) == []


def test_run_ocr_unknown_engine():
    with pytest.raises(ValueError):
        run_ocr(BLANK, "nope")


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")
def test_real_tesseract_pipeline_reads_clean_synthetic_document():
    label = make_label(__import__("random").Random(3))
    res = process_pages([render(label, "clean", 3)])
    f = res["fields"]
    hits = [
        f["vendor_name"]["value"] == label["vendor_name"],
        f["invoice_date"]["value"] == label["invoice_date"],
        f["total_amount"]["value"] is not None
        and abs(float(f["total_amount"]["value"]) - float(label["total_amount"])) < 0.005,
    ]
    # Tesseract output differs between versions (5.3 on CI vs 5.5 locally), so require most fields.
    assert sum(hits) >= 2, (hits, {k: v["value"] for k, v in f.items()}, label)
    assert res["engine"] == "tesseract" and f["total_amount"]["bbox"] is not None


# ----------------------------------------------------------------- synthetic
def test_synthetic_is_seeded_and_levels_differ():
    a, b = generate(5, 7), generate(5, 7)
    assert [x[0] for x in a] == [x[0] for x in b] and np.array_equal(a[1][2], b[1][2])
    assert {lvl for _, lvl, _ in generate(5, 0)} == {"clean", "noisy", "rotated", "degraded", "severe"}
    with pytest.raises(ValueError):
        render(a[0][0], "bogus")


def test_load_pages():
    assert load_pages(b"garbage") == [] and len(load_pages(png(BLANK))) == 1
    with pytest.raises(ValueError):
        process_bytes(b"garbage")


# ---------------------------------------------------------------------- store
def make_store(tmp_path=None):
    return Store(":memory:" if tmp_path is None else str(tmp_path / "s.db"))


def field(v, c):
    return {"value": v, "confidence": c, "raw": v, "bbox": None}


def test_complete_routes_by_threshold_and_missing_values():
    s = make_store()
    j1 = s.create_job("a.png", "image/png", "h1", "/x", 0.75)
    s.claim(j1)
    assert (
        s.complete(
            j1,
            "invoice",
            "fake",
            "t",
            {
                "vendor_name": field("A", 0.9),
                "invoice_date": field("2024-01-01", 0.8),
                "total_amount": field("5", 0.76),
            },
            0.75,
        )
        == st.AUTO
    )
    j2 = s.create_job("b.png", "image/png", "h2", "/x", 0.75)
    s.claim(j2)
    assert (
        s.complete(
            j2,
            "invoice",
            "fake",
            "t",
            {
                "vendor_name": field("A", 0.9),
                "invoice_date": field(None, 0.0),
                "total_amount": field("5", 0.5),
            },
            0.75,
        )
        == st.NEEDS_REVIEW
    )
    q = s.review_queue()
    assert [(i["job_id"], i["name"]) for i in q] == [(j2, "invoice_date"), (j2, "total_amount")]
    assert s.review_queue(limit=1, offset=1)[0]["name"] == "total_amount"


def test_claim_is_fifo_exclusive_and_specific():
    s = make_store()
    a = s.create_job("a", "", "h", "/x", 0.7)
    b = s.create_job("b", "", "h", "/x", 0.7)
    assert s.claim(b)["id"] == b and s.claim(b) is None
    assert s.claim()["id"] == a and s.claim() is None
    assert s.get_job("missing") is None


def test_review_flow_export_and_stats():
    s = make_store()
    j = s.create_job("a.png", "image/png", "sha", "/x", 0.75)
    s.claim(j)
    s.complete(
        j,
        "receipt",
        "fake",
        "ocr text",
        {
            "vendor_name": field("Cafe", 0.9),
            "invoice_date": field("2024-02-02", 0.3),
            "total_amount": field("10.0", 0.4),
        },
        0.75,
    )
    assert list(s.export_dataset("reviewed")) == []  # no human decision yet
    s.review_field(j, "invoice_date", "ann", "2024-02-03", "correct")
    job = s.review_field(j, "total_amount", "ann", None, "confirm")
    assert job["status"] == st.REVIEWED and job["fields"]["invoice_date"]["source"] == "human_corrected"
    rec = next(iter(s.export_dataset("reviewed")))
    assert rec["fields"]["invoice_date"] == {
        "predicted": "2024-02-02",
        "label": "2024-02-03",
        "confidence": 0.3,
        "source": "human_corrected",
    }
    assert rec["fields"]["total_amount"]["source"] == "human_confirmed"
    assert rec["fields"]["vendor_name"]["source"] == "auto_accepted" and rec["ocr_text"] == "ocr text"
    sts = s.stats()
    assert sts["fields"]["corrected"] == 1 and sts["correction_rate"] == 0.5 and sts["review_rate"] == 0.6667
    assert s.review_field(j, "nope", "ann", None, "confirm") is None and s.ping()


def test_export_all_includes_auto_accepted_jobs_but_reviewed_does_not():
    s = make_store()
    j = s.create_job("a", "", "h", "/x", 0.5)
    s.claim(j)
    s.complete(j, "invoice", "fake", "t", {"vendor_name": field("A", 0.9)}, 0.5)
    assert list(s.export_dataset("reviewed")) == []
    assert next(iter(s.export_dataset("all")))["fields"]["vendor_name"]["source"] == "auto_accepted"
    assert Store(":memory:").stats()["review_rate"] is None


def test_stale_jobs_are_requeued_then_failed(tmp_path):
    s = make_store(tmp_path)
    j = s.create_job("a", "", "h", "/x", 0.7)
    s.claim(j)
    assert s.recover_stale(0, max_attempts=3) == 1 and s.get_job(j)["status"] == st.QUEUED
    s.claim(j)
    s.requeue_or_fail(j, "boom", 3)
    s.claim(j)
    assert s.recover_stale(0, max_attempts=3) == 1 and s.get_job(j)["status"] == st.FAILED
    assert s.recover_stale(0, 3) == 0


def test_concurrent_workers_never_claim_the_same_job(tmp_path):
    s = make_store(tmp_path)
    for i in range(12):
        s.create_job(f"f{i}", "", "h", "/x", 0.7)
    got: List[str] = []
    lock = threading.Lock()

    def work():
        while (j := s.claim()) is not None:
            with lock:
                got.append(j["id"])

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(got) == 12 and len(set(got)) == 12


# ---------------------------------------------------------------------- queue
def test_dispatchers():
    s = make_store()
    j = s.create_job("a", "", "h", "/x", 0.7)
    d = SqliteDispatcher(s)
    d.publish(j)
    assert d.claim_next(0)["id"] == j and d.claim_next(0) is None
    assert isinstance(make_dispatcher(s, None), SqliteDispatcher)


def test_redis_dispatcher_with_fakeredis_and_lost_messages():
    fakeredis = pytest.importorskip("fakeredis")
    s, r = make_store(), fakeredis.FakeRedis()
    d = RedisDispatcher(s, r)
    a, b = s.create_job("a", "", "h", "/x", 0.7), s.create_job("b", "", "h", "/x", 0.7)
    d.publish(a)  # b's message is "lost": the sweep still finds it
    assert r.llen(REDIS_KEY) == 1
    assert d.claim_next(1)["id"] == a and d.claim_next(1)["id"] == b and d.claim_next(1) is None


# --------------------------------------------------------------------- worker
def test_worker_routes_confident_and_uncertain_documents(client, cfg):
    store, disp = client.app.state.store, client.app.state.dispatcher
    img = png(BLANK)
    ids = {}
    for key in ("sure", "unsure"):
        r = client.post("/v1/jobs", files={"file": (f"{key}.png", img, "image/png")})
        assert r.status_code == 202
        ids[key] = r.json()["job_id"]
    assert Worker(store, disp, cfg, fake_ocr(GOOD, 0.99)).run_once() == st.AUTO
    assert Worker(store, disp, cfg, fake_ocr(GOOD, 0.4)).run_once() == st.NEEDS_REVIEW
    assert Worker(store, disp, cfg, fake_ocr(GOOD)).run_once() is None
    assert store.get_job(ids["sure"])["status"] == st.AUTO
    assert store.get_job(ids["unsure"])["status"] == st.NEEDS_REVIEW


def test_worker_retries_then_fails_corrupt_file(client, cfg):
    store, disp = client.app.state.store, client.app.state.dispatcher
    jid = store.create_job("bad", "", "h", cfg.UPLOAD_DIR + "/bad", 0.75)
    import os

    os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)
    open(cfg.UPLOAD_DIR + "/bad", "wb").write(b"not an image")
    w = Worker(store, disp, cfg, fake_ocr(GOOD))
    assert [w.run_once() for _ in range(3)] == [st.QUEUED, st.QUEUED, st.FAILED]
    assert "decode" in store.get_job(jid)["error"]


def test_worker_run_forever_stops(client, cfg):
    stop = threading.Event()
    stop.set()
    Worker(client.app.state.store, client.app.state.dispatcher, cfg).run_forever(stop)


# ------------------------------------------------------------------------ api
def test_end_to_end_review_over_http(client, cfg):
    store, disp = client.app.state.store, client.app.state.dispatcher
    sub = client.post("/v1/jobs", files={"file": ("../../evil name.png", png(BLANK), "image/png")})
    jid = sub.json()["job_id"]
    assert client.get(f"/v1/jobs/{jid}").json()["status"] == "queued"
    low = ["Green Garden Restaurant", "Invoice Date: 2024-01-15", "Total: $105.00"]
    assert Worker(store, disp, cfg, fake_ocr(low, 0.5)).run_once() == st.NEEDS_REVIEW
    job = client.get(f"/v1/jobs/{jid}").json()
    assert "file_path" not in job and job["filename"] == "evil name.png"
    queue = client.get("/v1/reviews/queue").json()
    assert queue["count"] == 3 and queue["items"][0]["confidence"] <= queue["items"][-1]["confidence"]
    assert client.get(f"/v1/jobs/{jid}/image").headers["content-type"] == "image/png"

    url = f"/v1/reviews/{jid}/fields"
    assert (
        client.post(
            f"{url}/invoice_date", json={"action": "correct", "reviewer": "ann", "value": "15/01/2024"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"{url}/total_amount", json={"action": "correct", "reviewer": "ann", "value": "abc"}
        ).status_code
        == 422
    )
    assert (
        client.post(f"{url}/total_amount", json={"action": "correct", "reviewer": "ann"}).status_code == 422
    )
    assert client.post(f"{url}/colour", json={"action": "confirm", "reviewer": "a"}).status_code == 404
    assert (
        client.post(
            "/v1/reviews/nope/fields/vendor_name", json={"action": "confirm", "reviewer": "a"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"{url}/invoice_date", json={"action": "correct", "reviewer": "ann", "value": "2024-01-16"}
        ).status_code
        == 200
    )
    client.post(f"{url}/total_amount", json={"action": "correct", "reviewer": "ann", "value": "1,050.00"})
    done = client.post(f"{url}/vendor_name", json={"action": "confirm", "reviewer": "bob"}).json()
    assert done["status"] == "reviewed"

    lines = client.get("/v1/dataset/export").text.strip().splitlines()
    rec = json.loads(lines[0])
    assert len(lines) == 1 and rec["fields"]["invoice_date"]["label"] == "2024-01-16"
    assert rec["fields"]["vendor_name"]["source"] == "human_confirmed"
    assert client.get("/v1/dataset/export?include=all").status_code == 200
    assert client.get("/v1/stats").json()["correction_rate"] == 0.6667


def test_upload_validation(client, monkeypatch):
    f = lambda data: {"file": ("x.png", data, "image/png")}  # noqa: E731
    assert client.post("/v1/jobs", files=f(b"")).status_code == 400
    assert client.post("/v1/jobs", files=f(b"not an image")).status_code == 400
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 100)
    assert client.post("/v1/jobs", files=f(png(BLANK))).status_code == 413
    assert client.get("/v1/jobs/unknown").status_code == 404
    assert client.get("/v1/jobs/unknown/image").status_code == 404


def test_api_key_and_ops_endpoints(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "k")
    assert client.get("/v1/stats").status_code == 401
    assert client.get("/v1/stats", headers={"x-api-key": "k"}).status_code == 200
    for p in ("/health", "/healthz", "/readyz"):
        assert client.get(p).status_code == 200
    r = client.get("/healthz", headers={"x-request-id": "abc"})
    assert r.headers["x-request-id"] == "abc" and "x-request-id" in client.get("/health").headers


def test_embedded_worker_processes_a_job(cfg, monkeypatch):
    import time

    import app.review.worker as w

    monkeypatch.setattr(w, "Worker", lambda store, disp: Worker(store, disp, cfg, fake_ocr(GOOD, 0.99)))
    cfg.EMBEDDED_WORKER, cfg.WORKER_POLL_SECONDS = True, 0.05
    with TestClient(create_app(cfg)) as c:
        jid = c.post("/v1/jobs", files={"file": ("a.png", png(BLANK), "image/png")}).json()["job_id"]
        for _ in range(100):
            if c.get(f"/v1/jobs/{jid}").json()["status"] != "queued":
                break
            time.sleep(0.05)
        assert c.get(f"/v1/jobs/{jid}").json()["status"] in (st.AUTO, st.PROCESSING)


# ------------------------------------------------------------------- deskew
def _skew_error(img: np.ndarray) -> float:
    """Residual rotation (degrees) of dark text pixels, via minAreaRect folded to [-45, 45)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    pts = np.column_stack(np.where(th > 0))
    return abs(((cv2.minAreaRect(pts)[-1] + 45) % 90) - 45)


def test_deskew_leaves_straight_pages_alone_and_straightens_tilted_ones():
    from app.pipelines.preprocess import correct_skew

    label = make_label(__import__("random").Random(5))
    straight = render(label, "clean", 5)
    out = correct_skew(straight)
    assert out.shape == straight.shape
    assert float(np.abs(out.astype(int) - straight.astype(int)).mean()) < 2.0  # not rotated by ~90
    h, w = straight.shape[:2]
    for deg in (-3.0, 3.0):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        tilted = cv2.warpAffine(straight, m, (w, h), borderValue=(255, 255, 255))
        assert _skew_error(correct_skew(tilted)) < _skew_error(tilted)
