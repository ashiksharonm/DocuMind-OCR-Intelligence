"""SQLite job store: jobs, per-field results, reviewer corrections, and the dataset export.

SQLite is the system of record (WAL mode, one short-lived connection per call), which keeps the
default deployment dependency-free. Redis, when configured, only dispatches job ids.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

QUEUED, PROCESSING, AUTO, NEEDS_REVIEW, REVIEWED, FAILED = (
    "queued",
    "processing",
    "auto_accepted",
    "needs_review",
    "reviewed",
    "failed",
)
FIELD_NAMES = ("vendor_name", "invoice_date", "total_amount")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, filename TEXT, content_type TEXT, sha256 TEXT,
  file_path TEXT, preview_path TEXT, created_at REAL, updated_at REAL, started_at REAL,
  finished_at REAL, attempts INTEGER DEFAULT 0, error TEXT, doc_type TEXT, ocr_engine TEXT,
  ocr_text TEXT, threshold REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS fields (
  job_id TEXT NOT NULL REFERENCES jobs(id), name TEXT NOT NULL, predicted TEXT,
  confidence REAL, status TEXT NOT NULL, label TEXT, source TEXT, reviewer TEXT,
  reviewed_at REAL, detail TEXT, PRIMARY KEY (job_id, name)
);
CREATE INDEX IF NOT EXISTS fields_status ON fields(status);
"""


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._keep: Optional[sqlite3.Connection] = None
        if path == ":memory:":  # tests: share one connection
            self._keep = sqlite3.connect(":memory:", check_same_thread=False)
            self._keep.row_factory = sqlite3.Row
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._keep is not None:
            yield self._keep
            self._keep.commit()
            return
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """A write transaction: BEGIN IMMEDIATE ... COMMIT, rolled back on error."""
        with self._conn() as c:
            if self._keep is not None:
                yield c
                return
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            c.execute("COMMIT")

    # ---------------------------------------------------------------- jobs
    def create_job(
        self,
        filename: str,
        content_type: str,
        sha256: str,
        file_path: str,
        threshold: float,
        job_id: Optional[str] = None,
    ) -> str:
        jid, now = job_id or uuid.uuid4().hex, time.time()
        with self._tx() as c:
            c.execute(
                "INSERT INTO jobs(id,status,filename,content_type,sha256,file_path,created_at,"
                "updated_at,threshold) VALUES(?,?,?,?,?,?,?,?,?)",
                (jid, QUEUED, filename, content_type, sha256, file_path, now, now, threshold),
            )
        return jid

    def claim(self, job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Atomically move one queued job (a specific one, or the oldest) to processing."""
        with self._tx() as c:
            if job_id:
                row = c.execute("SELECT * FROM jobs WHERE id=? AND status=?", (job_id, QUEUED)).fetchone()
            else:
                row = c.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at LIMIT 1", (QUEUED,)
                ).fetchone()
            if row is None:
                return None
            now = time.time()
            c.execute(
                "UPDATE jobs SET status=?, started_at=?, updated_at=?, attempts=attempts+1 WHERE id=?",
                (PROCESSING, now, now, row["id"]),
            )
        return self.get_job(row["id"])

    def recover_stale(self, visibility_timeout_s: float, max_attempts: int) -> int:
        """Requeue jobs stuck in processing (crashed worker); fail those out of attempts."""
        cutoff, now, n = time.time() - visibility_timeout_s, time.time(), 0
        with self._tx() as c:
            rows = c.execute(
                "SELECT id, attempts FROM jobs WHERE status=? AND started_at<?", (PROCESSING, cutoff)
            ).fetchall()
            for r in rows:
                if r["attempts"] >= max_attempts:
                    c.execute(
                        "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                        (FAILED, "worker timed out too many times", now, r["id"]),
                    )
                else:
                    c.execute("UPDATE jobs SET status=?, updated_at=? WHERE id=?", (QUEUED, now, r["id"]))
                n += 1
        return n

    def requeue_or_fail(self, job_id: str, error: str, max_attempts: int) -> str:
        job = self.get_job(job_id)
        status = FAILED if job and job["attempts"] >= max_attempts else QUEUED
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error[:500], time.time(), job_id),
            )
        return status

    def complete(
        self,
        job_id: str,
        doc_type: str,
        engine: str,
        ocr_text: str,
        fields: Dict[str, Dict[str, Any]],
        threshold: float,
        preview_path: Optional[str] = None,
    ) -> str:
        """Store field results and route: any field under ``threshold`` (or empty) needs review."""
        now, needs = time.time(), False
        with self._tx() as c:
            c.execute("DELETE FROM fields WHERE job_id=?", (job_id,))
            for name, f in fields.items():
                value = f.get("value")
                pending = value is None or f["confidence"] < threshold
                needs = needs or pending
                c.execute(
                    "INSERT INTO fields(job_id,name,predicted,confidence,status,detail) VALUES(?,?,?,?,?,?)",
                    (
                        job_id,
                        name,
                        None if value is None else str(value),
                        f["confidence"],
                        "pending_review" if pending else "auto_accepted",
                        json.dumps(
                            {k: f.get(k) for k in ("extractor_confidence", "ocr_confidence", "raw", "bbox")}
                        ),
                    ),
                )
            status = NEEDS_REVIEW if needs else AUTO
            c.execute(
                "UPDATE jobs SET status=?, doc_type=?, ocr_engine=?, ocr_text=?, preview_path=?,"
                "finished_at=?, updated_at=?, error=NULL, threshold=? WHERE id=?",
                (status, doc_type, engine, ocr_text, preview_path, now, now, threshold, job_id),
            )
        return status

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            job = dict(row)
            job["fields"] = {
                r["name"]: self._field(r)
                for r in c.execute("SELECT * FROM fields WHERE job_id=? ORDER BY name", (job_id,))
            }
        return job

    @staticmethod
    def _field(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        d.update(json.loads(d.pop("detail") or "{}"))
        d.pop("job_id", None)
        return d

    # -------------------------------------------------------------- review
    def review_queue(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Pending fields, lowest confidence first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT f.*, j.filename, j.doc_type FROM fields f JOIN jobs j ON j.id=f.job_id "
                "WHERE f.status='pending_review' ORDER BY f.confidence ASC, j.created_at ASC "
                "LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        out = []
        for r in rows:
            d = self._field(r)
            d["job_id"] = r["job_id"]
            out.append(d)
        return out

    def review_field(
        self, job_id: str, name: str, reviewer: str, value: Optional[str], action: str
    ) -> Optional[Dict[str, Any]]:
        """Record a reviewer decision. ``action`` is ``confirm`` or ``correct`` (needs ``value``)."""
        now = time.time()
        with self._tx() as c:
            row = c.execute("SELECT * FROM fields WHERE job_id=? AND name=?", (job_id, name)).fetchone()
            if row is None:
                return None
            if action == "confirm":
                label, status, source = row["predicted"], "confirmed", "human_confirmed"
            else:
                label, status, source = value, "corrected", "human_corrected"
            c.execute(
                "UPDATE fields SET label=?, status=?, source=?, reviewer=?, reviewed_at=? "
                "WHERE job_id=? AND name=?",
                (label, status, source, reviewer, now, job_id, name),
            )
            left = c.execute(
                "SELECT COUNT(*) FROM fields WHERE job_id=? AND status='pending_review'", (job_id,)
            ).fetchone()[0]
            if left == 0:
                c.execute(
                    "UPDATE jobs SET status=?, updated_at=? WHERE id=? AND status=?",
                    (REVIEWED, now, job_id, NEEDS_REVIEW),
                )
        return self.get_job(job_id)

    # --------------------------------------------------------------- export
    def export_dataset(self, include: str = "reviewed") -> Iterator[Dict[str, Any]]:
        """Labelled records. ``reviewed``: jobs with at least one human decision.

        ``all`` also yields auto-accepted jobs, whose labels are the unverified predictions.
        """
        where = (
            "status IN ('reviewed','needs_review')"
            if include == "reviewed"
            else "status IN ('reviewed','needs_review','auto_accepted')"
        )
        with self._conn() as c:
            ids = [r["id"] for r in c.execute(f"SELECT id FROM jobs WHERE {where} ORDER BY created_at")]
        for jid in ids:
            job = self.get_job(jid)
            assert job is not None
            fields = {}
            for name, f in job["fields"].items():
                if f["status"] == "pending_review":
                    continue  # no label yet
                auto = f["status"] == "auto_accepted"
                fields[name] = {
                    "predicted": f["predicted"],
                    "label": f["predicted"] if auto else f["label"],
                    "confidence": f["confidence"],
                    "source": "auto_accepted" if auto else f["source"],
                }
            if include == "reviewed" and not any(v["source"] != "auto_accepted" for v in fields.values()):
                continue
            yield {
                "job_id": jid,
                "sha256": job["sha256"],
                "filename": job["filename"],
                "doc_type": job["doc_type"],
                "ocr_engine": job["ocr_engine"],
                "ocr_text": job["ocr_text"],
                "fields": fields,
            }

    def stats(self) -> Dict[str, Any]:
        with self._conn() as c:
            jobs = {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")}
            fs = {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM fields GROUP BY status")}
        reviewed = fs.get("confirmed", 0) + fs.get("corrected", 0)
        total = sum(fs.values())
        return {
            "jobs": jobs,
            "fields": fs,
            "review_rate": round((fs.get("pending_review", 0) + reviewed) / total, 4) if total else None,
            "correction_rate": round(fs.get("corrected", 0) / reviewed, 4) if reviewed else None,
        }

    def ping(self) -> bool:
        with self._conn() as c:
            return bool(c.execute("SELECT 1").fetchone()[0])
