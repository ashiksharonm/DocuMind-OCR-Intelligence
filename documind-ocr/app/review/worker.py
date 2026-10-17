"""Background worker: claim a queued job, run the pipeline, route fields by confidence.

    python -m app.review.worker            # separate process (docker compose service)
or in-process with EMBEDDED_WORKER=true (dev, Hugging Face Space).
"""

import logging
import os
import threading
from typing import Optional

from app.core.config import Settings, settings
from app.review.processing import OcrFn, load_pages, process_pages, write_preview
from app.review.queue import Dispatcher, make_dispatcher
from app.review.store import Store

log = logging.getLogger("documind.worker")


class Worker:
    def __init__(
        self, store: Store, dispatcher: Dispatcher, cfg: Settings = settings, ocr: Optional[OcrFn] = None
    ) -> None:
        self.store, self.dispatcher, self.cfg, self.ocr = store, dispatcher, cfg, ocr

    def run_once(self, timeout: float = 0.0) -> Optional[str]:
        """Process at most one job; returns its final status, or None when the queue is empty."""
        self.store.recover_stale(self.cfg.JOB_VISIBILITY_TIMEOUT_S, self.cfg.MAX_ATTEMPTS)
        job = self.dispatcher.claim_next(timeout)
        if job is None:
            return None
        jid = job["id"]
        try:
            with open(job["file_path"], "rb") as fh:
                pages = load_pages(fh.read())
            if not pages:
                raise ValueError("could not decode file as image or PDF")
            result = process_pages(pages, self.ocr)
            preview = write_preview(pages[0], os.path.join(self.cfg.UPLOAD_DIR, f"{jid}.preview.png"))
            status = self.store.complete(
                jid,
                result["doc_type"],
                result["engine"],
                result["ocr_text"],
                result["fields"],
                job["threshold"] or self.cfg.REVIEW_THRESHOLD,
                preview,
            )
            log.info("job %s -> %s", jid, status)
            return status
        except Exception as exc:  # noqa: BLE001 - any failure is retried then marked failed
            status = self.store.requeue_or_fail(jid, f"{type(exc).__name__}: {exc}", self.cfg.MAX_ATTEMPTS)
            log.warning("job %s failed (%s) -> %s", jid, exc, status)
            return status

    def run_forever(self, stop: Optional[threading.Event] = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            if self.run_once(self.cfg.WORKER_POLL_SECONDS) is None:
                stop.wait(self.cfg.WORKER_POLL_SECONDS)


def start_embedded(store: Store, dispatcher: Dispatcher) -> threading.Event:
    stop = threading.Event()
    threading.Thread(
        target=Worker(store, dispatcher).run_forever, args=(stop,), daemon=True, name="documind-worker"
    ).start()
    return stop


def main() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
    store = Store(settings.DB_PATH)
    log.info(
        "worker started (db=%s, ocr=%s, redis=%s)",
        settings.DB_PATH,
        settings.OCR_ENGINE,
        bool(settings.REDIS_URL),
    )
    Worker(store, make_dispatcher(store, settings.REDIS_URL)).run_forever()


if __name__ == "__main__":
    main()
