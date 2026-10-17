import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import Settings, settings
from app.middleware.perf import LatencyMiddleware
from app.review.api import router as review_router
from app.review.queue import make_dispatcher
from app.review.store import Store
from app.review.worker import start_embedded

log = logging.getLogger("documind")
__version__ = "1.1.0"


def create_app(cfg: Optional[Settings] = None, store: Optional[Store] = None) -> FastAPI:
    cfg = cfg or settings
    logging.basicConfig(level=cfg.LOG_LEVEL)
    store = store or Store(cfg.DB_PATH)
    dispatcher = make_dispatcher(store, cfg.REDIS_URL)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = start_embedded(store, dispatcher) if cfg.EMBEDDED_WORKER else None
        yield
        if stop:
            stop.set()

    app = FastAPI(
        title="DocuMind OCR Intelligence",
        description=(
            "Document extraction API for invoices, receipts and POS slips with confidence-routed "
            "human review: low-confidence fields go to a review queue and reviewer corrections are "
            "exported as a labelled dataset. Accuracy depends on the OCR engine and documents; "
            "see the README for the measured synthetic benchmark."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.store, app.state.dispatcher = store, dispatcher

    origins = [o.strip() for o in cfg.CORS_ORIGINS.split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(LatencyMiddleware)

    @app.middleware("http")
    async def request_id(request: Any, call_next: Any) -> Any:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info(
            "request id=%s %s %s -> %s %.1fms",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
        )
        return response

    app.include_router(router)  # legacy synchronous /extract, /batch, /metrics
    app.include_router(review_router)

    @app.get("/health", tags=["ops"])
    @app.get("/healthz", tags=["ops"])
    def health_check() -> dict:
        return {"status": "ok", "service": "DocuMind OCR", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def ready() -> dict:
        try:
            ok = store.ping()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"database unavailable: {exc}") from exc
        return {"status": "ready" if ok else "degraded", "ocr_engine": cfg.OCR_ENGINE}

    return app


_app: Optional[FastAPI] = None


def __getattr__(name: str) -> FastAPI:
    """``uvicorn app.main:app`` builds the app on first access (no DB side effects on import)."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
