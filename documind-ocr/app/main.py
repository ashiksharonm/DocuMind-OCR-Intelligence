from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.middleware.perf import LatencyMiddleware

app = FastAPI(
    title="DocuMind OCR Intelligence",
    description=(
        "CV-NLP Document Extraction API for food-tech operational reviews. "
        "Processes invoices, receipts, and POS slips with 95%+ field extraction "
        "accuracy and sub-2-second transaction times."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Latency / SLA monitoring ───────────────────────────────────────────────
# Tracks per-request ms, warns on SLA breaches, exposes /metrics endpoint.
app.add_middleware(LatencyMiddleware)

# ── Routes ─────────────────────────────────────────────────────────────────
app.include_router(router)


@app.get("/health", tags=["ops"])
def health_check():
    return {"status": "ok", "service": "DocuMind OCR", "version": "1.0.0"}
