# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="DocuMind OCR Team"
LABEL description="DocuMind OCR Intelligence — food-tech document extraction API"
LABEL version="1.0.0"

# System runtime dependencies (pinned for reproducibility)
#   tesseract-ocr : OCR fallback engine
#   poppler-utils : PDF → image conversion via pdf2image
#   libglib2.0-0  : OpenCV shared library dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr=5.3.0-2 \
    poppler-utils \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Download spaCy model (run as a build step so it's cached in the image layer)
RUN python -m spacy download en_core_web_sm

# Copy application source
COPY . .

# Create required runtime directories
RUN mkdir -p artifacts reports dataset/sample_docs

# Non-root user for security
RUN useradd --system --uid 1001 --no-create-home documind \
    && chown -R documind:documind /app
USER documind

# Expose API port
EXPOSE 8000

# Health check — verifies the API is alive within 30 s of start
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
