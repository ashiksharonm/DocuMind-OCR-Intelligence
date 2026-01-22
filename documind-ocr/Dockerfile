FROM python:3.11-slim

# Install system dependencies
# poppler-utils for pdf2image
# tesseract-ocr for pytesseract
# libgl1-mesa-glx for opencv (sometimes needed even with headless, but headless usually avoids it. Including just in case logic needs it or other libs)
# build-essential for compiling some python libs if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libglib2.0-0 \
    lsb-release \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY . .

# Create artifacts directory
RUN mkdir -p artifacts reports dataset

# Expose port
EXPOSE 8000

# Command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
