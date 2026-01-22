# DocuMind OCR Intelligence

DocuMind is an end-to-end document intelligence API designed to extract structured data from invoices using open-source tools.

## Features
- **Multi-Format Ingestion**: Support for PDF and Image (JPG/PNG) invoices.
- **Hybrid OCR**: Uses EasyOCR with Tesseract fallback for robust text detection.
- **Intelligent Extraction**: combination of Regex, Layout heuristics, and NLP for fields (Vendor, Date, Amount).
- **Classification**: Distinguishes between Invoices and Receipts.
- **Observability**: Detailed logging and confidence scoring.

## Tech Stack
- **Framework**: FastAPI, Uvicorn
- **OCR**: EasyOCR, Tesseract, OpenCV
- **Data**: Pandas, NumPy, Spacy
- **Ops**: Docker, GitHub Actions

## Setup & Running

### Local
1. Install system dependencies (Tesseract, Poppler):
   ```bash
   # MacOS
   brew install tesseract poppler
   # Ubuntu
   sudo apt-get install tesseract-ocr poppler-utils
   ```
2. Install Python dependencies:
   ```bash
   pip install -r documind-ocr/requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Run API:
   ```bash
   make run
   ```

### Docker
1. Build and Run:
   ```bash
   make docker-run
   ```

## Usage
**Extract Fields**:
```bash
curl -X POST "http://localhost:8000/extract" -F "file=@/path/to/invoice.pdf"
```

## Architecture
(See implementation plan for details)
