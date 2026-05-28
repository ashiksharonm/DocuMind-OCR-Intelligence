.PHONY: run test lint build clean docker-run benchmark evaluate

run:
	cd documind-ocr && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

build:
	cd documind-ocr && docker build -t documind-ocr .

docker-run:
	cd documind-ocr && docker compose up --build

test:
	cd documind-ocr && pytest tests/ -v --tb=short

lint:
	cd documind-ocr && ruff check .

benchmark:
	cd documind-ocr && python scripts/benchmark_preprocess.py

evaluate:
	cd documind-ocr && python scripts/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
