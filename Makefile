.PHONY: run test lint build clean

run:
	cd documind-ocr && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

build:
	cd documind-ocr && docker build -t documind-ocr .

docker-run:
	cd documind-ocr && docker-compose up --build

test:
	cd documind-ocr && pytest tests/

lint:
	cd documind-ocr && ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
