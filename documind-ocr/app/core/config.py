"""Environment-driven settings. Every value has a free, local default."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, extra="ignore", env_file=".env")

    PROJECT_NAME: str = "DocuMind OCR"
    API_V1_STR: str = "/api/v1"
    ARTIFACTS_DIR: str = "artifacts"
    REPORTS_DIR: str = "reports"

    # OCR: tesseract (default, system binary), paddleocr or easyocr (optional extras)
    OCR_ENGINE: str = "tesseract"

    # Human-in-the-loop review
    REVIEW_THRESHOLD: float = 0.75  # fields below this confidence go to the review queue
    DB_PATH: str = "data/documind.db"
    UPLOAD_DIR: str = "data/uploads"
    REDIS_URL: str | None = None  # optional: dispatch jobs through Redis instead of polling SQLite
    EMBEDDED_WORKER: bool = False  # run a worker thread inside the API process
    WORKER_POLL_SECONDS: float = 1.0
    JOB_VISIBILITY_TIMEOUT_S: int = 120  # a job stuck in "processing" longer than this is retried
    MAX_ATTEMPTS: int = 3
    MAX_UPLOAD_BYTES: int = 10_000_000
    API_KEY: str | None = None  # when set, /v1/* requires X-API-Key
    CORS_ORIGINS: str = ""  # comma separated; empty disables CORS
    LOG_LEVEL: str = "INFO"


settings = Settings()
