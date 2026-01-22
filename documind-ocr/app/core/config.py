import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuMind OCR"
    API_V1_STR: str = "/api/v1"
    ARTIFACTS_DIR: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")
    
    class Config:
        case_sensitive = True

settings = Settings()
