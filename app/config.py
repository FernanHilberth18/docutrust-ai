from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocuTrust AI"
    environment: str = "development"
    docutrust_data_dir: Path = Path(".data")
    min_confidence: float = 0.12
    max_upload_mb: int = 15

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
