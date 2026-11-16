"""Runtime settings read from environment variables (see `.env.example`)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by the API, the training code and the scripts."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    model_name: str = "gbg-risk"
    model_alias: str = "champion"
    risk_threshold: float = Field(default=0.5, gt=0, lt=1)


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read once per process."""
    return Settings()
