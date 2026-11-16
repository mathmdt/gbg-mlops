"""Runtime settings read from environment variables (see `.env.example`)."""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by the API, the training code and the scripts."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    model_name: str = "gbg-risk"
    model_alias: str = "champion"
    risk_threshold: float = Field(default=0.5, gt=0, lt=1)

    def export_mlflow_env(self) -> None:
        """Expose the S3 settings as process env vars, where MLflow and boto3 read them.

        Values already set in the environment are kept.
        """
        values = {
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_S3_ENDPOINT_URL": self.mlflow_s3_endpoint_url,
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
        }
        for key, value in values.items():
            if value is not None:
                os.environ.setdefault(key, value)


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read once per process."""
    return Settings()
