# MLflow tracking server for the dev compose: Postgres backend, S3 (MinIO) artifacts.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# Same MLflow and boto3 versions as uv.lock.
RUN pip install --no-cache-dir "mlflow==3.18.0" "boto3==1.44.3" "psycopg[binary]>=3.2,<4" \
    && useradd --create-home --uid 10001 mlflow

USER mlflow
EXPOSE 5000
