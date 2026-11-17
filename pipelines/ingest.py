"""Upload the raw files to the MinIO bucket `raw`, tagged with their SHA-256.

    uv run python -m pipelines.ingest                 # data/raw/* → s3://raw/

Each object carries its fingerprint in the `sha256` metadata (visible in the MinIO
console). A file already in the bucket with the same fingerprint is not sent again,
so the DAG can run this step every time.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from gbg_mlops.config import get_settings

RAW_DIR = Path("data/raw")
RAW_FILES = ("SRV3490.parquet", "GBG500.parquet", "GBG500_labels.csv")
BUCKET = "raw"  # created by `minio-init` in docker-compose.yml


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.mlflow_s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def remote_sha256(client, key: str) -> str | None:
    try:
        return client.head_object(Bucket=BUCKET, Key=key)["Metadata"].get("sha256")
    except ClientError as error:
        if error.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def ingest(raw_dir: Path = RAW_DIR, client=None) -> dict[str, str]:
    """Upload every raw file whose fingerprint differs from the bucket's; return the digests."""
    client = client or s3_client()
    digests = {}
    for name in RAW_FILES:
        path = raw_dir / name
        digest = sha256(path)
        if remote_sha256(client, name) == digest:
            print(f"{name}: unchanged ({digest[:12]})")
        else:
            client.upload_file(str(path), BUCKET, name, ExtraArgs={"Metadata": {"sha256": digest}})
            print(f"{name}: uploaded to s3://{BUCKET}/{name} ({digest[:12]})")
        digests[name] = digest
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="folder of the raw files")
    ingest(parser.parse_args().raw)


if __name__ == "__main__":
    main()
