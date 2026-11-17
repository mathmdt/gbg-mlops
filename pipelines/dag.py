"""Prefect DAG: ingestion → preparation → training → evaluation → promotion.

    make up     # MinIO, MLflow and the Prefect server (UI on http://localhost:4200)
    make dag    # runs the flow from the repository root; follow it in the Prefect UI

Training and evaluation are the scripts of `training/` (contract section 2: they
register a new `gbg-risk` version with its `gbg_reckless_auc`). The last step offers
that version to `champion` with the same metric-gated rule as `make promote`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

from gbg_mlops.config import get_settings
from pipelines import ingest, prepare
from scripts.promote_model import promote


@task(name="ingest")
def ingest_raw(raw_dir: Path) -> dict[str, str]:
    return ingest.ingest(raw_dir)


@task(name="prepare")
def prepare_data(raw_dir: Path, out_dir: Path) -> None:
    prepare.prepare(raw_dir, out_dir)


@task
def run_script(script: str) -> bool:
    """Run a script of `training/`; False when it does not exist yet."""
    if not Path(script).exists():
        # shortcut: lets the DAG run before training/ lands; make it an error once it has.
        get_run_logger().warning("%s not found: step skipped", script)
        return False
    subprocess.run([sys.executable, script], check=True)
    return True


@task(name="promote")
def promote_latest() -> dict:
    """Offer the latest registered version to `champion` (kept if not strictly better)."""
    settings = get_settings()
    settings.export_mlflow_env()
    uri = settings.mlflow_tracking_uri
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    versions = client.search_model_versions(f"name='{settings.model_name}'")
    latest = str(max(int(v.version) for v in versions))  # the one training just registered
    decision = promote(client, settings.model_name, settings.model_alias, latest)
    get_run_logger().info("%s: %s (restart the API to serve it)", decision.status, decision.reason)
    return vars(decision)


@flow(name="gbg-pipeline", log_prints=True)
def gbg_pipeline(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    train_script: str = "training/train.py",
    evaluate_script: str = "training/evaluate.py",
) -> None:
    ingest_raw(Path(raw_dir))
    prepare_data(Path(raw_dir), Path(out_dir))
    if run_script.with_options(name="train")(train_script):
        run_script.with_options(name="evaluate")(evaluate_script)
        promote_latest()


if __name__ == "__main__":
    gbg_pipeline()
