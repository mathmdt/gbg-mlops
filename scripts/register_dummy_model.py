"""Register the placeholder model under `MODEL_NAME` and point `champion` to it.

    uv run python scripts/register_dummy_model.py            # or: make dummy-model
    uv run python scripts/register_dummy_model.py --force-champion

Each run logs a new run in the `gbg-risk` experiment and creates a new version.
The alias is moved only if there is no champion yet or if the current champion
is itself a dummy model, so a real model is never replaced by accident.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from gbg_mlops.config import get_settings
from gbg_mlops.features import CHANNELS, STRIDE, WINDOW, Normalization
from gbg_mlops.model_loader import (
    EXPERIMENT_NAME,
    MODEL_ARTIFACT_PATH,
    MODEL_SIGNATURE,
    NORMALIZATION_ARTIFACT,
)

MODEL_CODE = Path(__file__).with_name("dummy_model.py")
DUMMY_TAG = ("model_type", "dummy")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def current_alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def register_dummy_model(
    tracking_uri: str,
    model_name: str,
    alias: str,
    force_champion: bool = False,
    artifact_location: str | None = None,
) -> dict:
    """Log, register and (if allowed) promote a dummy model. Returns a summary."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name="dummy-model") as run:
        mlflow.set_tags({DUMMY_TAG[0]: DUMMY_TAG[1], "git_commit": git_commit()})
        mlflow.log_params({"constant_risk": 0.05, "window": WINDOW, "stride": STRIDE})
        mlflow.log_dict(Normalization.identity().to_dict(), NORMALIZATION_ARTIFACT)
        info = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT_PATH,
            python_model=str(MODEL_CODE),
            signature=MODEL_SIGNATURE,
            input_example=np.zeros((1, len(CHANNELS), WINDOW), dtype=np.float32),
            pip_requirements=[f"mlflow=={mlflow.__version__}", f"numpy=={np.__version__}"],
            registered_model_name=model_name,
        )
    version = str(info.registered_model_version)
    client.set_model_version_tag(model_name, version, *DUMMY_TAG)
    client.update_model_version(
        model_name, version, description="Placeholder model: constant risk 0.05 per window."
    )

    previous = current_alias_version(client, model_name, alias)
    previous_is_dummy = (
        previous is not None
        and client.get_model_version(model_name, previous).tags.get(DUMMY_TAG[0]) == DUMMY_TAG[1]
    )
    promoted = previous is None or previous_is_dummy or force_champion
    if promoted:
        client.set_registered_model_alias(model_name, alias, version)

    return {
        "experiment_id": experiment_id,
        "run_id": run.info.run_id,
        "artifact_uri": run.info.artifact_uri,
        "model_uri": f"models:/{model_name}/{version}",
        "version": version,
        "alias": alias,
        "alias_version": version if promoted else previous,
        "promoted": promoted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force-champion",
        action="store_true",
        help="move the alias even if the current champion is a real model",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.export_mlflow_env()  # S3 endpoint and keys for boto3
    summary = register_dummy_model(
        settings.mlflow_tracking_uri,
        settings.model_name,
        settings.model_alias,
        force_champion=args.force_champion,
    )
    print(json.dumps(summary, indent=2))
    if not summary["promoted"]:
        print(
            f"'{settings.model_alias}' still points to version {summary['alias_version']} "
            "(a real model); use --force-champion to override."
        )


if __name__ == "__main__":
    main()
