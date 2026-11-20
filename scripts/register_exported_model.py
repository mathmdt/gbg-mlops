"""Register a pre-trained CNN model into MLflow and point champion alias to it.

Usage:
    uv run python scripts/register_exported_model.py
    uv run python scripts/register_exported_model.py --model-dir exported_model --force-champion
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from gbg_mlops.config import get_settings
from gbg_mlops.features import CHANNELS, WINDOW
from gbg_mlops.model import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_KERNEL_SIZE,
    RiskCNN1D,
    RiskPyfuncWrapper,
)
from gbg_mlops.model_loader import (
    EXPERIMENT_NAME,
    MODEL_ARTIFACT_PATH,
    MODEL_SIGNATURE,
)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def register_exported_model(
    model_dir: str | Path,
    tracking_uri: str,
    model_name: str,
    alias: str = "champion",
    force_champion: bool = True,
) -> dict[str, Any]:
    model_path = Path(model_dir)
    weights_path = model_path / "weights.pt"
    config_path = model_path / "config.json"
    norm_path = model_path / "normalization.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {weights_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization file at {norm_path}")

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    in_channels = config.get("in_channels", len(CHANNELS))
    conv_channels = tuple(config.get("conv_channels", DEFAULT_CONV_CHANNELS))
    kernel_size = config.get("kernel_size", DEFAULT_KERNEL_SIZE)
    dropout = config.get("dropout", DEFAULT_DROPOUT)

    model = RiskCNN1D(
        in_channels=in_channels,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = (
            experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT_NAME)
        )
    except Exception:
        experiment = client.get_experiment("0")
        experiment_id = experiment.experiment_id if experiment else "0"

    with mlflow.start_run(experiment_id=experiment_id, run_name="import-cnn1d") as run:
        mlflow.set_tags(
            {
                "model_type": "cnn1d",
                "git_commit": git_commit(),
                "imported": "true",
            }
        )

        eval_json_path = model_path / "gbg_evaluation.json"
        if eval_json_path.exists():
            with open(eval_json_path) as f:
                metrics = json.load(f)
            if "gbg_reckless_auc" in metrics:
                mlflow.log_metric("gbg_reckless_auc", float(metrics["gbg_reckless_auc"]))
            mlflow.log_artifact(str(eval_json_path))

        curves_path = model_path / "curves.png"
        if curves_path.exists():
            mlflow.log_artifact(str(curves_path))

        roc_path = model_path / "gbg_roc_curve.png"
        if roc_path.exists():
            mlflow.log_artifact(str(roc_path))

        mlflow.log_artifact(str(norm_path))

        input_example = np.zeros((1, in_channels, WINDOW), dtype=np.float32)
        info = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT_PATH,
            python_model=RiskPyfuncWrapper(model),
            signature=MODEL_SIGNATURE,
            input_example=input_example,
            registered_model_name=model_name,
        )

    version = str(info.registered_model_version)
    client.set_model_version_tag(model_name, version, "model_type", "cnn1d")
    client.update_model_version(
        model_name, version, description="Trained 1D CNN risk model (imported)."
    )

    if force_champion:
        client.set_registered_model_alias(model_name, alias, version)

    return {
        "status": "success",
        "model_name": model_name,
        "version": version,
        "alias": alias,
        "run_id": run.info.run_id,
        "promoted_champion": force_champion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        type=str,
        default="exported_model",
        help="directory containing weights.pt and normalization.json (default: exported_model)",
    )
    parser.add_argument(
        "--force-champion",
        action="store_true",
        default=True,
        help="set champion alias to this registered version (default: True)",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.export_mlflow_env()
    summary = register_exported_model(
        model_dir=args.model_dir,
        tracking_uri=settings.mlflow_tracking_uri,
        model_name=settings.model_name,
        alias=settings.model_alias,
        force_champion=args.force_champion,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
