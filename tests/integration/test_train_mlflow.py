"""Integration test verifying MLflow tracking during training.

Creates self-contained temporary datasets so it runs independently in CI without data/processed.
"""

from __future__ import annotations

import importlib.util

import mlflow
import numpy as np
import pytest

from gbg_mlops.features import CHANNELS, WINDOW, Normalization
from gbg_mlops.model_loader import NORMALIZATION_ARTIFACT
from training.train import train_model

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed"),
    pytest.mark.skipif(
        importlib.util.find_spec("mlflow") is None, reason="mlflow is not installed"
    ),
]


@pytest.fixture
def temp_mlflow_env(tmp_path):
    """Setup temporary MLflow database and synthetic datasets."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)

    # Generate small self-contained train.npz and val.npz
    rng = np.random.default_rng(42)
    n_train, n_val = 20, 10

    X_train = rng.normal(size=(n_train, len(CHANNELS), WINDOW)).astype(np.float32)
    y_train = np.zeros(n_train, dtype=np.int64)
    y_train[:3] = 1  # 3 positive samples

    X_val = rng.normal(size=(n_val, len(CHANNELS), WINDOW)).astype(np.float32)
    y_val = np.zeros(n_val, dtype=np.int64)
    y_val[:2] = 1  # 2 positive samples

    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    norm_path = tmp_path / "normalization.json"

    np.savez_compressed(train_path, X=X_train, y=y_train)
    np.savez_compressed(val_path, X=X_val, y=y_val)
    Normalization.identity().save(norm_path)

    config = {
        "seed": 42,
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "normalization_path": str(norm_path),
        },
        "model": {
            "in_channels": len(CHANNELS),
            "conv_channels": [16, 32, 64],
            "kernel_size": 3,
            "dropout": 0.1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 2,
            "min_delta": 0.001,
            "pos_weight": None,
        },
    }

    yield uri, config
    mlflow.set_tracking_uri(None)


def test_train_logs_to_mlflow(temp_mlflow_env):
    uri, config = temp_mlflow_env

    _, summary = train_model(
        config,
        track_mlflow=True,
        register_model=False,
        evaluate_gbg=False,
        tracking_uri=uri,
        run_name="test-integration-run",
    )

    assert summary["run_id"] is not None
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(summary["run_id"])

    # Check status and tags
    assert run.info.status == "FINISHED"
    assert run.data.tags.get("model_type") == "cnn1d"

    # Check params
    assert "pos_weight" in run.data.params
    assert "learning_rate" in run.data.params
    assert "conv_channels" in run.data.params

    # Check metrics
    assert "train_loss" in run.data.metrics
    assert "val_pr_auc" in run.data.metrics
    assert "best_val_pr_auc" in run.data.metrics

    # Check artifacts
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert NORMALIZATION_ARTIFACT in artifacts
    assert "curves.png" in artifacts
