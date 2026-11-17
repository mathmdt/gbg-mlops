"""Integration test verifying MLflow tracking during training."""

import mlflow
import pytest

from gbg_mlops.model_loader import NORMALIZATION_ARTIFACT
from training.train import load_config, train_model


@pytest.fixture
def temp_mlflow_store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    yield uri
    mlflow.set_tracking_uri(None)


def test_train_logs_to_mlflow(temp_mlflow_store):
    config = load_config("params.yaml")
    config["training"]["epochs"] = 2
    config["training"]["patience"] = 2

    _, summary = train_model(
        config,
        track_mlflow=True,
        tracking_uri=temp_mlflow_store,
        run_name="test-integration-run",
    )

    assert summary["run_id"] is not None
    client = mlflow.tracking.MlflowClient(tracking_uri=temp_mlflow_store)
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
