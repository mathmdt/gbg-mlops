"""Unit and integration tests for GBG500 evaluation (training/evaluate.py)."""

from __future__ import annotations

import importlib.util

import mlflow
import numpy as np
import pytest

from gbg_mlops.model_loader import EVALUATION_METRIC
from training.evaluate import evaluate_gbg500, log_to_mlflow
from training.model import RiskCNN1D

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


@pytest.fixture
def eval_files(tmp_path):
    """Create miniature gbg_eval.npz and gbg_eval_labels.csv for testing."""
    npz_path = tmp_path / "gbg_eval.npz"
    labels_path = tmp_path / "gbg_eval_labels.csv"

    # 4 rides: 1 Reckless, 3 Safe
    X = np.random.randn(8, 6, 300).astype(np.float32)
    ride_id = np.array(["r0", "r0", "r1", "r1", "r2", "r2", "r3", "r3"])
    np.savez_compressed(npz_path, X=X, ride_id=ride_id)

    labels_csv = "ride_id,label\nr0,Safe\nr1,Reckless\nr2,Safe\nr3,Safe\n"
    labels_path.write_text(labels_csv)

    return npz_path, labels_path


def test_evaluate_gbg500_computes_metrics(eval_files):
    npz_path, labels_path = eval_files
    model = RiskCNN1D()

    results = evaluate_gbg500(model, npz_path=npz_path, labels_path=labels_path)

    assert EVALUATION_METRIC in results
    assert "gbg_random_baseline_auc" in results
    assert "beats_random" in results
    assert results["n_rides_total"] == 4
    assert results["n_reckless"] == 1
    assert "Safe" in results["class_summary"]
    assert "Reckless" in results["class_summary"]
    assert len(results["rides"]) == 4


def test_evaluate_gbg500_logs_to_mlflow(eval_files, tmp_path):
    npz_path, labels_path = eval_files
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)

    # Create dummy experiment and run
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exp_id = client.create_experiment("gbg-risk")
    run = client.create_run(experiment_id=exp_id)

    model = RiskCNN1D()
    results = evaluate_gbg500(model, npz_path=npz_path, labels_path=labels_path)

    # Attach to existing run
    logged_run_id = log_to_mlflow(results, run_id=run.info.run_id, tracking_uri=uri)
    assert logged_run_id == run.info.run_id

    # Verify run metrics and artifacts
    run_data = client.get_run(logged_run_id).data
    assert EVALUATION_METRIC in run_data.metrics
    assert "gbg_random_baseline_auc" in run_data.metrics

    artifacts = [a.path for a in client.list_artifacts(logged_run_id)]
    assert "gbg_evaluation.json" in artifacts
    assert "gbg_roc_curve.png" in artifacts

    mlflow.set_tracking_uri(None)
