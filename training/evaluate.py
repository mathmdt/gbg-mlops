"""Evaluation script on GBG500 dataset (CONTRACTS.md section 2 and Fiche Ariane).

- Evaluates risk score per ride on GBG500 (`aggregate_ride_score` = max window probability).
- Computes mandatory ROC-AUC metric: Reckless (positive) vs all other rides.
- Compares against random score baseline (expected AUC ~ 0.50).
- Logs evaluation metrics and artifacts into the corresponding MLflow run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from gbg_mlops.config import get_settings
from gbg_mlops.features import aggregate_ride_score
from gbg_mlops.model_loader import EVALUATION_METRIC, EXPERIMENT_NAME
from training.model import DEFAULT_CONV_CHANNELS, DEFAULT_DROPOUT, DEFAULT_KERNEL_SIZE, RiskCNN1D


def load_gbg_data(
    npz_path: str | Path = "data/processed/gbg_eval.npz",
    labels_path: str | Path = "data/processed/gbg_eval_labels.csv",
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Load GBG500 windows, ride IDs and labels mapping."""
    npz_file = Path(npz_path)
    if not npz_file.exists():
        raise FileNotFoundError(f"GBG evaluation file not found: {npz_file}")
    data = np.load(npz_file)
    X = np.asarray(data["X"], dtype=np.float32)
    ride_ids = np.asarray(data["ride_id"], dtype=str)

    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"GBG labels file not found: {labels_file}")

    labels_map: dict[str, str] = {}
    with open(labels_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip CSV header
        for row in reader:
            if len(row) >= 2:
                labels_map[row[0].strip()] = row[1].strip()

    return X, ride_ids, labels_map


def load_model(
    weights_path: str | Path = "training/best_model.pt",
    device: torch.device | None = None,
) -> RiskCNN1D:
    """Load RiskCNN1D model with trained weights."""
    if device is None:
        device = torch.device("cpu")

    model = RiskCNN1D(
        in_channels=6,
        conv_channels=DEFAULT_CONV_CHANNELS,
        kernel_size=DEFAULT_KERNEL_SIZE,
        dropout=DEFAULT_DROPOUT,
    )

    path = Path(weights_path)
    if path.exists():
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: weights file {path} not found. Using initialized model.")

    model.to(device)
    model.eval()
    return model


def evaluate_gbg500(
    model: RiskCNN1D,
    npz_path: str | Path = "data/processed/gbg_eval.npz",
    labels_path: str | Path = "data/processed/gbg_eval_labels.csv",
    seed: int = 42,
) -> dict[str, Any]:
    """Score GBG500 rides, compute ROC-AUC Reckless vs rest and compare with random baseline."""
    X, ride_ids, labels_map = load_gbg_data(npz_path, labels_path)

    # Unique rides
    unique_rides = sorted(set(ride_ids.tolist()))
    if not unique_rides:
        raise ValueError("No rides found in GBG evaluation data")

    ride_scores: list[float] = []
    y_true: list[int] = []
    ride_labels: list[str] = []
    ride_details: list[dict[str, Any]] = []

    for ride in unique_rides:
        indices = np.flatnonzero(ride_ids == ride)
        windows = X[indices]
        window_probs = model.predict_proba(windows)
        score = aggregate_ride_score(window_probs)
        label = labels_map.get(ride, "Safe")
        is_reckless = int(label.lower() == "reckless")

        ride_scores.append(score)
        y_true.append(is_reckless)
        ride_labels.append(label)
        ride_details.append(
            {
                "ride_id": ride,
                "label": label,
                "is_reckless": is_reckless,
                "risk_score": round(score, 4),
                "n_windows": len(windows),
            }
        )

    y_arr = np.asarray(y_true, dtype=int)
    scores_arr = np.asarray(ride_scores, dtype=np.float64)

    # Calculate ROC-AUC and PR-AUC
    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos

    if n_pos > 0 and n_neg > 0:
        reckless_auc = float(roc_auc_score(y_arr, scores_arr))
        reckless_pr_auc = float(average_precision_score(y_arr, scores_arr))
    else:
        # Fallback if only one class is present in a tiny slice
        reckless_auc = 0.5
        reckless_pr_auc = float(n_pos / max(1, len(y_arr)))

    # Compare with random baseline
    rng = np.random.default_rng(seed)
    n_simulations = 200
    random_aucs = []
    if n_pos > 0 and n_neg > 0:
        for _ in range(n_simulations):
            rand_scores = rng.uniform(0.0, 1.0, size=len(y_arr))
            random_aucs.append(roc_auc_score(y_arr, rand_scores))
        random_baseline_auc = float(np.mean(random_aucs))
    else:
        random_baseline_auc = 0.5

    beats_random = reckless_auc > random_baseline_auc
    auc_lift = reckless_auc - random_baseline_auc

    # Breakdown of scores per class
    class_scores: dict[str, list[float]] = {}
    for detail in ride_details:
        class_scores.setdefault(detail["label"], []).append(detail["risk_score"])

    class_summary = {
        cls: {
            "count": len(vals),
            "mean_score": round(float(np.mean(vals)), 4),
            "max_score": round(float(np.max(vals)), 4),
        }
        for cls, vals in class_scores.items()
    }

    results = {
        EVALUATION_METRIC: round(reckless_auc, 4),
        "gbg_reckless_pr_auc": round(reckless_pr_auc, 4),
        "gbg_random_baseline_auc": round(random_baseline_auc, 4),
        "gbg_auc_lift_vs_random": round(auc_lift, 4),
        "beats_random": beats_random,
        "n_rides_total": len(unique_rides),
        "n_reckless": n_pos,
        "class_summary": class_summary,
        "rides": ride_details,
    }
    return results


def plot_gbg_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, auc_val: float) -> plt.Figure:
    """Generate ROC curve plot for GBG Reckless detection."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"1D CNN (AUC = {auc_val:.3f})")
    ax.plot(
        [0, 1],
        [0, 1],
        color="tab:gray",
        lw=1.5,
        linestyle="--",
        label="Random baseline (AUC = 0.500)",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (Safe, Tandem, Bad weather)")
    ax.set_ylabel("True Positive Rate (Reckless)")
    ax.set_title("ROC Curve — GBG500 Reckless Detection")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def log_to_mlflow(
    results: dict[str, Any],
    run_id: str | None = None,
    tracking_uri: str | None = None,
) -> str | None:
    """Log GBG evaluation metrics and artifacts to the target MLflow run."""
    settings = get_settings()
    settings.export_mlflow_env()
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    target_run_id = run_id
    if target_run_id is None:
        # Search for latest run in experiment
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1,
                order_by=["start_time DESC"],
            )
            if runs:
                target_run_id = runs[0].info.run_id

    if target_run_id is None:
        print("Notice: No MLflow run found to attach GBG evaluation to.")
        return None

    with mlflow.start_run(run_id=target_run_id):
        # Mandatory metric for promotion
        mlflow.log_metric(EVALUATION_METRIC, results[EVALUATION_METRIC])
        mlflow.log_metric("gbg_reckless_pr_auc", results["gbg_reckless_pr_auc"])
        mlflow.log_metric("gbg_random_baseline_auc", results["gbg_random_baseline_auc"])
        mlflow.log_metric("gbg_auc_lift_vs_random", results["gbg_auc_lift_vs_random"])
        mlflow.log_metric("gbg_beats_random", int(results["beats_random"]))

        # Log class mean scores
        for cls_name, stats in results.get("class_summary", {}).items():
            metric_key = f"gbg_score_mean_{cls_name.lower().replace(' ', '_')}"
            mlflow.log_metric(metric_key, stats["mean_score"])

        # Log JSON report artifact
        mlflow.log_dict(results, "gbg_evaluation.json")

        # Generate ROC curve plot artifact
        y_true = np.array([r["is_reckless"] for r in results["rides"]])
        y_scores = np.array([r["risk_score"] for r in results["rides"]])
        fig = plot_gbg_roc_curve(y_true, y_scores, results[EVALUATION_METRIC])
        mlflow.log_figure(fig, "gbg_roc_curve.png")
        plt.close(fig)

    print(f"Logged GBG evaluation to MLflow run {target_run_id}:")
    print(f"  {EVALUATION_METRIC} = {results[EVALUATION_METRIC]}")
    print(f"  Random baseline AUC = {results['gbg_random_baseline_auc']}")
    print(f"  Beats random: {results['beats_random']}")
    return target_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default="data/processed/gbg_eval.npz", help="Path to GBG eval npz")
    parser.add_argument(
        "--labels", default="data/processed/gbg_eval_labels.csv", help="Path to GBG labels CSV"
    )
    parser.add_argument("--weights", default="training/best_model.pt", help="Path to model weights")
    parser.add_argument("--run-id", default=None, help="MLflow run ID to attach metrics to")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    model = load_model(args.weights)
    results = evaluate_gbg500(model, npz_path=args.npz, labels_path=args.labels)

    print(
        json.dumps(
            {
                EVALUATION_METRIC: results[EVALUATION_METRIC],
                "gbg_random_baseline_auc": results["gbg_random_baseline_auc"],
                "beats_random": results["beats_random"],
                "class_summary": results["class_summary"],
            },
            indent=2,
        )
    )

    if not args.no_mlflow:
        log_to_mlflow(results, run_id=args.run_id, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
