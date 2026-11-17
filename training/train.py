"""Training script for the 1D CNN risk detection model.

Contract & requirements (CONTRACTS.md section 2 and Fiche Ariane):
- Loads processed `.npz` files (train, val).
- Weighted loss using `pos_weight` to counter severe class imbalance.
- Early stopping on the validation set.
- Reproducibility with fixed random seed.
- Hyperparameters managed in `params.yaml`.
- Tracks every run in MLflow: parameters, metrics per epoch, loss curves, normalization.json.
- Registers model as PyFunc with signature in `gbg-risk` (without moving champion).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from gbg_mlops.config import get_settings
from gbg_mlops.features import CHANNELS, STRIDE, WINDOW, Normalization
from gbg_mlops.model_loader import (
    EXPERIMENT_NAME,
    MODEL_ARTIFACT_PATH,
    MODEL_SIGNATURE,
    NORMALIZATION_ARTIFACT,
)
from training.evaluate import evaluate_gbg500
from training.model import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_KERNEL_SIZE,
    RiskCNN1D,
    RiskPyfuncWrapper,
)


def git_commit() -> str:
    """Return the current Git commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def data_split_hash(split_path: str | Path = "data/processed/split.json") -> str:
    """Return SHA-256 hash of split.json if present."""
    p = Path(split_path)
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return "unknown"


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load configuration dictionary from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load X and y from processed .npz file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    data = np.load(data_path)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    return X, y


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute PR-AUC, ROC-AUC and F1 score robustly."""
    metrics: dict[str, float] = {}
    n_pos = int((y_true > 0.5).sum())
    n_neg = len(y_true) - n_pos

    # F1 score at threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC and PR-AUC require at least one sample of each class
    if n_pos > 0 and n_neg > 0:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = float(n_pos / len(y_true))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(n_pos / max(1, len(y_true)))

    return metrics


def plot_learning_curves(history: list[dict[str, float]]) -> plt.Figure:
    """Create a 2-panel plot showing loss and PR-AUC curves across epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_pr = [h["train_pr_auc"] for h in history]
    val_pr = [h["val_pr_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss subplot
    ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue", marker="o", markersize=3)
    ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange", marker="s", markersize=3)
    ax1.set_title("BCE Loss with pos_weight")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # PR-AUC subplot
    ax2.plot(epochs, train_pr, label="Train PR-AUC", color="tab:blue", marker="o", markersize=3)
    ax2.plot(epochs, val_pr, label="Val PR-AUC", color="tab:orange", marker="s", markersize=3)
    ax2.set_title("Precision-Recall AUC (Minority Class)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PR-AUC")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    return fig


class EarlyStopping:
    """Early stops training when validation PR-AUC does not improve."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def __call__(self, score: float, model: nn.Module) -> None:
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0


def get_or_create_experiment(tracking_uri: str, experiment_name: str) -> str:
    """Retrieve existing experiment ID or create a new one."""
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp:
        return exp.experiment_id
    return client.create_experiment(experiment_name)


def train_model(
    config: dict[str, Any],
    device: torch.device | None = None,
    track_mlflow: bool = True,
    register_model: bool = True,
    evaluate_gbg: bool = True,
    tracking_uri: str | None = None,
    run_name: str | None = None,
) -> tuple[RiskCNN1D, dict[str, Any]]:
    """Train RiskCNN1D, log to MLflow, evaluate on GBG500, and register in gbg-risk."""
    seed = config.get("seed", 42)
    seed_everything(seed)

    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    settings = get_settings()
    settings.export_mlflow_env()
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)

    # 1. Load data
    data_cfg = config.get("data", {})
    X_train, y_train = load_npz(data_cfg.get("train_path", "data/processed/train.npz"))
    X_val, y_val = load_npz(data_cfg.get("val_path", "data/processed/val.npz"))

    # Load normalisation
    norm_path = Path(data_cfg.get("normalization_path", "data/processed/normalization.json"))
    if norm_path.exists():
        normalization = Normalization.load(norm_path)
    else:
        normalization = Normalization.identity()

    # 2. Setup DataLoaders
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model initialization
    model_cfg = config.get("model", {})
    conv_channels = tuple(model_cfg.get("conv_channels", DEFAULT_CONV_CHANNELS))
    kernel_size = model_cfg.get("kernel_size", DEFAULT_KERNEL_SIZE)
    dropout = model_cfg.get("dropout", DEFAULT_DROPOUT)

    model = RiskCNN1D(
        in_channels=model_cfg.get("in_channels", 6),
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    # 4. Calculate pos_weight for BCEWithLogitsLoss
    configured_weight = train_cfg.get("pos_weight")
    if configured_weight is not None:
        pos_weight_val = float(configured_weight)
    else:
        n_pos = float((y_train > 0.5).sum())
        n_neg = float(len(y_train) - n_pos)
        pos_weight_val = n_neg / max(1.0, n_pos)

    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Optimizer
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. Early stopping setup
    patience = train_cfg.get("patience", 5)
    min_delta = train_cfg.get("min_delta", 0.001)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    epochs = train_cfg.get("epochs", 30)
    history: list[dict[str, float]] = []

    # 7. Setup MLflow run if tracking enabled
    active_mlflow_run = None
    if track_mlflow:
        mlflow.set_tracking_uri(uri)
        try:
            exp_id = get_or_create_experiment(uri, EXPERIMENT_NAME)
            active_mlflow_run = mlflow.start_run(
                experiment_id=exp_id, run_name=run_name or "cnn1d-train"
            )
            mlflow.set_tags(
                {
                    "model_type": "cnn1d",
                    "git_commit": git_commit(),
                    "data_split": data_split_hash(),
                }
            )
            mlflow.log_params(
                {
                    "seed": seed,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "epochs_max": epochs,
                    "patience": patience,
                    "min_delta": min_delta,
                    "pos_weight": round(pos_weight_val, 4),
                    "conv_channels": str(conv_channels),
                    "kernel_size": kernel_size,
                    "dropout": dropout,
                    "window": WINDOW,
                    "stride": STRIDE,
                }
            )
            # Log normalization.json artifact
            mlflow.log_dict(normalization.to_dict(), NORMALIZATION_ARTIFACT)
        except Exception as e:
            print(f"Warning: MLflow tracking setup failed ({e}). Proceeding without MLflow.")
            active_mlflow_run = None

    registered_version: str | None = None
    gbg_metrics: dict[str, Any] | None = None

    try:
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_losses: list[float] = []
            all_train_preds: list[np.ndarray] = []
            all_train_targets: list[np.ndarray] = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                with torch.no_grad():
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_train_preds.append(probs)
                    all_train_targets.append(batch_y.cpu().numpy())

            train_loss = float(np.mean(train_losses))
            train_metrics = compute_metrics(
                np.concatenate(all_train_targets), np.concatenate(all_train_preds)
            )

            # Validation phase
            model.eval()
            val_losses: list[float] = []
            all_val_preds: list[np.ndarray] = []
            all_val_targets: list[np.ndarray] = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_losses.append(loss.item())

                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_val_preds.append(probs)
                    all_val_targets.append(batch_y.cpu().numpy())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            val_metrics = compute_metrics(
                np.concatenate(all_val_targets), np.concatenate(all_val_preds)
            )

            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pr_auc": train_metrics["pr_auc"],
                "train_roc_auc": train_metrics["roc_auc"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_loss,
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_f1": val_metrics["f1"],
            }
            history.append(epoch_stats)

            if active_mlflow_run:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_pr_auc": train_metrics["pr_auc"],
                        "train_roc_auc": train_metrics["roc_auc"],
                        "train_f1": train_metrics["f1"],
                        "val_loss": val_loss,
                        "val_pr_auc": val_metrics["pr_auc"],
                        "val_roc_auc": val_metrics["roc_auc"],
                        "val_f1": val_metrics["f1"],
                    },
                    step=epoch,
                )

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} PR-AUC: {train_metrics['pr_auc']:.4f} | "
                f"Val Loss: {val_loss:.4f} PR-AUC: {val_metrics['pr_auc']:.4f}"
            )

            early_stopping(val_metrics["pr_auc"], model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Restore best weights and persist checkpoint
        if early_stopping.best_state_dict is not None:
            model.load_state_dict(early_stopping.best_state_dict)

        weights_path = Path("training/best_model.pt")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)

        # Log curves and final metrics
        if active_mlflow_run:
            fig = plot_learning_curves(history)
            mlflow.log_figure(fig, "curves.png")
            plt.close(fig)

            mlflow.log_metrics(
                {
                    "best_val_pr_auc": early_stopping.best_score or 0.0,
                    "epochs_trained": len(history),
                }
            )

        # Optional: Evaluation on GBG500 logged in the same run
        if evaluate_gbg:
            gbg_npz = Path("data/processed/gbg_eval.npz")
            gbg_labels = Path("data/processed/gbg_eval_labels.csv")
            if gbg_npz.exists() and gbg_labels.exists():
                try:
                    gbg_metrics = evaluate_gbg500(model, gbg_npz, gbg_labels)
                    if active_mlflow_run:
                        from gbg_mlops.model_loader import EVALUATION_METRIC
                        from training.evaluate import plot_gbg_roc_curve

                        mlflow.log_metric(EVALUATION_METRIC, gbg_metrics[EVALUATION_METRIC])
                        mlflow.log_metric("gbg_reckless_pr_auc", gbg_metrics["gbg_reckless_pr_auc"])
                        mlflow.log_metric(
                            "gbg_random_baseline_auc", gbg_metrics["gbg_random_baseline_auc"]
                        )
                        mlflow.log_metric(
                            "gbg_auc_lift_vs_random", gbg_metrics["gbg_auc_lift_vs_random"]
                        )
                        mlflow.log_metric("gbg_beats_random", int(gbg_metrics["beats_random"]))
                        mlflow.log_dict(gbg_metrics, "gbg_evaluation.json")

                        y_t = np.array([r["is_reckless"] for r in gbg_metrics["rides"]])
                        y_s = np.array([r["risk_score"] for r in gbg_metrics["rides"]])
                        roc_fig = plot_gbg_roc_curve(y_t, y_s, gbg_metrics[EVALUATION_METRIC])
                        mlflow.log_figure(roc_fig, "gbg_roc_curve.png")
                        plt.close(roc_fig)
                except Exception as e:
                    print(f"Warning: GBG evaluation failed ({e})")

        # Step 7: Register model as PyFunc in Model Registry
        if register_model and active_mlflow_run:
            print(f"Registering model in MLflow Registry as '{settings.model_name}'...")
            info = mlflow.pyfunc.log_model(
                name=MODEL_ARTIFACT_PATH,
                python_model=RiskPyfuncWrapper(model),
                signature=MODEL_SIGNATURE,
                input_example=np.zeros((1, len(CHANNELS), WINDOW), dtype=np.float32),
                registered_model_name=settings.model_name,
            )
            if info.registered_model_version is not None:
                registered_version = str(info.registered_model_version)
                client = MlflowClient(tracking_uri=uri)
                client.set_model_version_tag(
                    settings.model_name, registered_version, "model_type", "cnn1d"
                )
                client.set_model_version_tag(
                    settings.model_name, registered_version, "git_commit", git_commit()
                )
                client.set_model_version_tag(
                    settings.model_name, registered_version, "data_split", data_split_hash()
                )
                desc = (
                    f"1D CNN risk detection model. "
                    f"Val PR-AUC: {early_stopping.best_score or 0.0:.4f}. "
                )
                if gbg_metrics:
                    desc += f"GBG500 AUC: {gbg_metrics['gbg_reckless_auc']:.4f}."
                client.update_model_version(
                    settings.model_name, registered_version, description=desc
                )

    finally:
        if active_mlflow_run:
            mlflow.end_run()

    summary = {
        "pos_weight": pos_weight_val,
        "epochs_trained": len(history),
        "best_val_pr_auc": early_stopping.best_score or 0.0,
        "history": history,
        "run_id": active_mlflow_run.info.run_id if active_mlflow_run else None,
        "registered_version": registered_version,
        "gbg_metrics": gbg_metrics,
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    parser.add_argument("--run-name", default=None, help="Name for the MLflow run")
    parser.add_argument("--tracking-uri", default=None, help="Explicit MLflow tracking URI")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override pos_weight")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--kernel-size", type=int, default=None, help="Override kernel size")
    parser.add_argument(
        "--conv-channels",
        type=str,
        default=None,
        help="Comma-separated channels e.g. 64,128,256",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--no-register", action="store_true", help="Do not register in registry")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.pos_weight is not None:
        config.setdefault("training", {})["pos_weight"] = args.pos_weight
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.kernel_size is not None:
        config.setdefault("model", {})["kernel_size"] = args.kernel_size
    if args.conv_channels is not None:
        channels = [int(c.strip()) for c in args.conv_channels.split(",")]
        config.setdefault("model", {})["conv_channels"] = channels

    print(f"Starting training with config: {args.config}")
    model, summary = train_model(
        config,
        track_mlflow=not args.no_mlflow,
        register_model=not args.no_register,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
    )
    print(f"Training finished: {summary['epochs_trained']} epochs.")
    print(f"Best validation PR-AUC: {summary['best_val_pr_auc']:.4f}")
    if summary.get("gbg_metrics"):
        print(f"GBG500 Reckless AUC: {summary['gbg_metrics']['gbg_reckless_auc']:.4f}")
        print(f"Beats random baseline: {summary['gbg_metrics']['beats_random']}")
    if summary.get("run_id"):
        print(f"MLflow Run ID: {summary['run_id']}")
    if summary.get("registered_version"):
        ver = summary["registered_version"]
        print(f"Registered model version: {ver}")
        print(f"-> To promote this version to champion, run: make promote VERSION={ver}")


if __name__ == "__main__":
    main()
