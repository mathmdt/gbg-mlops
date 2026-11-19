"""Run and compare at least 3 model variations in MLflow (Fiche Ariane step 8).

Comparisons explored:
1. Baseline CNN 1D: kernel_size=5, channels=[32, 64, 128], auto pos_weight (~8.5)
2. Loss weighting ablation: pos_weight=1.0 (unweighted BCE: tests class imbalance impact)
3. Architecture variation: kernel_size=7, channels=[64, 128, 256], pos_weight=15.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gbg_mlops.config import get_settings  # noqa: E402
from gbg_mlops.model_loader import EXPERIMENT_NAME  # noqa: E402
from training.train import load_config, train_model  # noqa: E402

EXPERIMENTS = [
    {
        "name": "run-1-baseline-posweight-auto",
        "description": "Baseline 3-layer CNN 1D with auto pos_weight",
        "model": {"kernel_size": 5, "conv_channels": [32, 64, 128], "dropout": 0.2},
        "training": {"pos_weight": None, "learning_rate": 0.001, "epochs": 20},
    },
    {
        "name": "run-2-ablation-no-posweight",
        "description": "Ablation: unweighted loss (pos_weight=1.0) showing imbalance effect",
        "model": {"kernel_size": 5, "conv_channels": [32, 64, 128], "dropout": 0.2},
        "training": {"pos_weight": 1.0, "learning_rate": 0.001, "epochs": 20},
    },
    {
        "name": "run-3-deep-kernel7",
        "description": "Deeper capacity and larger receptive field (kernel 7, channels 64-128-256)",
        "model": {"kernel_size": 7, "conv_channels": [64, 128, 256], "dropout": 0.3},
        "training": {"pos_weight": 15.0, "learning_rate": 0.0008, "epochs": 20},
    },
]


def run_comparisons(config_path: str = "params.yaml") -> list[dict]:
    settings = get_settings()
    settings.export_mlflow_env()
    base_config = load_config(config_path)

    results = []
    print("\n=======================================================")
    print(f"Launching 3 Experiment Runs in MLflow ({EXPERIMENT_NAME})")
    print("=======================================================\n")

    for i, exp in enumerate(EXPERIMENTS, start=1):
        print(f"\n[{i}/3] Running experiment: {exp['name']}")
        print(f"    Description: {exp['description']}")

        # Clone and update config
        cfg = json.loads(json.dumps(base_config))
        cfg.setdefault("model", {}).update(exp["model"])
        cfg.setdefault("training", {}).update(exp["training"])

        model, summary = train_model(
            cfg,
            track_mlflow=True,
            register_model=True,
            evaluate_gbg=True,
            run_name=exp["name"],
        )

        gbg_auc = summary["gbg_metrics"]["gbg_reckless_auc"] if summary.get("gbg_metrics") else None
        beats_random = (
            summary["gbg_metrics"]["beats_random"] if summary.get("gbg_metrics") else False
        )

        row = {
            "run_name": exp["name"],
            "run_id": summary.get("run_id"),
            "kernel_size": exp["model"]["kernel_size"],
            "channels": str(exp["model"]["conv_channels"]),
            "pos_weight": summary["pos_weight"],
            "epochs_trained": summary["epochs_trained"],
            "best_val_pr_auc": round(summary["best_val_pr_auc"], 4),
            "gbg_reckless_auc": gbg_auc,
            "beats_random": beats_random,
            "registered_version": summary.get("registered_version"),
        }
        results.append(row)

    print("\n=======================================================")
    print("                COMPARISON SUMMARY                     ")
    print("=======================================================")
    header = (
        f"{'Run Name':<30} | {'Kernel':<6} | {'PosWt':<6} | "
        f"{'Val PR-AUC':<10} | {'GBG AUC':<8} | {'Version':<7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        v_str = str(r["registered_version"]) if r["registered_version"] else "-"
        auc_str = f"{r['gbg_reckless_auc']:.4f}" if r["gbg_reckless_auc"] is not None else "-"
        print(
            f"{r['run_name']:<30} | {r['kernel_size']:<6} | {r['pos_weight']:<6.2f} | "
            f"{r['best_val_pr_auc']:<10.4f} | {auc_str:<8} | {v_str:<7}"
        )
    print("=======================================================\n")
    print("View and compare runs visually on: http://localhost:5001")
    return results


if __name__ == "__main__":
    run_comparisons()
