"""1D Convolutional Neural Network for IMU window risk scoring.

Follows the model contract in CONTRACTS.md (section 2):
- Input: normalised windows of shape `(n_windows, 6, 300)` float32.
- Output: 1D array of shape `(n_windows,)` with risk probabilities in `[0, 1]`.
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import torch
import torch.nn as nn

from gbg_mlops.features import CHANNELS, WINDOW

DEFAULT_CONV_CHANNELS = (32, 64, 128)
DEFAULT_KERNEL_SIZE = 5
DEFAULT_DROPOUT = 0.2


class RiskCNN1D(nn.Module):
    """3-layer 1D CNN with Conv1d -> BatchNorm -> ReLU -> pooling blocks."""

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        conv_channels: tuple[int, int, int] = DEFAULT_CONV_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels
        padding = kernel_size // 2

        # Layer 1: Conv1d -> BatchNorm -> ReLU -> MaxPool1d
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 2: Conv1d -> BatchNorm -> ReLU -> MaxPool1d
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 3: Conv1d -> BatchNorm -> ReLU -> AdaptiveAvgPool1d
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Output head: Linear logit for numerical stability with BCEWithLogitsLoss
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits of shape `(batch_size,)` for input `(batch_size, 6, 300)`."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)  # (batch_size, c3, 1)
        x = torch.flatten(x, start_dim=1)  # (batch_size, c3)
        logits = self.head(x).squeeze(-1)  # (batch_size,)
        return logits

    @torch.no_grad()
    def predict_proba(self, windows: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return risk probability per window in `[0, 1]` with shape `(n_windows,)`."""
        self.eval()
        if isinstance(windows, np.ndarray):
            tensor = torch.from_numpy(windows.astype(np.float32, copy=False))
        else:
            tensor = windows.float()

        device = next(self.parameters()).device
        tensor = tensor.to(device)

        logits = self.forward(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        return probs.reshape(-1)


class RiskPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper conforming to CONTRACTS.md Section 2."""

    def __init__(self, model: RiskCNN1D | None = None) -> None:
        self.model = model

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load PyTorch weights from the MLflow artifact directory if artifacts are present."""
        if hasattr(self, "model") and self.model is not None:
            self.model.eval()
            return

        import os

        artifacts = getattr(context, "artifacts", {}) or {}
        if "weights" in artifacts:
            weights_path = artifacts["weights"]
            config_path = artifacts.get("config", "")

            config = {}
            if os.path.exists(config_path):
                import json

                with open(config_path) as f:
                    config = json.load(f)

            self.model = RiskCNN1D(
                in_channels=config.get("in_channels", len(CHANNELS)),
                conv_channels=tuple(config.get("conv_channels", DEFAULT_CONV_CHANNELS)),
                kernel_size=config.get("kernel_size", DEFAULT_KERNEL_SIZE),
                dropout=config.get("dropout", DEFAULT_DROPOUT),
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | np.ndarray | None = None,
        model_input: np.ndarray | None = None,
        params: dict | None = None,
    ) -> np.ndarray:
        """Return risk probability array `(n_windows,)` for `(n_windows, 6, 300)` input."""
        # Support both direct calls: predict(windows) and MLflow calls: predict(context, windows)
        if model_input is None and context is not None:
            model_input = context

        if model_input is None:
            raise ValueError("model_input must be provided")

        windows = np.asarray(model_input, dtype=np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (len(CHANNELS), WINDOW):
            raise ValueError(
                f"expected (n, {len(CHANNELS)}, {WINDOW}) windows, got {windows.shape}"
            )
        if not np.isfinite(windows).all():
            raise ValueError("input contains non-finite values (NaN or Inf)")

        if self.model is None:
            raise RuntimeError("PyTorch model is not initialized or loaded")

        probs = self.model.predict_proba(windows)
        if probs.shape != (len(windows),):
            raise RuntimeError(f"expected {len(windows)} probabilities, got {probs.shape[0]}")
        return probs
