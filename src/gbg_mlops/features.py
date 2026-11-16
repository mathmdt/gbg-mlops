"""Preprocessing shared by the pipelines, training and the API.

Single source of truth for channel order, units, cleaning, windowing,
normalisation and ride-level aggregation, so that a ride is transformed
identically everywhere. Contract: see `CONTRACTS.md`, section 1.

Array conventions:
- a raw ride is `(n_points, 6)`, channels in `CHANNELS` order;
- windows are `(n_windows, 6, WINDOW)`, float32, channels first (PyTorch Conv1d).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Defaults agreed in CONTRACTS.md. They can be changed, but only together with
# the processed data and a retrained model.
CHANNELS: tuple[str, ...] = ("ax", "ay", "az", "rx", "ry", "rz")
ACC_CHANNELS = slice(0, 3)  # m/s²
GYRO_CHANNELS = slice(3, 6)  # rad/s inside the project
SAMPLE_PERIOD_MS = 100  # 10 Hz
WINDOW = 300  # points per window (30 s)
STRIDE = 150  # points between window starts (50 % overlap)
DEG_PER_RAD = 180 / math.pi  # ≈ 57.2958

Units = Literal["rad_s", "deg_s"]


def to_rad_s(values: np.ndarray, units: Units) -> np.ndarray:
    """Return a copy of a `(n, 6)` ride with the gyroscope expressed in rad/s.

    SRV3490 is recorded in rad/s, GBG500 in °/s.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    if out.ndim != 2 or out.shape[1] != len(CHANNELS):
        raise ValueError(f"expected shape (n, {len(CHANNELS)}), got {out.shape}")
    if units == "deg_s":
        out[:, GYRO_CHANNELS] /= DEG_PER_RAD
    elif units != "rad_s":
        raise ValueError(f"unknown gyroscope units: {units!r}")
    return out


def clean_ride(time_index: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time, drop duplicated timestamps and fill missing values.

    - duplicated `time_index`: the first occurrence is kept;
    - after deduplication the step must be exactly `SAMPLE_PERIOD_MS`;
    - missing values (NaN) are linearly interpolated per channel, with the
      nearest valid value at the edges. A channel with no valid value is an error.
    """
    time_index = np.asarray(time_index, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if time_index.ndim != 1 or values.shape != (len(time_index), len(CHANNELS)):
        raise ValueError("time_index must be (n,) and values (n, 6)")
    if np.isinf(values).any():
        raise ValueError("infinite values are not allowed")
    order = np.argsort(time_index, kind="stable")
    time_index, values = time_index[order], values[order]
    keep = np.concatenate([[True], np.diff(time_index) != 0])
    time_index, values = time_index[keep], values[keep].copy()
    if len(time_index) > 1 and not np.all(np.diff(time_index) == SAMPLE_PERIOD_MS):
        raise ValueError(f"time_index must increase by {SAMPLE_PERIOD_MS} ms between points")
    positions = np.arange(len(time_index))
    for channel in range(values.shape[1]):
        column = values[:, channel]
        missing = np.isnan(column)
        if missing.all():
            raise ValueError(f"channel {CHANNELS[channel]} has no valid value")
        if missing.any():
            column[missing] = np.interp(positions[missing], positions[~missing], column[~missing])
    return time_index, values


def window_starts(n_points: int, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Start indices of the sliding windows.

    Windows start every `stride` points; one last window aligned on the end of
    the ride is added if needed so that every point is covered.
    """
    if n_points < window:
        raise ValueError(f"a ride needs at least {window} points, got {n_points}")
    starts = list(range(0, n_points - window + 1, stride))
    if starts[-1] != n_points - window:
        starts.append(n_points - window)
    return np.asarray(starts, dtype=np.int64)


def make_windows(values: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Cut a clean `(n, 6)` ride into `(n_windows, 6, window)` float32 windows."""
    starts = window_starts(len(values), window, stride)
    stacked = np.stack([values[s : s + window] for s in starts])  # (k, window, 6)
    return np.ascontiguousarray(stacked.transpose(0, 2, 1), dtype=np.float32)


def window_labels(topple: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Label each window 1 if it contains at least one topple point, else 0."""
    topple = np.nan_to_num(np.asarray(topple, dtype=np.float64)) > 0
    starts = window_starts(len(topple), window, stride)
    return np.array([int(topple[s : s + window].any()) for s in starts], dtype=np.int64)


@dataclass(frozen=True)
class Normalization:
    """Per-channel mean and standard deviation, fitted on the training windows only."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    channels: tuple[str, ...] = CHANNELS
    window: int = WINDOW
    stride: int = STRIDE
    gyro_units: str = "rad_s"

    @classmethod
    def fit(cls, windows: np.ndarray) -> Normalization:
        """Compute the statistics over all points of `(n, 6, window)` windows."""
        mean = windows.mean(axis=(0, 2), dtype=np.float64)
        std = windows.std(axis=(0, 2), dtype=np.float64)
        std = np.where(std > 1e-8, std, 1.0)  # constant channel: leave it unscaled
        return cls(mean=tuple(map(float, mean)), std=tuple(map(float, std)))

    @classmethod
    def identity(cls) -> Normalization:
        """Neutral normalisation (mean 0, std 1), used by the dummy model."""
        return cls(mean=(0.0,) * len(CHANNELS), std=(1.0,) * len(CHANNELS))

    def apply(self, windows: np.ndarray) -> np.ndarray:
        """Normalise `(n, 6, window)` windows channel by channel."""
        mean = np.asarray(self.mean, dtype=np.float32)[None, :, None]
        std = np.asarray(self.std, dtype=np.float32)[None, :, None]
        return ((windows - mean) / std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "channels": list(self.channels),
            "mean": list(self.mean),
            "std": list(self.std),
            "window": self.window,
            "stride": self.stride,
            "gyro_units": self.gyro_units,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Normalization:
        """Build from the `normalization.json` content, checking it against the contract."""
        norm = cls(
            mean=tuple(float(v) for v in data["mean"]),
            std=tuple(float(v) for v in data["std"]),
            channels=tuple(data["channels"]),
            window=int(data["window"]),
            stride=int(data["stride"]),
            gyro_units=str(data["gyro_units"]),
        )
        if norm.channels != CHANNELS or norm.gyro_units != "rad_s":
            raise ValueError("normalization.json does not match the channel/unit contract")
        if len(norm.mean) != len(CHANNELS) or len(norm.std) != len(CHANNELS):
            raise ValueError("normalization.json must hold one mean and one std per channel")
        if any(s <= 0 for s in norm.std):
            raise ValueError("normalization std must be strictly positive")
        return norm

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Normalization:
        return cls.from_dict(json.loads(Path(path).read_text()))


def prepare_ride(
    time_index: np.ndarray,
    values: np.ndarray,
    units: Units,
    normalization: Normalization,
) -> np.ndarray:
    """Raw ride → normalised model input `(n_windows, 6, window)` (API and evaluation)."""
    _, clean = clean_ride(time_index, to_rad_s(values, units))
    windows = make_windows(clean, normalization.window, normalization.stride)
    return normalization.apply(windows)


def aggregate_ride_score(window_scores: np.ndarray) -> float:
    """Ride risk score = maximum of its window scores."""
    scores = np.asarray(window_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("window scores must be a non-empty array of finite values")
    return float(scores.max())
