"""End-to-end tests: real HTTP calls to a running API (dev stack or Kubernetes).

    make up && make dummy-model      # or a deployed API
    make e2e                         # API_URL=http://localhost:8000 by default

Excluded from `make test` (marker e2e). Set E2E_RIDE_FILE to a real ride (.csv or
.parquet, e.g. exported with scripts/export_ride.py) to also score it.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

API_URL = os.environ.get("API_URL", "http://localhost:8000")
READY_TIMEOUT_S = float(os.environ.get("E2E_READY_TIMEOUT", "120"))


def synthetic_ride(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """One minute of plausible IMU data at 10 Hz (gyroscope in °/s, like GBG500)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time_index": np.arange(n) * 100,
            "ax": rng.normal(0, 1, n),
            "ay": rng.normal(-9.8, 1, n),
            "az": rng.normal(0, 1, n),
            "rx": rng.normal(0, 5, n),
            "ry": rng.normal(0, 5, n),
            "rz": rng.normal(0, 5, n),
        }
    )


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=API_URL, timeout=60) as client:
        yield client


@pytest.fixture(scope="module")
def health(client: httpx.Client) -> dict:
    """Wait until the API has a model loaded (a fresh deployment needs some time)."""
    deadline, last = time.monotonic() + READY_TIMEOUT_S, "no answer"
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                return response.json()
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as error:
            last = f"{type(error).__name__}: {error}"
        time.sleep(3)
    pytest.fail(f"API at {API_URL} not ready after {READY_TIMEOUT_S:.0f} s ({last})")


def check_prediction(body: dict, health: dict) -> None:
    """Fields and consistency rules of PredictResponse (CONTRACTS.md section 3)."""
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["is_reckless"] == (body["risk_score"] >= body["threshold"])
    assert body["n_windows"] >= 1
    assert body["model_name"] == health["model_name"]
    assert body["model_version"] == health["model_version"]


def test_health_reports_the_served_model(health: dict):
    assert health["status"] == "ready"
    assert health["model_alias"] == "champion"
    assert health["model_version"]


def test_predict_file_with_a_ride(client: httpx.Client, health: dict):
    content = synthetic_ride().to_csv(index=False).encode()
    response = client.post(
        "/predict/file",
        files={"file": ("e2e_ride.csv", content)},
        data={"units": "deg_s", "ride_id": "e2e-ride"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    check_prediction(body, health)
    assert body["ride_id"] == "e2e-ride"


def test_predict_json_with_a_ride(client: httpx.Client, health: dict):
    ride = synthetic_ride(n=400, seed=1)
    payload = {"ride_id": "e2e-json", "units": "deg_s", "samples": ride.to_dict("records")}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    check_prediction(response.json(), health)


def test_too_short_ride_is_rejected(client: httpx.Client, health: dict):
    content = synthetic_ride(n=100).to_csv(index=False).encode()
    response = client.post(
        "/predict/file", files={"file": ("short.csv", content)}, data={"units": "deg_s"}
    )
    assert response.status_code == 422


@pytest.mark.skipif(not os.environ.get("E2E_RIDE_FILE"), reason="set E2E_RIDE_FILE")
def test_predict_a_real_ride(client: httpx.Client, health: dict):
    path = Path(os.environ["E2E_RIDE_FILE"])
    response = client.post(
        "/predict/file",
        files={"file": (path.name, io.BytesIO(path.read_bytes()))},
        data={"units": os.environ.get("E2E_RIDE_UNITS", "deg_s")},
    )
    assert response.status_code == 200, response.text
    check_prediction(response.json(), health)
