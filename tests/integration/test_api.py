"""API contract tests with an injected fake model (no MLflow server)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from gbg_mlops.config import Settings
from gbg_mlops.features import DEG_PER_RAD, Normalization
from gbg_mlops.model_loader import LoadedModel

SETTINGS = Settings(_env_file=None, risk_threshold=0.5)


class RecordingModel:
    """Returns a fixed score per window and keeps the last input."""

    def __init__(self, score: float = 0.05):
        self.score, self.last_input = score, None

    def predict(self, model_input):
        self.last_input = np.asarray(model_input)
        return np.full(len(model_input), self.score, dtype=np.float32)


def loaded(model, normalization=None) -> LoadedModel:
    return LoadedModel(
        model, normalization or Normalization.identity(), "gbg-risk", "7", "champion"
    )


def client_for(model=None, normalization=None, error: Exception | None = None) -> TestClient:
    def loader(settings):
        if error:
            raise error
        return loaded(model or RecordingModel(), normalization)

    return TestClient(create_app(loader, SETTINGS))


def ride_samples(n: int = 450, gyro: float = 0.5) -> list[dict]:
    return [
        {
            "time_index": i * 100,
            "ax": 0.1,
            "ay": -9.8,
            "az": 0.3,
            "rx": gyro,
            "ry": 0.0,
            "rz": -gyro,
        }
        for i in range(n)
    ]


def ride_frame(n: int = 450) -> pd.DataFrame:
    return pd.DataFrame(ride_samples(n)).assign(ride_id="ride-42")


def test_health_ready():
    with client_for() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_name": "gbg-risk",
        "model_version": "7",
        "model_alias": "champion",
        "detail": None,
    }


def test_not_ready_when_loading_fails():
    with client_for(error=RuntimeError("no version 'gbg-risk@champion'")) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "not_ready"
        assert "gbg-risk@champion" in health.json()["detail"]
        body = {"units": "rad_s", "samples": ride_samples()}
        assert client.post("/predict", json=body).status_code == 503


def test_predict_returns_the_contract_response():
    model = RecordingModel(score=0.8)
    with client_for(model) as client:
        response = client.post(
            "/predict", json={"ride_id": "r1", "units": "rad_s", "samples": ride_samples(450)}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["ride_id"] == "r1" and body["n_windows"] == 2
    assert body["risk_score"] == pytest.approx(0.8) and body["is_reckless"] is True
    assert body["threshold"] == 0.5 and body["predicted_class"] is None
    assert (body["model_name"], body["model_version"]) == ("gbg-risk", "7")
    assert model.last_input.shape == (2, 6, 300) and model.last_input.dtype == np.float32


def test_ride_score_is_the_max_of_windows():
    class Rising(RecordingModel):
        def predict(self, model_input):
            return np.linspace(0.1, 0.3, len(model_input)).astype(np.float32)

    with client_for(Rising()) as client:
        body = client.post("/predict", json={"units": "rad_s", "samples": ride_samples(600)}).json()
    assert body["risk_score"] == pytest.approx(0.3) and body["is_reckless"] is False


def test_degrees_are_converted_to_radians():
    model = RecordingModel()
    with client_for(model) as client:
        client.post("/predict", json={"units": "rad_s", "samples": ride_samples(gyro=0.5)})
        in_rad = model.last_input.copy()
        client.post(
            "/predict", json={"units": "deg_s", "samples": ride_samples(gyro=0.5 * DEG_PER_RAD)}
        )
    np.testing.assert_allclose(model.last_input, in_rad, rtol=1e-5)
    np.testing.assert_allclose(model.last_input[:, 3, :], 0.5, rtol=1e-5)  # rx in rad/s


def test_normalization_from_the_run_is_applied():
    model = RecordingModel()
    norm = Normalization(mean=(0.1, -9.8, 0.3, 0.0, 0.0, 0.0), std=(1, 2, 1, 0.25, 1, 1))
    with client_for(model, norm) as client:
        client.post("/predict", json={"units": "rad_s", "samples": ride_samples(gyro=0.5)})
    np.testing.assert_allclose(model.last_input[:, :3, :], 0, atol=1e-6)
    np.testing.assert_allclose(model.last_input[:, 3, :], 2.0, rtol=1e-5)


def test_missing_values_are_interpolated():
    model = RecordingModel()
    samples = ride_samples()
    samples[10] = {**samples[10], "ax": None, "rx": None}
    with client_for(model) as client:
        assert (
            client.post("/predict", json={"units": "rad_s", "samples": samples}).status_code == 200
        )
    assert not np.isnan(model.last_input).any()


@pytest.mark.parametrize(
    "body",
    [
        {"units": "rad_s", "samples": ride_samples(299)},
        {"units": "rad_s", "samples": ride_samples()[:150] + ride_samples()[151:]},
        {"samples": ride_samples()},
        {"units": "rad_s", "samples": ride_samples(), "unexpected": True},
    ],
)
def test_invalid_rides_are_rejected(body):
    with client_for() as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert all("input" not in item for item in response.json()["detail"])


def test_model_breaking_the_contract_gives_500():
    class Broken(RecordingModel):
        def predict(self, model_input):
            return np.array([2.0], dtype=np.float32)

    with client_for(Broken()) as client:
        response = client.post("/predict", json={"units": "rad_s", "samples": ride_samples()})
    assert response.status_code == 500


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_predict_file(fmt):
    frame = ride_frame()
    frame = pd.concat([frame, frame.iloc[[5]]])  # duplicated timestamp, as in raw data
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False) if fmt == "csv" else frame.to_parquet(buffer, index=False)
    with client_for() as client:
        response = client.post(
            "/predict/file",
            files={"file": (f"ride.{fmt}", buffer.getvalue())},
            data={"units": "deg_s"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["ride_id"] == "ride-42" and response.json()["n_windows"] == 2


@pytest.mark.parametrize(
    "filename, content",
    [
        ("ride.txt", b"time_index,ax\n0,1\n"),
        ("ride.csv", ride_frame().drop(columns=["rz"]).to_csv(index=False).encode()),
        ("ride.csv", ride_frame(200).to_csv(index=False).encode()),
        ("ride.csv", ride_frame().assign(ride_id=["a", "b"] * 225).to_csv(index=False).encode()),
        ("ride.parquet", b"not a parquet file"),
    ],
)
def test_invalid_files_are_rejected(filename, content):
    with client_for() as client:
        response = client.post(
            "/predict/file", files={"file": (filename, content)}, data={"units": "rad_s"}
        )
    assert response.status_code == 422


def test_openapi_documents_the_routes():
    with client_for() as client:
        schema = client.get("/openapi.json").json()
    assert {"/health", "/predict", "/predict/file"} <= set(schema["paths"])
    assert schema["components"]["schemas"]["PredictRequest"]["examples"]
