"""Web app client tests, against the real API with a fake model (no server needed)."""

from __future__ import annotations

import pytest

from app.client import ApiError, get_health, predict_file
from tests.integration.test_api import RecordingModel, client_for, ride_frame


def csv_bytes(n: int = 450) -> bytes:
    return ride_frame(n).to_csv(index=False).encode()


def test_health_ready():
    with client_for() as client:
        health = get_health(client)
    assert health["status"] == "ready"
    assert health["model_version"] == "7"


def test_predict_file_returns_the_score():
    with client_for(RecordingModel(score=0.8)) as client:
        result = predict_file(client, "ride.csv", csv_bytes(), "deg_s")
    assert result["risk_score"] == pytest.approx(0.8)
    assert result["is_reckless"] is True
    assert result["model_version"] == "7"


def test_too_short_ride_gives_a_readable_message():
    with client_for() as client, pytest.raises(ApiError, match="Invalid ride file"):
        predict_file(client, "ride.csv", csv_bytes(n=100), "deg_s")


def test_no_model_loaded():
    with (
        client_for(error=RuntimeError("down")) as client,
        pytest.raises(ApiError, match="No model"),
    ):
        predict_file(client, "ride.csv", csv_bytes(), "deg_s")
