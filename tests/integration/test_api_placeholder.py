"""Until a model is served, /health must answer 503 with the contract body."""

from fastapi.testclient import TestClient

from api.main import app


def test_health_reports_not_ready():
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["model_name"] == "gbg-risk" and body["model_version"] is None
