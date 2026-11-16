"""API contract: request validation and response shapes."""

import json

import pytest
from pydantic import ValidationError

from gbg_mlops.schemas import HealthResponse, PredictRequest, PredictResponse


def samples(n: int = 300, step: int = 100) -> list[dict]:
    return [
        {"time_index": i * step, "ax": 0.1, "ay": -9.8, "az": 0.2, "rx": 1.0, "ry": 2.0, "rz": 3.0}
        for i in range(n)
    ]


def test_valid_request_with_missing_values():
    data = samples()
    data[10]["ax"] = None
    request = PredictRequest(ride_id="r1", units="deg_s", samples=data)
    assert len(request.samples) == 300 and request.samples[10].ax is None


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(units="rpm"),
        lambda d: d.pop("units"),
        lambda d: d.update(samples=samples(299)),
        lambda d: d.update(samples=samples(20_001)),
        lambda d: d.update(samples=samples(300, step=200)),
        lambda d: d.update(extra_field=1),
        lambda d: d["samples"][0].update(temperature=20),
        lambda d: d["samples"][0].update(ax=True),
        lambda d: d["samples"][5].update(time_index=-1),
        lambda d: d.update(samples=[{**s, "rz": None} for s in d["samples"]]),
    ],
)
def test_invalid_requests_are_rejected(change):
    data = {"ride_id": "r1", "units": "rad_s", "samples": samples()}
    change(data)
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(data)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_are_rejected_from_json(value):
    body = json.dumps({"units": "rad_s", "samples": samples()}).replace("0.1", value, 1)
    with pytest.raises(ValidationError):
        PredictRequest.model_validate_json(body)


def test_examples_in_schema_are_valid():
    schema = PredictRequest.model_json_schema()
    example = schema["examples"][0]
    assert example["units"] == "deg_s"
    response = PredictResponse.model_json_schema()["examples"][0]
    assert PredictResponse.model_validate(response).predicted_class is None


def test_response_bounds_and_health():
    base = PredictResponse.model_json_schema()["examples"][0]
    with pytest.raises(ValidationError):
        PredictResponse.model_validate({**base, "risk_score": 1.2})
    assert HealthResponse(status="not_ready", detail="loading").model_name is None
