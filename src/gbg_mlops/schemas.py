"""Pydantic schemas of the API: the public contract (CONTRACTS.md, section 3)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gbg_mlops.features import CHANNELS, SAMPLE_PERIOD_MS, WINDOW

MIN_SAMPLES = WINDOW  # one full window
MAX_SAMPLES = 20_000  # ≈ 33 minutes at 10 Hz

_EXAMPLE_SAMPLES = [
    {"time_index": 0, "ax": -2.513, "ay": -9.385, "az": -1.053, "rx": -0.009, "ry": 0.009,
     "rz": -0.201},
    {"time_index": 100, "ax": -2.491, "ay": -9.385, "az": -1.079, "rx": -0.037, "ry": 0.027,
     "rz": -0.156},
]  # fmt: skip


class Sample(BaseModel):
    """One IMU measurement. A channel may be `null` (missing): it is interpolated."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    time_index: int = Field(ge=0, description="Milliseconds since the start of the ride")
    ax: float | None = Field(description="Acceleration x (m/s²)")
    ay: float | None = Field(description="Acceleration y (m/s²)")
    az: float | None = Field(description="Acceleration z (m/s²)")
    rx: float | None = Field(description="Angular velocity x, in `units`")
    ry: float | None = Field(description="Angular velocity y, in `units`")
    rz: float | None = Field(description="Angular velocity z, in `units`")

    @field_validator("*", mode="before")
    @classmethod
    def reject_booleans(cls, value):
        if isinstance(value, bool):
            raise ValueError("booleans are not valid measurements")
        return value


class PredictRequest(BaseModel):
    """One ride: at least one window of samples, regularly spaced every 100 ms."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"ride_id": "demo-ride-001", "units": "deg_s", "samples": _EXAMPLE_SAMPLES}
            ]
        },
    )

    ride_id: str | None = Field(default=None, max_length=128, description="Optional client id")
    units: Literal["deg_s", "rad_s"] = Field(
        description="Gyroscope units of rx/ry/rz (GBG500: deg_s, SRV3490: rad_s)"
    )
    samples: list[Sample] = Field(
        min_length=MIN_SAMPLES,
        max_length=MAX_SAMPLES,
        description=f"Between {MIN_SAMPLES} and {MAX_SAMPLES} samples",
    )

    @model_validator(mode="after")
    def regular_time_index(self) -> PredictRequest:
        times = [s.time_index for s in self.samples]
        if any(b - a != SAMPLE_PERIOD_MS for a, b in zip(times, times[1:], strict=False)):
            raise ValueError(
                f"time_index must be sorted and increase by exactly {SAMPLE_PERIOD_MS} ms"
            )
        for channel in CHANNELS:
            if all(getattr(s, channel) is None for s in self.samples):
                raise ValueError(f"channel {channel} has no valid value")
        return self


class PredictResponse(BaseModel):
    """Ride-level risk score, with the model version that produced it."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "prediction_id": "3f6c2a8e-1b7d-4c55-9d0e-2a4b8f9c1e77",
                    "ride_id": "demo-ride-001",
                    "risk_score": 0.05,
                    "is_reckless": False,
                    "threshold": 0.5,
                    "predicted_class": None,
                    "model_name": "gbg-risk",
                    "model_version": "1",
                    "n_windows": 19,
                }
            ]
        },
    )

    prediction_id: UUID
    ride_id: str | None
    risk_score: float = Field(ge=0, le=1, description="Max of the window probabilities")
    is_reckless: bool = Field(description="risk_score >= threshold")
    threshold: float = Field(gt=0, lt=1)
    predicted_class: str | None = Field(
        default=None, description="Reserved for a 5-class model; null for now"
    )
    model_name: str
    model_version: str
    n_windows: int = Field(ge=1)


class HealthResponse(BaseModel):
    """Service state; HTTP 200 when ready, 503 otherwise."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ready", "not_ready"]
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    detail: str | None = None
