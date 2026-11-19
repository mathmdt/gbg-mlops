"""HTTP client of the GBG risk API, used by the web app.

Calls only GET /health and POST /predict/file (contract: CONTRACTS.md, section 3).
"""

from __future__ import annotations

from typing import Any

import httpx

UNITS = {"deg_s": "°/s (GBG500)", "rad_s": "rad/s (SRV3490)"}


class ApiError(Exception):
    """The API could not be reached or refused the request."""


def get_health(client: httpx.Client) -> dict[str, Any]:
    """Body of /health: status "ready" with the model version, or "not_ready" with the reason."""
    try:
        response = client.get("/health")
    except httpx.HTTPError as error:
        raise ApiError(f"API unreachable at {client.base_url}") from error
    return response.json()


def predict_file(
    client: httpx.Client,
    filename: str,
    content: bytes,
    units: str,
    ride_id: str | None = None,
) -> dict[str, Any]:
    """Send one ride file to POST /predict/file and return the prediction."""
    data = {"units": units}
    if ride_id:
        data["ride_id"] = ride_id
    files = {"file": (filename, content)}
    try:
        response = client.post("/predict/file", files=files, data=data)
    except httpx.HTTPError as error:
        raise ApiError(f"API unreachable at {client.base_url}") from error
    if response.status_code != 200:
        raise ApiError(error_message(response))
    return response.json()


def error_message(response: httpx.Response) -> str:
    """Readable message for an error answer of the API."""
    if response.status_code == 422:
        detail = response.json().get("detail", [])
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", "") for item in detail)
        return f"Invalid ride file: {detail}"
    messages = {
        413: "File larger than 20 MB",
        500: "The model failed on this ride",
        503: "No model loaded in the API yet",
    }
    return messages.get(response.status_code, f"HTTP error {response.status_code}")
