"""FastAPI application serving `MODEL_NAME@MODEL_ALIAS` from the MLflow registry.

The model is loaded once at startup (lifespan). If loading fails, the API still
starts, `/health` answers 503 with the reason and predictions are refused.
Run locally with `make api`, or in the dev stack with `make up`.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import PurePath
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gbg_mlops import __version__
from gbg_mlops.config import Settings, get_settings
from gbg_mlops.features import CHANNELS, Units, clean_ride
from gbg_mlops.model_loader import LoadedModel, load_from_registry
from gbg_mlops.schemas import (
    MAX_SAMPLES,
    MIN_SAMPLES,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# Child of uvicorn's logger, so messages use the handlers uvicorn configures.
logger = logging.getLogger("uvicorn.error").getChild("gbg")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FILE_COLUMNS = ("time_index", *CHANNELS)

ModelLoader = Callable[[Settings], LoadedModel]

DESCRIPTION = """
Risk score of a bike/scooter ride recorded by an IMU (accelerometer + gyroscope, 10 Hz).

The ride is converted to rad/s, cleaned, cut into 30 s windows and normalised with the
statistics of the training run; the model scores each window and the ride score is the
maximum. **A score is an estimate, not a certainty.** Contract: `CONTRACTS.md`.
"""

TAGS = [
    {"name": "health", "description": "Service and model status."},
    {"name": "prediction", "description": "Risk score of one ride (JSON or file)."},
]


def _unprocessable(message: str, loc: tuple[str, ...] = ("body",)) -> HTTPException:
    """422 with the same body shape as FastAPI validation errors."""
    detail = [{"loc": list(loc), "msg": message, "type": "value_error"}]
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def create_app(loader: ModelLoader = load_from_registry, settings: Settings | None = None):
    """Build the application; `loader` is injectable so tests need no MLflow server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or get_settings()
        app.state.model = None
        app.state.load_error = None
        try:
            app.state.model = loader(app.state.settings)
            logger.info(
                "Loaded model %s version %s (@%s)",
                app.state.model.name,
                app.state.model.version,
                app.state.model.alias,
            )
        except Exception as error:  # the API must start even without a model
            app.state.load_error = f"{type(error).__name__}: {error}"
            logger.error("Model not loaded: %s", app.state.load_error)
        yield
        app.state.model = None

    app = FastAPI(
        title="GBG risk API",
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Never echo the input back (it can be large, or contain NaN/Infinity).
        detail = [{k: item[k] for k in ("loc", "msg", "type")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def require_model(request: Request) -> LoadedModel:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        return model

    def respond(
        request: Request,
        time_index: np.ndarray,
        values: np.ndarray,
        units: Units,
        ride_id: str | None,
    ) -> PredictResponse:
        model = require_model(request)
        try:
            score, n_windows = model.score_ride(time_index, values, units)
        except ValueError as error:  # preprocessing rejected the ride
            raise _unprocessable(str(error)) from error
        except Exception as error:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from error
        threshold = request.app.state.settings.risk_threshold
        response = PredictResponse(
            prediction_id=uuid4(),
            ride_id=ride_id,
            risk_score=score,
            is_reckless=score >= threshold,
            threshold=threshold,
            predicted_class=None,
            model_name=model.name,
            model_version=model.version,
            n_windows=n_windows,
        )
        logger.info(
            "prediction %s ride=%s score=%.4f windows=%d version=%s",
            response.prediction_id,
            ride_id,
            score,
            n_windows,
            model.version,
        )
        return response

    @app.get(
        "/health",
        tags=["health"],
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "Model not loaded"}},
        summary="Service and model status",
    )
    def health(request: Request):
        """200 with the served model identity when ready, 503 with the reason otherwise."""
        state, settings = request.app.state, request.app.state.settings
        if state.model is None:
            body = HealthResponse(
                status="not_ready",
                model_name=settings.model_name,
                model_alias=settings.model_alias,
                detail=state.load_error,
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        return HealthResponse(
            status="ready",
            model_name=state.model.name,
            model_version=state.model.version,
            model_alias=state.model.alias,
        )

    @app.post(
        "/predict",
        tags=["prediction"],
        response_model=PredictResponse,
        responses={
            422: {"description": "Invalid ride"},
            503: {"description": "Model not ready"},
        },
        summary="Score one ride sent as JSON",
    )
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        """Samples must be sorted, every 100 ms; `null` channel values are interpolated."""
        time_index = np.array([s.time_index for s in payload.samples], dtype=np.int64)
        values = np.array(
            [
                [np.nan if (v := getattr(s, c)) is None else v for c in CHANNELS]
                for s in payload.samples
            ],
            dtype=np.float64,
        )
        return respond(request, time_index, values, payload.units, payload.ride_id)

    @app.post(
        "/predict/file",
        tags=["prediction"],
        response_model=PredictResponse,
        responses={
            413: {"description": "File larger than 20 MB"},
            422: {"description": "Invalid file or ride"},
            503: {"description": "Model not ready"},
        },
        summary="Score one ride sent as a CSV or Parquet file",
    )
    async def predict_file(
        request: Request,
        file: Annotated[
            UploadFile,
            File(
                description="One ride: columns time_index, ax, ay, az, rx, ry, rz (.csv, .parquet)"
            ),
        ],
        units: Annotated[
            Literal["deg_s", "rad_s"], Form(description="Gyroscope units (GBG500: deg_s)")
        ],
        ride_id: Annotated[
            str | None, Form(max_length=128, description="Optional; default: ride_id column")
        ] = None,
    ) -> PredictResponse:
        """The file is cleaned like the raw data: sorted, duplicated timestamps dropped."""
        require_model(request)
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File larger than 20 MB")
        frame = _read_ride_file(file.filename or "", content)
        if ride_id is None and "ride_id" in frame.columns:
            ride_id = str(frame["ride_id"].iloc[0])[:128]
        try:
            numbers = frame[list(FILE_COLUMNS)].apply(pd.to_numeric, errors="raise")
            time_index, values = clean_ride(
                numbers["time_index"].to_numpy(dtype=np.int64),
                numbers[list(CHANNELS)].to_numpy(dtype=np.float64),
            )
        except (ValueError, TypeError) as error:
            raise _unprocessable(str(error), ("body", "file")) from error
        if not MIN_SAMPLES <= len(time_index) <= MAX_SAMPLES:
            raise _unprocessable(
                f"a ride needs {MIN_SAMPLES} to {MAX_SAMPLES} samples, got {len(time_index)}",
                ("body", "file"),
            )
        return respond(request, time_index, values, units, ride_id)

    return app


def _read_ride_file(filename: str, content: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV/Parquet file holding exactly one ride."""
    suffix = PurePath(filename).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(content))
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(io.BytesIO(content))
        else:
            raise _unprocessable("file must be .csv or .parquet", ("body", "file"))
    except HTTPException:
        raise
    except Exception as error:
        raise _unprocessable(f"cannot read the file: {error}", ("body", "file")) from error
    missing = [c for c in FILE_COLUMNS if c not in frame.columns]
    if missing:
        raise _unprocessable(f"missing columns: {missing}", ("body", "file"))
    if "ride_id" in frame.columns and frame["ride_id"].nunique(dropna=False) > 1:
        raise _unprocessable("the file must contain a single ride", ("body", "file"))
    return frame


app = create_app()
