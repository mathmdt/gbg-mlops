"""FastAPI application serving the registered model.

Placeholder until step 5: the service starts and answers `/health` with 503
(`not_ready`), as the contract requires when no model is loaded.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from gbg_mlops.config import get_settings
from gbg_mlops.schemas import HealthResponse

app = FastAPI(title="GBG risk API", version="0.1.0")


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health() -> JSONResponse:
    """Report whether a model is loaded (never, in this placeholder)."""
    settings = get_settings()
    body = HealthResponse(
        status="not_ready",
        model_name=settings.model_name,
        model_alias=settings.model_alias,
        detail="Model loading is not implemented yet",
    )
    return JSONResponse(status_code=503, content=body.model_dump())
