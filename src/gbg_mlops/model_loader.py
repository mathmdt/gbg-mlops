"""MLflow side of the model contract (CONTRACTS.md, section 2) and model loading.

Constants are shared by training, the dummy model and the API. `load_from_registry`
resolves `models:/<name>@<alias>` once, loads that exact version and its
`normalization.json`, and checks them against the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlflow
import numpy as np
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import Schema, TensorSpec

from gbg_mlops.config import Settings
from gbg_mlops.features import (
    CHANNELS,
    WINDOW,
    Normalization,
    Units,
    aggregate_ride_score,
    prepare_ride,
)

EXPERIMENT_NAME = "gbg-risk"
MODEL_ARTIFACT_PATH = "model"
NORMALIZATION_ARTIFACT = "normalization.json"

MODEL_SIGNATURE = ModelSignature(
    # Unnamed tensors: pyfunc then passes a plain ndarray (named ones become a dict).
    inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, len(CHANNELS), WINDOW))]),
    outputs=Schema([TensorSpec(np.dtype(np.float32), (-1,))]),
)


class WindowModel(Protocol):
    """Anything with a pyfunc-like `predict(windows) -> probabilities`."""

    def predict(self, model_input: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class LoadedModel:
    """A served model, its preprocessing parameters and its registry identity."""

    model: WindowModel
    normalization: Normalization
    name: str
    version: str
    alias: str
    run_id: str | None = None

    def score_ride(
        self, time_index: np.ndarray, values: np.ndarray, units: Units
    ) -> tuple[float, int]:
        """Return `(ride risk score, number of windows)` for one raw ride."""
        windows = prepare_ride(time_index, values, units, self.normalization)
        scores = np.asarray(self.model.predict(windows), dtype=np.float64).reshape(-1)
        if scores.shape != (len(windows),):
            raise RuntimeError(
                f"model returned {scores.shape[0]} scores for {len(windows)} windows"
            )
        if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
            raise RuntimeError("model returned scores outside [0, 1]")
        return aggregate_ride_score(scores), len(windows)


def load_from_registry(settings: Settings) -> LoadedModel:
    """Load `MODEL_NAME@MODEL_ALIAS` and its normalisation from the MLflow registry.

    Raises an exception with an explicit message if anything is missing or does
    not match the contract.
    """
    settings.export_mlflow_env()
    uri = settings.mlflow_tracking_uri
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    try:
        version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    except mlflow.exceptions.MlflowException as error:
        raise RuntimeError(
            f"no version '{settings.model_name}@{settings.model_alias}' in the registry at {uri}"
        ) from error
    number = str(version.version)

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    # Load the resolved version, so that the alias cannot move between the two calls.
    model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{number}")
    inputs = model.metadata.get_input_schema()
    if inputs is None or not inputs.is_tensor_spec():
        raise RuntimeError("model has no tensor input signature")
    shape = tuple(inputs.inputs[0].shape)
    if shape != (-1, len(CHANNELS), WINDOW):
        raise RuntimeError(f"model input shape {shape} != (-1, {len(CHANNELS)}, {WINDOW})")

    path = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path=NORMALIZATION_ARTIFACT, tracking_uri=uri
    )
    normalization = Normalization.load(path)
    if normalization.window != WINDOW:
        raise RuntimeError("normalization.json window does not match the model input")

    return LoadedModel(
        model=model,
        normalization=normalization,
        name=settings.model_name,
        version=number,
        alias=settings.model_alias,
        run_id=version.run_id,
    )
