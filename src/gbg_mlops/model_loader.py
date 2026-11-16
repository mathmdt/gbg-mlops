"""MLflow side of the model contract (CONTRACTS.md, section 2).

Constants shared by training, the dummy model and the API. Loading from the
registry is added in step 5.
"""

import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

from gbg_mlops.features import CHANNELS, WINDOW

EXPERIMENT_NAME = "gbg-risk"
MODEL_ARTIFACT_PATH = "model"
NORMALIZATION_ARTIFACT = "normalization.json"

MODEL_SIGNATURE = ModelSignature(
    # Unnamed tensors: pyfunc then passes a plain ndarray (named ones become a dict).
    inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, len(CHANNELS), WINDOW))]),
    outputs=Schema([TensorSpec(np.dtype(np.float32), (-1,))]),
)
