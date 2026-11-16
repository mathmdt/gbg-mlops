"""Placeholder risk model, logged with MLflow "models from code".

MLflow stores this file next to the model and runs it at load time (no pickle).
It respects the model contract (CONTRACTS.md, section 2): input
`(n_windows, 6, 300)` float32, output one probability per window.
"""

import mlflow
import numpy as np
from mlflow.pyfunc import PythonModel

CONSTANT_RISK = 0.05


class DummyRiskModel(PythonModel):
    """Returns the same low risk for every window."""

    def predict(self, context, model_input, params=None):
        windows = np.asarray(model_input)
        if windows.ndim != 3 or windows.shape[1:] != (6, 300):
            raise ValueError(f"expected (n, 6, 300) windows, got {windows.shape}")
        return np.full(windows.shape[0], CONSTANT_RISK, dtype=np.float32)


mlflow.models.set_model(DummyRiskModel())
