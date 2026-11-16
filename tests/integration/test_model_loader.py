"""load_from_registry against a throwaway local MLflow store with the dummy model."""

import mlflow
import numpy as np
import pytest

from gbg_mlops.config import Settings
from gbg_mlops.features import Normalization
from gbg_mlops.model_loader import load_from_registry
from scripts.register_dummy_model import register_dummy_model


@pytest.fixture
def settings(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    # Registered in monkeypatch so that export_mlflow_env() changes are undone.
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_S3_ENDPOINT_URL"):
        monkeypatch.setenv(var, uri if var == "MLFLOW_TRACKING_URI" else "http://unused")
    register_dummy_model(uri, "gbg-risk", "champion", artifact_location=(tmp_path / "a").as_uri())
    yield Settings(_env_file=None, mlflow_tracking_uri=uri)
    mlflow.set_tracking_uri(None)


def test_loads_champion_with_its_normalization(settings):
    loaded = load_from_registry(settings)
    assert (loaded.name, loaded.version, loaded.alias) == ("gbg-risk", "1", "champion")
    assert loaded.normalization == Normalization.identity()
    time_index = np.arange(600) * 100
    score, n_windows = loaded.score_ride(time_index, np.zeros((600, 6)), "deg_s")
    assert score == pytest.approx(0.05) and n_windows == 3


def test_missing_alias_is_explicit(settings):
    with pytest.raises(RuntimeError, match="gbg-risk@challenger"):
        load_from_registry(settings.model_copy(update={"model_alias": "challenger"}))
