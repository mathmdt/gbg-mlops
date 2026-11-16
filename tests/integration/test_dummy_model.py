"""Dummy model registration against a throwaway local MLflow store (no server)."""

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from gbg_mlops.features import Normalization
from gbg_mlops.model_loader import NORMALIZATION_ARTIFACT
from scripts.register_dummy_model import register_dummy_model

MODEL, ALIAS = "gbg-risk", "champion"


@pytest.fixture
def store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    yield uri, (tmp_path / "artifacts").as_uri()
    mlflow.set_tracking_uri(None)


def test_registers_versions_and_moves_alias(store):
    uri, artifacts = store
    first = register_dummy_model(uri, MODEL, ALIAS, artifact_location=artifacts)
    second = register_dummy_model(uri, MODEL, ALIAS, artifact_location=artifacts)
    assert (first["version"], second["version"]) == ("1", "2")
    assert first["promoted"] and second["promoted"]
    client = MlflowClient(uri, uri)
    assert str(client.get_model_version_by_alias(MODEL, ALIAS).version) == "2"
    assert client.get_model_version(MODEL, "1").tags["model_type"] == "dummy"


def test_champion_respects_the_model_contract(store):
    uri, artifacts = store
    summary = register_dummy_model(uri, MODEL, ALIAS, artifact_location=artifacts)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL}@{ALIAS}")
    out = model.predict(np.zeros((4, 6, 300), dtype=np.float32))
    assert out.shape == (4,) and out.dtype == np.float32
    np.testing.assert_allclose(out, 0.05)
    inputs = model.metadata.get_input_schema().inputs[0]
    assert inputs.shape == (-1, 6, 300)
    path = mlflow.artifacts.download_artifacts(
        run_id=summary["run_id"], artifact_path=NORMALIZATION_ARTIFACT
    )
    assert Normalization.load(path) == Normalization.identity()
    with pytest.raises(Exception):  # noqa: B017 - MLflow wraps the ValueError
        model.predict(np.zeros((2, 6, 100), dtype=np.float32))


def test_real_champion_is_not_replaced(store):
    uri, artifacts = store
    register_dummy_model(uri, MODEL, ALIAS, artifact_location=artifacts)
    client = MlflowClient(uri, uri)
    client.set_model_version_tag(MODEL, "1", "model_type", "cnn")  # pretend it is real
    summary = register_dummy_model(uri, MODEL, ALIAS, artifact_location=artifacts)
    assert not summary["promoted"]
    assert str(client.get_model_version_by_alias(MODEL, ALIAS).version) == "1"
    forced = register_dummy_model(
        uri, MODEL, ALIAS, force_champion=True, artifact_location=artifacts
    )
    assert forced["promoted"]
    assert str(client.get_model_version_by_alias(MODEL, ALIAS).version) == forced["version"]
