"""Promotion and rollback against a throwaway local MLflow registry (SQLite)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from gbg_mlops.model_loader import EVALUATION_METRIC
from scripts.promote_model import promote, rollback
from scripts.register_dummy_model import register_dummy_model

MODEL, ALIAS = "gbg-risk", "champion"


@pytest.fixture
def registry(tmp_path):
    """Return (client, make_version, uri); no champion at start."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifacts = (tmp_path / "artifacts").as_uri()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    def make_version(metric: float | None) -> str:
        summary = register_dummy_model(uri, MODEL, ALIAS, artifact_location=artifacts)
        client.delete_registered_model_alias(MODEL, ALIAS)  # tests control the alias
        if metric is not None:
            client.log_metric(summary["run_id"], EVALUATION_METRIC, metric)
        return summary["version"]

    yield client, make_version, uri
    mlflow.set_tracking_uri(None)


def champion(client: MlflowClient) -> str:
    return str(client.get_model_version_by_alias(MODEL, ALIAS).version)


def test_first_promotion_without_champion(registry):
    client, make_version, _ = registry
    v1 = make_version(0.60)
    decision = promote(client, MODEL, ALIAS, v1)
    assert decision.status == "promoted" and decision.champion_before is None
    assert champion(client) == v1
    tags = client.get_model_version(MODEL, v1).tags
    assert tags["previous_champion"] == "none" and tags["promoted_at"]


def test_better_candidate_is_promoted(registry):
    client, make_version, _ = registry
    v1, v2 = make_version(0.60), make_version(0.75)
    promote(client, MODEL, ALIAS, v1)
    decision = promote(client, MODEL, ALIAS, v2)
    assert decision.status == "promoted"
    assert (decision.candidate_metric, decision.champion_metric) == (0.75, 0.60)
    assert champion(client) == v2
    assert client.get_model_version(MODEL, v2).tags["previous_champion"] == v1


@pytest.mark.parametrize("metric", [0.50, 0.60, None])
def test_worse_equal_or_unmeasured_candidate_is_refused(registry, metric):
    client, make_version, _ = registry
    v1, v2 = make_version(0.60), make_version(metric)
    promote(client, MODEL, ALIAS, v1)
    decision = promote(client, MODEL, ALIAS, v2)
    assert decision.status == "refused" and champion(client) == v1
    assert "promoted_at" not in client.get_model_version(MODEL, v2).tags


def test_champion_without_metric_is_replaced(registry):
    client, make_version, _ = registry
    dummy, real = make_version(None), make_version(0.55)
    promote(client, MODEL, ALIAS, dummy, force=True)
    assert promote(client, MODEL, ALIAS, real).status == "promoted"


def test_force_skips_the_comparison(registry):
    client, make_version, _ = registry
    v1, v2 = make_version(0.90), make_version(0.10)
    promote(client, MODEL, ALIAS, v1)
    decision = promote(client, MODEL, ALIAS, v2, force=True)
    assert decision.status == "promoted" and decision.forced and champion(client) == v2


def test_rollback_restores_the_previous_champion(registry):
    client, make_version, _ = registry
    v1, v2 = make_version(0.60), make_version(0.70)
    promote(client, MODEL, ALIAS, v1)
    promote(client, MODEL, ALIAS, v2)
    decision = rollback(client, MODEL, ALIAS)
    assert (decision.champion_before, decision.champion_after) == (v2, v1)
    assert champion(client) == v1
    assert "rolled_back_at" in client.get_model_version(MODEL, v2).tags


def test_rollback_needs_a_previous_champion(registry):
    client, make_version, _ = registry
    with pytest.raises(ValueError, match="no champion"):
        rollback(client, MODEL, ALIAS)
    promote(client, MODEL, ALIAS, make_version(0.6))
    with pytest.raises(ValueError, match="no previous"):
        rollback(client, MODEL, ALIAS)


def test_same_version_is_unchanged(registry):
    client, make_version, _ = registry
    v1 = make_version(0.6)
    promote(client, MODEL, ALIAS, v1)
    assert promote(client, MODEL, ALIAS, v1).status == "unchanged"


def test_cli_prints_json_and_exit_codes(registry):
    client, make_version, uri = registry
    v1, v2 = make_version(0.7), make_version(0.5)
    env = {"MLFLOW_TRACKING_URI": uri, "MODEL_NAME": MODEL, "MODEL_ALIAS": ALIAS}

    def run(*args: str) -> tuple[int, dict]:
        result = subprocess.run(
            [sys.executable, "scripts/promote_model.py", *args],
            capture_output=True,
            text=True,
            env={**os.environ, **env},
        )
        return result.returncode, json.loads(result.stdout)

    assert run("--candidate-version", v1)[0] == 0
    code, out = run("--candidate-version", v2)
    assert code == 2 and out["status"] == "refused" and out["champion_after"] == v1
    code, out = run("--candidate-version", "99")
    assert code == 1 and out["status"] == "error"
