"""Settings are read from the environment with safe defaults."""

from gbg_mlops.config import Settings


def test_defaults(monkeypatch):
    for var in ("MODEL_NAME", "MODEL_ALIAS", "RISK_THRESHOLD"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.model_name == "gbg-risk"
    assert settings.model_alias == "champion"
    assert settings.risk_threshold == 0.5


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_ALIAS", "challenger")
    monkeypatch.setenv("RISK_THRESHOLD", "0.7")
    settings = Settings(_env_file=None)
    assert settings.model_alias == "challenger"
    assert settings.risk_threshold == 0.7
