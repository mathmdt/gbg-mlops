"""Tests for the 1D CNN risk model and its pyfunc wrapper (Fiche Ariane step 9).

Validates the mandatory contract:
- The model accepts `(n, 6, 300)` float32 arrays for any batch size n.
- The model returns an 1D array of `n` probabilities strictly bounded in `[0, 1]`.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gbg_mlops.features import CHANNELS, WINDOW, Normalization
from gbg_mlops.model_loader import LoadedModel
from training.model import RiskCNN1D, RiskPyfuncWrapper

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


@pytest.mark.parametrize("n", [1, 2, 7, 19, 64, 100])
def test_model_accepts_various_batch_sizes(n: int) -> None:
    """The model must accept (n, 6, 300) float32 for any batch size n."""
    model = RiskCNN1D()
    windows = np.random.randn(n, len(CHANNELS), WINDOW).astype(np.float32)

    probs = model.predict_proba(windows)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (n,), f"Expected shape ({n},), got {probs.shape}"
    assert probs.dtype == np.float32, f"Expected float32, got {probs.dtype}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must be in [0, 1]"


def test_model_handles_zeros_and_extremes() -> None:
    """Probabilities must remain bounded in [0, 1] even with extreme values."""
    model = RiskCNN1D()

    # All zeros
    zeros = np.zeros((4, len(CHANNELS), WINDOW), dtype=np.float32)
    probs_zeros = model.predict_proba(zeros)
    assert probs_zeros.shape == (4,)
    assert np.isfinite(probs_zeros).all()
    assert (probs_zeros >= 0.0).all() and (probs_zeros <= 1.0).all()

    # Extreme sensor readings (e.g. violent shocks or rotations)
    extremes = np.full((3, len(CHANNELS), WINDOW), 100.0, dtype=np.float32)
    extremes[1] = -100.0
    probs_extremes = model.predict_proba(extremes)
    assert probs_extremes.shape == (3,)
    assert np.isfinite(probs_extremes).all()
    assert (probs_extremes >= 0.0).all() and (probs_extremes <= 1.0).all()


def test_model_eval_determinism() -> None:
    """In eval mode, the model must produce deterministic predictions for identical inputs."""
    model = RiskCNN1D()
    windows = np.random.randn(5, len(CHANNELS), WINDOW).astype(np.float32)

    probs1 = model.predict_proba(windows)
    probs2 = model.predict_proba(windows)

    np.testing.assert_array_equal(probs1, probs2)


def test_pyfunc_wrapper_contract() -> None:
    """The PyFunc wrapper must accept (n, 6, 300) and return (n,) float32 probabilities."""
    model = RiskCNN1D()
    wrapper = RiskPyfuncWrapper(model=model)

    windows = np.random.randn(5, len(CHANNELS), WINDOW).astype(np.float32)
    probs = wrapper.predict(context=None, model_input=windows)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (5,)
    assert probs.dtype == np.float32
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_pyfunc_wrapper_rejects_invalid_inputs() -> None:
    """The PyFunc wrapper must raise ValueError for inputs violating the contract."""
    model = RiskCNN1D()
    wrapper = RiskPyfuncWrapper(model=model)

    # 2D instead of 3D
    with pytest.raises(ValueError, match="expected"):
        wrapper.predict(context=None, model_input=np.ones((6, WINDOW), dtype=np.float32))

    # Wrong number of channels (5 channels instead of 6)
    with pytest.raises(ValueError, match="expected"):
        wrapper.predict(context=None, model_input=np.ones((4, 5, WINDOW), dtype=np.float32))

    # Wrong window length (150 instead of 300)
    with pytest.raises(ValueError, match="expected"):
        wrapper.predict(context=None, model_input=np.ones((4, 6, 150), dtype=np.float32))

    # Non-finite values (NaN / Inf)
    bad_nan = np.ones((2, 6, WINDOW), dtype=np.float32)
    bad_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        wrapper.predict(context=None, model_input=bad_nan)

    bad_inf = np.ones((2, 6, WINDOW), dtype=np.float32)
    bad_inf[1, 2, 50] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        wrapper.predict(context=None, model_input=bad_inf)


def test_integration_with_loaded_model_pipeline() -> None:
    """The CNN model integrates seamlessly with LoadedModel.score_ride."""
    model = RiskCNN1D()
    wrapper = RiskPyfuncWrapper(model=model)
    loaded = LoadedModel(
        model=wrapper,
        normalization=Normalization.identity(),
        name="gbg-risk",
        version="test",
        alias="champion",
    )

    # 1 ride with 450 points (produces 2 windows of 300 points with stride 150)
    n_points = 450
    time_index = np.arange(n_points) * 100
    values = np.zeros((n_points, 6))

    risk_score, n_windows = loaded.score_ride(time_index, values, "rad_s")

    assert n_windows == 2
    assert isinstance(risk_score, float)
    assert 0.0 <= risk_score <= 1.0
