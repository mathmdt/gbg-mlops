"""The web app shows a verdict only for the file it was computed for."""

from __future__ import annotations

from types import SimpleNamespace

from app.session import current, store, upload_key

RIDE_A = SimpleNamespace(file_id="id-a", name="ride_a.csv", size=100)
RIDE_B = SimpleNamespace(file_id="id-b", name="ride_b.csv", size=200)
PREDICTION = {"risk_score": 0.8, "is_reckless": True}


def test_result_is_shown_for_the_file_it_was_computed_for():
    state = {}
    store(state, upload_key(RIDE_A), PREDICTION)
    assert current(state, upload_key(RIDE_A)) == PREDICTION


def test_uploading_another_ride_hides_the_previous_result():
    state = {}
    store(state, upload_key(RIDE_A), PREDICTION)
    assert current(state, upload_key(RIDE_B)) is None


def test_removing_the_file_hides_the_result():
    state = {}
    store(state, upload_key(RIDE_A), PREDICTION)
    assert current(state, upload_key(None)) is None


def test_a_failed_prediction_clears_the_result():
    state = {}
    store(state, upload_key(RIDE_A), PREDICTION)
    store(state, upload_key(RIDE_A), None)
    assert current(state, upload_key(RIDE_A)) is None
