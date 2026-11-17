"""Test web app: upload one ride and get its risk score from the API.

Run it with: uv run --group app streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# Streamlit puts app/ on the import path, not the repository root: add the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.client import UNITS, ApiError, get_health, predict_file  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="GBG ride risk", page_icon="🛴")

with st.sidebar:
    st.header("Model")
    st.caption(f"API: {API_URL}")
    try:
        with httpx.Client(base_url=API_URL, timeout=5) as client:
            health = get_health(client)
        if health.get("status") == "ready":
            st.success(f"Ready: {health['model_name']} version {health['model_version']}")
        else:
            st.warning("API up, but no model loaded")
            st.caption(health.get("detail") or "")
    except ApiError as error:
        st.error(str(error))

st.title("GBG ride risk")
st.write("Upload one ride (CSV or Parquet, 10 Hz) to get its risk score.")

uploaded = st.file_uploader("Ride file", type=["csv", "parquet"])
units = st.radio("Gyroscope units", options=list(UNITS), format_func=UNITS.get, horizontal=True)
ride_id = st.text_input("Ride id (optional)")

if st.button("Score the ride", type="primary", disabled=uploaded is None):
    content = uploaded.getvalue()
    try:
        with httpx.Client(base_url=API_URL, timeout=60) as client:
            result = predict_file(client, uploaded.name, content, units, ride_id or None)
    except ApiError as error:
        st.error(str(error))
    else:
        if result["is_reckless"]:
            st.error(f"Reckless ride: risk score {result['risk_score']:.3f}")
        else:
            st.success(f"Not reckless: risk score {result['risk_score']:.3f}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk score", f"{result['risk_score']:.3f}")
        col2.metric("Threshold", result["threshold"])
        col3.metric("Model version", result["model_version"])
        with st.expander("Full API response"):
            st.json(result)
