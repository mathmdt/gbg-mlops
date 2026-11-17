"""Test web app: upload one ride and get its risk score from the API.

Run it with: uv run --group app streamlit run app/streamlit_app.py
The theme (colours, toolbar) lives in .streamlit/config.toml.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import altair as alt
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Streamlit puts app/ on the import path, not the repository root: add the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.client import UNITS, ApiError, get_health, predict_file  # noqa: E402
from app.route import route_map_html, simulated_route  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAFE, WARNING, DANGER = "#34d399", "#fbbf24", "#f43f5e"
WARNING_ICON = (
    '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20L12 3z"/>'
    '<path d="M12 10v4.5M12 17.5v.01"/></svg>'
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');
.hero, .card, .gauge, .empty, .footer { font-family: 'Inter', system-ui, sans-serif; }
.block-container { max-width: 1150px; padding-top: 3.5rem; }
.hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 30px 34px;
  background: radial-gradient(circle at 85% 20%, rgba(34,211,238,.35), transparent 45%),
              radial-gradient(circle at 10% 110%, rgba(168,85,247,.35), transparent 45%),
              linear-gradient(135deg, #111a33 0%, #0b1020 100%);
  border: 1px solid #1f2a44; margin-bottom: 26px;
}
.hero .eyebrow { color: #22d3ee; font-size: .78rem; letter-spacing: .22em;
  text-transform: uppercase; font-weight: 600; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; margin: 6px 0 4px;
  background: linear-gradient(90deg, #f8fafc, #67e8f9); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; padding: 0; }
.hero p { color: #94a3b8; margin: 0; max-width: 620px; }
.status { position: absolute; top: 28px; right: 30px; display: flex; align-items: center;
  gap: 8px; padding: 7px 14px; border-radius: 999px; font-size: .82rem; font-weight: 600;
  background: rgba(15,23,42,.7); border: 1px solid #1f2a44; }
.dot { width: 9px; height: 9px; border-radius: 50%; box-shadow: 0 0 12px currentColor; }
.step { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; color: #e2e8f0;
  margin: 0 0 6px; }
.step span { color: #22d3ee; margin-right: 8px; }
.empty { border: 1px dashed #1f2a44; border-radius: 16px; padding: 46px 20px;
  text-align: center; color: #64748b; }
.gauge { text-align: center; }
.gauge .value { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; }
.verdict { display: inline-block; margin-top: 4px; padding: 6px 16px; border-radius: 999px;
  font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .85rem; }
.cards { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.card { background: #131b2e; border: 1px solid #1f2a44; border-radius: 14px; padding: 14px 16px; }
.card .label { color: #64748b; font-size: .72rem; text-transform: uppercase;
  letter-spacing: .12em; }
.card .number { font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; margin-top: 2px;
  overflow-wrap: anywhere; }
.map-legend { display: flex; gap: 14px; color: #64748b; font-size: .8rem; margin-top: 4px; }
.map-legend i { display: inline-block; width: 14px; height: 4px; border-radius: 2px;
  margin-right: 6px; vertical-align: middle; }
.sim-banner { color: #fbbf24; font-size: .8rem; margin-top: 6px; display: flex; gap: 7px; }
.sim-banner svg { width: 16px; height: 16px; flex-shrink: 0; fill: none; stroke: currentColor;
  stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.footer { color: #475569; font-size: .78rem; text-align: center; margin-top: 40px; }
</style>
"""


def risk_color(score: float, threshold: float) -> str:
    if score >= threshold:
        return DANGER
    return WARNING if score >= 0.6 * threshold else SAFE


def model_status() -> str:
    """Pill shown in the hero: model served by the API, or why there is none."""
    try:
        with httpx.Client(base_url=API_URL, timeout=5) as client:
            health = get_health(client)
    except ApiError:
        return (
            f'<div class="status" style="color:{DANGER}"><span class="dot" '
            f'style="background:{DANGER}"></span>API offline</div>'
        )
    if health.get("status") != "ready":
        return (
            f'<div class="status" style="color:{WARNING}"><span class="dot" '
            f'style="background:{WARNING}"></span>No model loaded</div>'
        )
    return (
        f'<div class="status" style="color:{SAFE}"><span class="dot" style="background:{SAFE}">'
        f"</span>{health['model_name']} v{health['model_version']} · "
        f"@{health['model_alias']}</div>"
    )


def gauge(score: float, threshold: float) -> str:
    """Half-circle SVG gauge with the threshold marked."""
    color = risk_color(score, threshold)
    length = np.pi * 90
    filled = length * min(max(score, 0.0), 1.0)
    angle = np.pi * (1 - threshold)
    tick_x, tick_y = 110 + 102 * np.cos(angle), 110 - 102 * np.sin(angle)
    verdict = "Reckless" if score >= threshold else "Not reckless"
    return f"""
<div class="gauge">
  <svg viewBox="0 0 220 125" width="100%" style="max-width:330px">
    <path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="#1f2a44"
          stroke-width="16" stroke-linecap="round"/>
    <path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="{color}"
          stroke-width="16" stroke-linecap="round"
          stroke-dasharray="{filled:.1f} {length:.1f}"/>
    <circle cx="{tick_x:.1f}" cy="{tick_y:.1f}" r="4" fill="#e2e8f0"/>
  </svg>
  <div class="value" style="color:{color}; margin-top:-62px;
       text-shadow: 0 0 24px {color}88">{score:.3f}</div>
  <div style="color:#64748b; font-size:.8rem">risk score · threshold {threshold}</div>
  <div class="verdict" style="color:{color}; background:{color}22;
       border:1px solid {color}66">{verdict}</div>
</div>
"""


def card(label: str, value: object) -> str:
    return (
        f'<div class="card"><div class="label">{label}</div><div class="number">{value}</div></div>'
    )


@st.cache_data(show_spinner=False)
def read_ride(filename: str, content: bytes) -> pd.DataFrame | None:
    """Raw file as a table, for the chart only (the API does its own validation)."""
    try:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        return pd.read_parquet(io.BytesIO(content))
    except Exception:
        return None


def ride_chart(ride: pd.DataFrame) -> alt.Chart:
    """Norm of the acceleration and of the rotation over time."""
    signals = pd.DataFrame(
        {
            "time (s)": ride["time_index"] / 1000,
            "rotation": np.linalg.norm(ride[["rx", "ry", "rz"]].to_numpy(), axis=1),
            "acceleration": np.linalg.norm(ride[["ax", "ay", "az"]].to_numpy(), axis=1),
        }
    ).melt("time (s)", var_name="signal", value_name="norm")
    return (
        alt.Chart(signals)
        .mark_line(strokeWidth=1.3, opacity=0.9)
        .encode(
            x=alt.X("time (s):Q", title="time (s)"),
            y=alt.Y("norm:Q", title=None),
            color=alt.Color(
                "signal:N",
                scale=alt.Scale(domain=["acceleration", "rotation"], range=["#22d3ee", "#a855f7"]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(height=250)
    )


st.set_page_config(page_title="GBG ride risk", page_icon="🛴", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f"""
<div class="hero">
  {model_status()}
  <div class="eyebrow">Göteborg · IMU 10 Hz · MLOps</div>
  <h1>GBG Ride Risk</h1>
  <p>Drop a bike or scooter ride: the API cuts it into 30 s windows, the model scores each
  window, and the ride takes the highest score.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown('<p class="step"><span>01</span>Upload a ride</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Ride file", type=["csv", "parquet"], label_visibility="collapsed")
    units = st.segmented_control(
        "Gyroscope units",
        options=list(UNITS),
        format_func=UNITS.get,
        default="deg_s",
        help="GBG500 rides are in °/s, SRV3490 rides in rad/s.",
    )
    ride_id = st.text_input("Ride id", placeholder="optional, e.g. demo-ride-001")
    clicked = st.button(
        "Score the ride",
        type="primary",
        icon=":material/bolt:",
        disabled=uploaded is None or units is None,
        width="stretch",
    )

with right:
    st.markdown('<p class="step"><span>02</span>Ride signals</p>', unsafe_allow_html=True)
    ride = read_ride(uploaded.name, uploaded.getvalue()) if uploaded else None
    columns = {"time_index", "ax", "ay", "az", "rx", "ry", "rz"}
    if ride is not None and columns <= set(ride.columns):
        ride = ride.sort_values("time_index").fillna(0)
        route_tab, signals_tab = st.tabs(
            [":material/route: Route (simulated)", ":material/show_chart: Signals"]
        )
        with route_tab:
            rotation = np.linalg.norm(ride[["rx", "ry", "rz"]].to_numpy(), axis=1)
            components.html(route_map_html(simulated_route(rotation)), height=304)
            st.markdown(
                '<div class="map-legend"><span><i style="background:#34d399"></i>calm</span>'
                '<span><i style="background:#fbbf24"></i>agitated</span>'
                '<span><i style="background:#f43f5e"></i>most agitated</span></div>'
                f'<div class="sim-banner">{WARNING_ICON}Simulated route: the IMU data has no GPS. '
                "Position is "
                "illustrative, colours come from the real signals.</div>",
                unsafe_allow_html=True,
            )
        with signals_tab:
            st.altair_chart(ride_chart(ride), width="stretch")
            st.caption(f"{len(ride)} points · {len(ride) / 10:.0f} s · norms of the raw signals")
    else:
        st.markdown(
            '<div class="empty">The route and the signals of the ride appear here.</div>',
            unsafe_allow_html=True,
        )

if clicked:
    try:
        with httpx.Client(base_url=API_URL, timeout=60) as client:
            st.session_state.result = predict_file(
                client, uploaded.name, uploaded.getvalue(), units, ride_id or None
            )
    except ApiError as error:
        st.session_state.result = None
        st.error(str(error), icon=":material/error:")

result = st.session_state.get("result")
if result:
    st.markdown('<p class="step"><span>03</span>Verdict</p>', unsafe_allow_html=True)
    gauge_column, cards_column = st.columns([5, 7], gap="large")
    gauge_column.markdown(gauge(result["risk_score"], result["threshold"]), unsafe_allow_html=True)
    cards_column.markdown(
        '<div class="cards">'
        + card("Model", f"{result['model_name']} v{result['model_version']}")
        + card("Windows scored", result["n_windows"])
        + card("Ride", result.get("ride_id") or (uploaded.name if uploaded else "—"))
        + card("Prediction", str(result["prediction_id"])[:8])
        + "</div>",
        unsafe_allow_html=True,
    )
    with cards_column.expander("Full API response"):
        st.json(result)

st.markdown(
    f'<div class="footer">API {API_URL} · Albert School MLOps project · GBG500</div>',
    unsafe_allow_html=True,
)
