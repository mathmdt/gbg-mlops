"""Which prediction the page may show: only the one made for the file currently uploaded.

Streamlit keeps `st.session_state` between reruns, so a result must stay attached to the
file it was computed for. Uploading another file (or removing it) hides the old verdict.
"""

from __future__ import annotations

from typing import Any


def upload_key(uploaded: Any) -> str | None:
    """Identity of the uploaded file: changes for every new upload, even of the same file."""
    if uploaded is None:
        return None
    return str(getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}")


def store(state: Any, key: str | None, result: dict | None) -> None:
    """Remember `result` as the prediction for the file `key`."""
    state["result"] = {"file": key, "body": result} if result is not None else None


def current(state: Any, key: str | None) -> dict | None:
    """The stored prediction if it belongs to the file `key`, else None."""
    stored = state.get("result")
    if key is None or not stored or stored.get("file") != key:
        return None
    return stored["body"]
