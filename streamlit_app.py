"""Streamlit entry point. Deploy target for Streamlit Community Cloud."""

from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "index.html"
BRIDGE = ROOT / "component" / "bridge.js"
BUILD = Path(tempfile.mkdtemp(prefix='hirely-component-'))
COUNTER = ROOT / "data" / "cache" / "analysis_count.txt"

REQUIRED_DATA = {
    "O*NET software skills": ROOT / "data" / "raw" / "onet" / "Software Skills.xlsx",
    "O*NET occupations": ROOT / "data" / "raw" / "onet" / "Occupation Data.xlsx",
    "O*NET essential skills": ROOT / "data" / "raw" / "onet" / "Essential Skills.xlsx",
    "BLS market data": ROOT / "data" / "raw" / "bls" / "occupation_market.csv",
}

st.set_page_config(page_title="Hirely — Resume Analysis", page_icon="📄",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """<style>
      [data-testid="stAppViewContainer"] { background: #fcfcfa; }
      [data-testid="stHeader"], [data-testid="stToolbar"],
      [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
      .block-container { padding: 0 !important; max-width: 100% !important; }
      [data-testid="stVerticalBlock"] { gap: 0 !important; }
      iframe { display: block; border: none; }
    </style>""",
    unsafe_allow_html=True,
)


def _load_secrets() -> None:
    """Streamlit keeps secrets outside the environment; the pipeline reads env."""
    try:
        for name in ("GROQ_API_KEY", "LLM_MODEL"):
            if name in st.secrets and not os.getenv(name):
                os.environ[name] = str(st.secrets[name])
    except Exception:
        pass
    os.environ.setdefault("LLM_CACHE", "0")


def _build_component() -> str:
    """
    Serve the real page as the component, with a bridge appended.

    The app is not rebuilt for Streamlit — the same index.html the FastAPI
    service serves is used verbatim, so the deployed page is the page. The
    bridge only replaces the one thing that cannot work inside a component:
    the upload POST becomes a component value, and the report comes back as a
    render argument.
    """
    html = FRONTEND.read_text()
    bridge = BRIDGE.read_text()
    (BUILD / "index.html").write_text(f"{html}\n<script>\n{bridge}\n</script>\n")
    return str(BUILD)


def _missing_data() -> list:
    return [name for name, path in REQUIRED_DATA.items() if not path.exists()]


def _read_count() -> int:
    try:
        return int(COUNTER.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def _bump_count() -> None:
    n = _read_count()
    try:
        COUNTER.parent.mkdir(parents=True, exist_ok=True)
        COUNTER.write_text(str(n + 1))
    except OSError:
        pass


def _analyse(payload: dict) -> dict:
    """
    Run the pipeline on the uploaded bytes.

    Deliberately uncached: caching would hold resume bytes for the session and
    the page promises the opposite.
    """
    from src.pipeline import build_report

    name = payload.get("filename") or "resume.txt"
    suffix = Path(name).suffix.lower() or ".txt"
    content = base64.b64decode(payload.get("data") or "")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        started = time.perf_counter()
        report = build_report(
            tmp_path,
            target_soc=(payload.get("target_soc") or "").strip() or None,
            job_description=(payload.get("job_description") or "").strip() or None,
            stage=(payload.get("stage") or "").strip() or None,
        )
        report["resume"] = name
        report["elapsed"] = round(time.perf_counter() - started, 1)
        _bump_count()
        return report
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


_load_secrets()

missing = _missing_data()
if missing:
    st.error(
        "This deployment is missing its datasets: " + ", ".join(missing) +
        ". Commit the files under `data/` listed in the README, or run the "
        "download scripts before deploying."
    )
    st.stop()

hirely = components.declare_component("hirely", path=_build_component())
report = st.session_state.get("report")
payload = hirely(
    report=report,
    elapsed=(report or {}).get("elapsed"),
    count=_read_count(),
    default=None,
)

if isinstance(payload, dict) and payload.get("action") == "reset":
    st.session_state.pop("report", None)
    st.session_state.pop("token", None)

elif isinstance(payload, dict) and payload.get("action") == "analyze":
    token = payload.get("data", "")[:64] + payload.get("target_soc", "") + \
        payload.get("stage", "") + payload.get("job_description", "")[:64]
    if st.session_state.get("token") != token:
        st.session_state["token"] = token
        with st.spinner(""):
            st.session_state["report"] = _analyse(payload)
        st.rerun()
