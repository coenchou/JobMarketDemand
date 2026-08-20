"""Streamlit entry point. Deploy target for Streamlit Community Cloud."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "index.html"

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
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding: 1.2rem 1rem 0 1rem; max-width: 1100px;}
    </style>""",
    unsafe_allow_html=True,
)


def _load_key() -> None:
    """Streamlit stores secrets outside the environment; the pipeline reads env."""
    try:
        for name in ("GROQ_API_KEY", "LLM_MODEL"):
            if name in st.secrets and not os.getenv(name):
                os.environ[name] = str(st.secrets[name])
    except Exception:
        pass
    os.environ.setdefault("LLM_CACHE", "0")


COUNTER = ROOT / "data" / "cache" / "analysis_count.txt"


def _bump_count() -> int:
    try:
        n = int(COUNTER.read_text().strip() or 0)
    except (OSError, ValueError):
        n = 0
    n += 1
    try:
        COUNTER.parent.mkdir(parents=True, exist_ok=True)
        COUNTER.write_text(str(n))
    except OSError:
        pass
    return n


def _read_count() -> int:
    try:
        return int(COUNTER.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def _missing_data() -> list:
    return [name for name, path in REQUIRED_DATA.items() if not path.exists()]


def _analyse(content: bytes, suffix: str, job_description: str) -> dict:
    """
    Deliberately uncached. Caching would hold resume bytes in memory for the
    session, and the page promises the opposite.
    """
    from src.pipeline import build_report

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        return build_report(tmp_path, job_description=job_description or None)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _render_report(report: dict) -> None:
    """
    Reuse the existing report UI. It already renders from a report object, so
    the page is handed the JSON directly instead of fetching it — the controls
    that would need an API are hidden, and switching between roles still works
    because every role's analysis travels inside the report.
    """
    html = FRONTEND.read_text()
    payload = json.dumps(report)
    html += f"""
<script>
window.addEventListener('load', function () {{
  renderReport({payload}, null);
  ['rtJd', 'secJobs'].forEach(function (id) {{
    var el = document.getElementById(id);
    if (el) el.style.display = 'none';
  }});
  document.querySelectorAll('.rt-toggle, .rpt-actions').forEach(function (el) {{
    el.style.display = 'none';
  }});
}});
</script>"""
    st.components.v1.html(html, height=3800, scrolling=True)


_load_key()

missing = _missing_data()
if missing:
    st.error(
        "This deployment is missing its datasets: " + ", ".join(missing) +
        ". Commit the files under `data/` listed in the README, or run the "
        "download scripts before deploying."
    )
    st.stop()

if "report" not in st.session_state:
    st.title("Hirely")
    st.caption("AI-powered tailored resume advice using real world labor-market data.")
    uploaded = st.file_uploader("Resume", type=["pdf", "txt"], label_visibility="collapsed")
    with st.expander("Score against a specific job posting instead"):
        jd = st.text_area("Paste the posting", height=160, label_visibility="collapsed")
    analysed = _read_count()
    st.caption(
        (f"{analysed:,} resumes analyzed so far — none kept. " if analysed else "")
        + "Your resume is never stored: it is analyzed in memory and discarded "
        "once the report is built."
    )
    if uploaded is not None and st.button("Run analysis", type="primary"):
        with st.spinner("Reading the resume, matching occupations, scoring…"):
            st.session_state["report"] = _analyse(
                uploaded.getvalue(), Path(uploaded.name).suffix.lower(), jd)
            st.session_state["report"]["resume"] = uploaded.name
            _bump_count()
        st.rerun()
else:
    if st.button("← New analysis"):
        del st.session_state["report"]
        st.rerun()
    _render_report(st.session_state["report"])
