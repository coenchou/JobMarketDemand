"""FastAPI server — resume upload and full analysis endpoint."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_CACHE", "0")

from src.pipeline import build_report
from src.job_search import find_jobs

app = FastAPI(title="Hirely API")

COUNTER_FILE = ROOT / "data" / "cache" / "analysis_count.txt"

_COUNTER_LOCK = threading.Lock()

UPLOAD_DIR = Path(tempfile.gettempdir()) / "hirely-uploads"
shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _read_count() -> int:
    try:
        return int(COUNTER_FILE.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def _bump_count() -> int:
    with _COUNTER_LOCK:
        n = _read_count() + 1
        try:
            COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = COUNTER_FILE.with_suffix(".tmp")
            tmp.write_text(str(n))
            os.replace(tmp, COUNTER_FILE)
        except OSError:
            pass
    return n

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

FRONTEND = ROOT / "index.html"


REQUIRED_DATA = {
    "onet_software_skills": ROOT / "data" / "raw" / "onet" / "Software Skills.xlsx",
    "onet_occupations": ROOT / "data" / "raw" / "onet" / "Occupation Data.xlsx",
    "onet_essential_skills": ROOT / "data" / "raw" / "onet" / "Essential Skills.xlsx",
    "bls_market": ROOT / "data" / "raw" / "bls" / "occupation_market.csv",
}
OPTIONAL_DATA = {
    "posting_demand": ROOT / "data" / "processed" / "posting_skills.csv",
    "occupation_embeddings": ROOT / "data" / "processed" / "occ_embeddings.npz",
}


@app.get("/health")
def health() -> Dict[str, Any]:
    required = {k: v.exists() for k, v in REQUIRED_DATA.items()}
    optional = {k: v.exists() for k, v in OPTIONAL_DATA.items()}
    from src.embeddings import _load_model, embeddings_enabled

    return {
        "ok": all(required.values()),
        "required": required,
        "optional": optional,
        "llm": bool(os.getenv("GROQ_API_KEY")),
        "model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        "embeddings_enabled": embeddings_enabled(),
        # Reports whether the model is already resident. Calling _load_model()
        # here would download 90 MB and allocate half a gigabyte on every
        # health check, which is exactly what a health check must not do.
        "embeddings_loaded": _load_model.cache_info().currsize > 0,
    }


@app.get("/stats")
def stats() -> Dict[str, int]:
    """How many analyses have run. The only number this service remembers."""
    return {"analyses": _read_count()}


@app.get("/")
def serve_frontend():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"message": "Career Intelligence API — POST a resume to /analyze"}


class JobQuery(BaseModel):
    title: str
    skills: list[str] = []
    remote: bool = True
    location: str = ""
    top_n: int = 5


@app.post("/jobs")
def job_search(q: JobQuery) -> Dict[str, Any]:
    try:
        return find_jobs(
            q.title,
            q.skills,
            remote=q.remote,
            location=q.location,
            top_n=max(1, min(q.top_n, 8)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job search failed: {e}")


@app.post("/analyze")
def analyze_resume(
    file: UploadFile = File(...),
    target_soc: str = Form(""),
    job_description: str = Form(""),
) -> Dict[str, Any]:
    """
    Analyse a resume. By default it is scored against the best-matching
    occupation; `target_soc` pins a different one and `job_description` scores
    it against a pasted posting instead.
    """
    ext = Path(file.filename or "resume.txt").suffix.lower()
    if ext not in {".txt", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    content = file.file.read()

    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=UPLOAD_DIR)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(content)
        report = build_report(
            tmp_path,
            target_soc=target_soc.strip() or None,
            job_description=job_description.strip() or None,
        )
        report["resume"] = file.filename
        _bump_count()
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
