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

# The landing page promises that nothing resume-derived is stored. The LLM
# cache would break that promise — cached responses contain the skills and
# experience extracted from the resume — so the server runs with it off.
# The CLI keeps caching (regenerating dev fixtures is where it pays for
# itself); export LLM_CACHE=1 to re-enable here if you accept the tradeoff.
os.environ.setdefault("LLM_CACHE", "0")

from src.pipeline import build_report
from src.job_search import find_jobs

app = FastAPI(title="Hirely API")

# Privacy model: uploads live in a temp file for the seconds the analysis
# takes, then are unlinked. The only thing that persists is this counter —
# one integer, no content, no filenames.
COUNTER_FILE = ROOT / "data" / "cache" / "analysis_count.txt"

# Requests run on parallel threadpool threads, so the read-modify-write needs
# a lock, and the write must be atomic (write_text truncates first — a reader
# landing in that window would see an empty file and reset the count to zero).
_COUNTER_LOCK = threading.Lock()

# Uploads get their own directory so a crash can't strand a resume among
# files we don't own. Anything left from a previous run — the one way the
# in-request cleanup can be skipped (power loss, SIGKILL) — is purged the
# moment the server starts.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = ROOT / "index.html"


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

    # The path exists before the first resume byte is written, and the finally
    # owns it from that moment — so no failure between write and analysis can
    # strand resume content on disk. (A hard kill mid-request is covered by the
    # startup purge of UPLOAD_DIR.)
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
