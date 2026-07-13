"""FastAPI server — resume upload and full analysis endpoint."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_report
from src.job_search import find_jobs

app = FastAPI(title="Career Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = ROOT / "index.html"


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
def analyze_resume(file: UploadFile = File(...)) -> Dict[str, Any]:
    ext = Path(file.filename or "resume.txt").suffix.lower()
    if ext not in {".txt", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    content = file.file.read()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = build_report(tmp_path)
        report["resume"] = file.filename
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
