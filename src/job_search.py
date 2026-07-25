"""
Live job matching.

Pipeline:
  1. fetch listings from The Muse public API (keyless), filtered by
     coarse category + location / remote
  2. pre-rank them by embedding similarity to the candidate's profile
     (reuses the sentence-transformers model in nlp_skills)
  3. run a single Groq/Llama pass over the top N to produce, per job,
     a fit score, a one-line rationale, and matching / missing skills

Every stage degrades gracefully: no network → empty list; no Groq key
→ embedding-only ranking with heuristic skill overlap.
"""

from __future__ import annotations

import json
import os
import re
import sys
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from src.nlp_skills import _embed, _norm

MUSE_API = "https://www.themuse.com/api/public/jobs"

# Matched-occupation title  →  a high-confidence The Muse category.
# Omitted when nothing matches (we then fetch broadly and let ranking sort it out).
_CATEGORY_KEYWORDS: List[tuple] = [
    ("Software Engineering", ("software", "developer", "programmer", "web develop", "devops", "engineer, computer")),
    ("Data Science",         ("data scien", "data analyst", "machine learning", "statistician", "data engineer")),
    ("Data and Analytics",   ("analytics", "business intelligence", "database")),
    ("Design and UX",        ("designer", "ux", "ui ", "graphic")),
    ("Product Management",    ("product manager", "product owner")),
    ("Project Management",    ("project manager", "program manager", "scrum")),
    ("Sales",                ("sales", "account executive", "business development")),
    ("Marketing",            ("marketing", "seo", "content strateg", "social media")),
    ("Finance",              ("financial", "accountant", "accounting", "auditor", "finance")),
    ("Human Resources",       ("human resources", "recruiter", "talent")),
    ("Customer Service",      ("customer service", "support specialist", "customer success")),
    ("Healthcare",           ("nurse", "physician", "therapist", "medical", "clinical")),
    ("Education",            ("teacher", "professor", "instructor", "tutor")),
]


def _title_to_category(title: str) -> Optional[str]:
    t = title.lower()
    for category, needles in _CATEGORY_KEYWORDS:
        if any(n in t for n in needles):
            return category
    return None


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------------------------

def fetch_jobs(
    title: str,
    *,
    remote: bool = True,
    location: str = "",
    pages: int = 2,
    timeout: float = 8.0,
) -> List[Dict[str, Any]]:
    """Fetch raw job listings from The Muse. Returns [] on any network failure."""
    category = _title_to_category(title)
    jobs: List[Dict[str, Any]] = []

    for page in range(pages):
        params = {"page": page, "descending": "true"}
        if category:
            params["category"] = category
        if remote:
            params["location"] = "Flexible / Remote"
        elif location:
            params["location"] = location

        url = f"{MUSE_API}?{urlencode(params)}"
        try:
            req = Request(url, headers={"User-Agent": "Hirely/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (URLError, ValueError, TimeoutError, OSError) as e:
            print(f"[job_search] fetch failed (page {page}): {e}", file=sys.stderr)
            break

        for r in data.get("results", []):
            desc = _strip_html(r.get("contents", ""))
            if not desc:
                continue
            locs = [l.get("name", "") for l in r.get("locations", [])]
            jobs.append({
                "title": r.get("name", ""),
                "company": (r.get("company") or {}).get("name", ""),
                "locations": locs,
                "location": ", ".join(locs) if locs else ("Remote" if remote else ""),
                "url": (r.get("refs") or {}).get("landing_page", ""),
                "category": category or "",
                "level": ", ".join(l.get("name", "") for l in r.get("levels", [])),
                "description": desc,
                "posted": r.get("publication_date", ""),
            })

        if data.get("page_count") and page + 1 >= data["page_count"]:
            break

    return jobs


# ---------------------------------------------------------------------------
# 2. Embedding pre-rank
# ---------------------------------------------------------------------------

def _profile_text(title: str, skills: List[str]) -> str:
    return f"{title}. Skills: {', '.join(skills[:40])}"


def rank_jobs(
    title: str,
    skills: List[str],
    jobs: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Attach a 0-1 `similarity` to each job and return the top_n."""
    if not jobs:
        return []

    profile = _profile_text(title, skills)
    texts = [profile] + [f"{j['title']}. {j['description'][:600]}" for j in jobs]
    embs = _embed(texts)

    if embs is not None:
        prof_emb, job_embs = embs[0], embs[1:]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.nan_to_num(job_embs @ prof_emb, nan=0.0, posinf=0.0, neginf=0.0)
        for j, s in zip(jobs, sims):
            j["similarity"] = round(float(np.clip(s, 0.0, 1.0)), 4)
    else:
        # keyword fallback: token overlap with skills
        skill_words = {w for s in skills for w in _norm(s).split() if len(w) >= 3}
        for j in jobs:
            jw = set(_norm(j["description"]).split())
            hits = len(skill_words & jw)
            j["similarity"] = round(min(1.0, hits / max(8, len(skill_words) or 1)), 4)

    jobs.sort(key=lambda x: x["similarity"], reverse=True)
    return jobs[:top_n]


# ---------------------------------------------------------------------------
# 3. Groq per-job fit analysis (single batched call)
# ---------------------------------------------------------------------------

_FIT_PROMPT = """You are a technical recruiter. A candidate is looking for work.

CANDIDATE
Target role: {title}
Skills: {skills}

JOBS (each has an index):
{jobs}

For EACH job, judge how well this candidate fits. Return ONLY JSON of the form:
{{"analyses": [
  {{"index": 0,
    "fit": 0-100,
    "summary": "one concrete sentence on why this fits or doesn't, referencing the candidate's actual skills",
    "matching_skills": ["skills the candidate has that this job wants"],
    "missing_skills": ["important things this job wants that the candidate lacks"]}}
]}}

Rules: base `fit` on real skill overlap and seniority, not enthusiasm.
Keep each list to at most 5 items. Do not invent skills the candidate did not list."""


def analyze_fit(title: str, skills: List[str], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add fit/summary/matching/missing to each job via Groq; heuristic fallback otherwise."""
    if not jobs:
        return jobs

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
            job_lines = "\n".join(
                f"[{i}] {j['title']} at {j['company'] or 'n/a'} — {j['description'][:500]}"
                for i, j in enumerate(jobs)
            )
            prompt = _FIT_PROMPT.format(
                title=title or "unspecified",
                skills=", ".join(skills[:40]) or "none listed",
                jobs=job_lines,
            )
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            parsed = json.loads(resp.choices[0].message.content)
            analyses = parsed.get("analyses", parsed if isinstance(parsed, list) else [])
            by_index = {a.get("index", n): a for n, a in enumerate(analyses)}
            for i, j in enumerate(jobs):
                a = by_index.get(i, {})
                j["fit"] = int(a.get("fit", round(j.get("similarity", 0) * 100)))
                j["summary"] = str(a.get("summary", "")).strip()
                j["matching_skills"] = [str(s) for s in a.get("matching_skills", [])][:5]
                j["missing_skills"] = [str(s) for s in a.get("missing_skills", [])][:5]
            return jobs
        except Exception as e:
            print(f"[job_search] Groq fit analysis failed: {e}", file=sys.stderr)

    # Heuristic fallback (no key or call failed)
    skill_set = {_norm(s) for s in skills}
    for j in jobs:
        jd = _norm(j["description"])
        matching = sorted({s for s in skills if _norm(s) in jd}, key=len, reverse=True)[:5]
        j["fit"] = round(j.get("similarity", 0) * 100)
        j["summary"] = ""
        j["matching_skills"] = matching
        j["missing_skills"] = []
    return jobs


# ---------------------------------------------------------------------------
# External job-board deep links
#
# The Muse has a thin, mostly-remote catalog, so it is unreliable for on-site
# search. Rather than scrape LinkedIn/Indeed (against ToS, fragile), we build
# pre-filled search URLs that drop the user straight into those sites with the
# right role + location applied.
# ---------------------------------------------------------------------------

def build_search_links(title: str, remote: bool, location: str) -> List[Dict[str, str]]:
    q = title or "jobs"
    loc = "Remote" if remote else (location or "")
    linkedin = f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(q)}"
    if remote:
        linkedin += "&f_WT=2"          # LinkedIn remote work-type filter
    elif loc:
        linkedin += f"&location={quote_plus(loc)}"
    indeed = f"https://www.indeed.com/jobs?q={quote_plus(q)}"
    if loc:
        indeed += f"&l={quote_plus(loc)}"
    google = f"https://www.google.com/search?q={quote_plus(q + ' jobs ' + loc)}&ibp=htl;jobs"
    return [
        {"site": "LinkedIn", "url": linkedin},
        {"site": "Indeed", "url": indeed},
        {"site": "Google Jobs", "url": google},
    ]


def _filter_by_location(jobs: List[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
    """Keep only jobs whose location text contains the queried city."""
    city = location.split(",")[0].strip().lower()
    if not city:
        return jobs
    return [j for j in jobs if city in j.get("location", "").lower()]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def find_jobs(
    title: str,
    skills: List[str],
    *,
    remote: bool = True,
    location: str = "",
    top_n: int = 5,
) -> Dict[str, Any]:
    raw = fetch_jobs(title, remote=remote, location=location)
    # On-site: The Muse leaks remote/multi-city results, so keep only true local
    # matches; the deep links cover everything the thin catalog misses.
    if not remote and location:
        raw = _filter_by_location(raw, location)
    ranked = rank_jobs(title, skills, raw, top_n=top_n)
    analyzed = analyze_fit(title, skills, ranked)
    analyzed.sort(key=lambda x: x.get("fit", 0), reverse=True)
    return {
        "query": {"title": title, "remote": remote, "location": location},
        "fetched": len(raw),
        "search_links": build_search_links(title, remote, location),
        "jobs": [
            {
                "title": j["title"],
                "company": j["company"],
                "location": j["location"],
                "url": j["url"],
                "level": j["level"],
                "posted": j["posted"][:10] if j.get("posted") else "",
                "fit": j.get("fit", 0),
                "summary": j.get("summary", ""),
                "matching_skills": j.get("matching_skills", []),
                "missing_skills": j.get("missing_skills", []),
            }
            for j in analyzed
        ],
    }
