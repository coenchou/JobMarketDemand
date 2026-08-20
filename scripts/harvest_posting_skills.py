"""
Harvest what live job postings actually ask for.

O*NET's technology lists are survey-derived and lag the market badly: they will
tell a machine-learning engineer to learn Microsoft Project, and they have
never heard of dbt, GraphQL or LangChain. This walks The Muse's public catalog
(keyless), extracts skill mentions from each posting with the same vocabulary
the report uses, and writes per-category demand shares.

    python -m scripts.harvest_posting_skills --pages 40 --out data/processed/posting_skills.csv

Output columns:
    category   The Muse category the posting was filed under
    skill      canonical skill name
    postings   how many postings in that category mention it
    total      postings scanned in that category
    share      postings / total

Re-run it periodically; the report degrades gracefully to O*NET-only scoring
when the file is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from html import unescape
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MUSE_API = "https://www.themuse.com/api/public/jobs"
DEFAULT_OUT = ROOT / "data" / "processed" / "posting_skills.csv"

MIN_POSTINGS = 3


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


CATEGORIES: Tuple[str, ...] = (
    "Software Engineering",
    "Data and Analytics",
    "Science and Engineering",
    "IT",
    "Business Operations",
    "Project Management",
    "Sales",
    "Advertising and Marketing",
    "Accounting and Finance",
    "Human Resources and Recruitment",
    "Healthcare",
    "Design and UX",
    "Education",
)


def fetch_pages(
    pages: int,
    category: Optional[str] = None,
    delay: float = 0.4,
    timeout: float = 15.0,
) -> Iterator[dict]:
    """Yield raw postings from the public catalog, newest first."""
    for page in range(pages):
        params = {"page": page, "descending": "true"}
        if category:
            params["category"] = category
        url = f"{MUSE_API}?{urlencode(params)}"
        try:
            req = Request(url, headers={"User-Agent": "Hirely/1.0 (skill research)"})
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (URLError, ValueError, TimeoutError, OSError) as e:
            print(f"[harvest] page {page} failed: {e}", file=sys.stderr)
            break
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            yield r
        if data.get("page_count") and page + 1 >= data["page_count"]:
            break
        time.sleep(delay)


def harvest(pages: int) -> List[Dict[str, object]]:
    from src.target_role import extract_posting_skills

    per_category: Dict[str, Set[str]] = defaultdict(set)
    hits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scanned = 0

    for category in CATEGORIES:
        before = scanned
        for job in fetch_pages(pages, category=category):
            text = _strip_html(job.get("contents", ""))
            if len(text) < 200:
                continue
            job_id = str(job.get("id", scanned))
            scanned += 1

            per_category[category].add(job_id)
            for name, _ in extract_posting_skills(text):
                hits[category][name] += 1
        print(f"[harvest] {category}: {scanned - before} postings", flush=True)

    rows: List[Dict[str, object]] = []
    for category, skill_counts in hits.items():
        total = len(per_category[category])
        if total < MIN_POSTINGS:
            continue
        for skill, n in skill_counts.items():
            if n < MIN_POSTINGS:
                continue
            rows.append({
                "category": category,
                "skill": skill,
                "postings": n,
                "total": total,
                "share": round(n / total, 4),
            })

    rows.sort(key=lambda r: (r["category"], -float(r["share"])))
    print(f"[harvest] {scanned} postings scanned, {len(rows)} category/skill rows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=12,
                    help="pages per category (20 postings each)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rows = harvest(args.pages)
    if not rows:
        print("[harvest] nothing harvested; leaving existing file alone", file=sys.stderr)
        return

    import pandas as pd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[harvest] wrote {out}")


if __name__ == "__main__":
    main()
