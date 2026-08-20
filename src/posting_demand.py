"""
Live-market demand, from job postings rather than survey data.

O*NET's hot / in-demand flags come from a slow survey cycle: they miss whole
categories of current tooling and keep recommending software the market has
moved past. `scripts/harvest_posting_skills.py` walks a public job catalog and
records, per occupational category, the share of postings that mention each
skill. This module turns that into two signals:

  demand_share(skill, title)   0-1, how often postings for this kind of role
                               ask for the skill — used to weight tools and
                               rank gaps by real demand.
  market_only_skills(title)    skills postings ask for that O*NET does not list
                               for the occupation at all — the gaps the
                               occupational data structurally cannot see.

Everything returns neutral values when the harvest file is absent, so the
system runs unchanged without it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.embeddings import _norm

ROOT = Path(__file__).resolve().parents[1]
POSTING_CSV = ROOT / "data" / "processed" / "posting_skills.csv"

# Occupation title keyword → posting category. The catalog's own category names
# (as returned by the API, not the stale list job_search.py used to guess with).
_TITLE_TO_CATEGORY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Data and Analytics", ("data scien", "data analyst", "analytics", "statistic",
                            "machine learning", "data engineer", "business intelligence",
                            "database", "research scientist")),
    ("Software Engineering", ("software", "developer", "programmer", "web develop",
                              "devops", "systems engineer", "computer network",
                              "information security", "computer program")),
    ("Science and Engineering", ("engineer", "scientist", "chemist", "biologist",
                                 "environmental", "epidemiolog", "physicist")),
    ("IT", ("information technology", "support specialist", "systems administrator",
            "network administrator", "computer user support")),
    ("Project Management", ("project manager", "program manager", "scrum", "product manager")),
    ("Advertising and Marketing", ("marketing", "advertis", "seo", "content strateg",
                                   "social media", "public relations")),
    ("Sales", ("sales", "account executive", "business development", "retail")),
    ("Accounting and Finance", ("account", "financial", "auditor", "finance", "actuar")),
    ("Human Resources and Recruitment", ("human resources", "recruit", "talent")),
    ("Business Operations", ("operations", "management analyst", "logistic", "supply chain",
                             "administrative")),
    ("Healthcare", ("nurse", "physician", "therapist", "medical", "clinical", "health")),
    ("Education", ("teacher", "professor", "instructor", "tutor", "education")),
    ("Design and UX", ("designer", "ux", "user experience", "graphic")),
    ("Legal Services", ("lawyer", "attorney", "paralegal", "legal")),
)


def category_for_title(title: Optional[str]) -> Optional[str]:
    """Map an occupation title onto a posting category, or None if unclear."""
    if not title:
        return None
    t = title.lower()
    for category, needles in _TITLE_TO_CATEGORY:
        if any(n in t for n in needles):
            return category
    return None


@lru_cache(maxsize=1)
def _load() -> Dict[str, Dict[str, float]]:
    """category → {normalised skill: share of postings mentioning it}."""
    if not POSTING_CSV.exists():
        return {}
    try:
        df = pd.read_csv(POSTING_CSV)
    except (OSError, ValueError):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for row in df.itertuples():
        out.setdefault(str(row.category), {})[_norm(str(row.skill))] = float(row.share)
    return out


def available() -> bool:
    """True when harvested posting data is present."""
    return bool(_load())


def demand_share(skill: str, title: Optional[str]) -> float:
    """
    Share of postings for this kind of role that mention the skill (0-1).
    Falls back to the all-category average, then 0.0.
    """
    data = _load()
    if not data:
        return 0.0
    key = _norm(skill)
    category = category_for_title(title)
    if category and category in data:
        hit = data[category].get(key)
        if hit is not None:
            return hit
    shares = [d[key] for d in data.values() if key in d]
    return round(sum(shares) / len(shares), 4) if shares else 0.0


def demand_multiplier(skill: str, title: Optional[str]) -> float:
    """
    1.0 when postings never mention the skill, rising to ~1.5 for a skill
    named in most postings. Multiplies O*NET's own hot / in-demand weighting
    rather than replacing it — the two disagree often, and neither is ground
    truth on its own.
    """
    return 1.0 + 0.5 * min(1.0, demand_share(skill, title) / 0.5)


def market_only_skills(
    title: Optional[str],
    known_tools: List[str],
    top_n: int = 8,
    min_share: float = 0.06,
) -> List[Dict[str, float]]:
    """
    Skills postings ask for that the occupation's O*NET tool list omits
    entirely — ranked by share. These are the gaps survey data cannot produce.
    """
    data = _load()
    category = category_for_title(title)
    if not data or not category or category not in data:
        return []

    known = {_norm(t) for t in known_tools}
    out = [
        {"skill": skill, "share": share}
        for skill, share in data[category].items()
        if share >= min_share and skill not in known
    ]
    out.sort(key=lambda d: -d["share"])
    return out[:top_n]


def display_name(normalised: str) -> str:
    """Recover a presentable name for a normalised skill key."""
    from src.target_role import _skill_vocabulary
    return _skill_vocabulary().get(normalised, normalised.title())
