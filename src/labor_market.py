"""
BLS labor-market grounding for the two headline scores.

Loads the consolidated Employment Projections table (built by
scripts/build_bls_market.py) and turns a candidate's education / experience
into occupation-relative *fit* measures, plus a market *outlook* from real
growth and openings data — replacing the previously hand-tuned curves.

Everything degrades gracefully: if the SOC has no BLS row, callers fall back
to their prior heuristic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKET_CSV = ROOT / "data" / "raw" / "bls" / "occupation_market.csv"

# Ordinal ladders so "meets the bar" is comparable across schemes.
_EDU_RANK = {
    "No formal educational credential": 0,
    "High school diploma or equivalent": 1,
    "Some college, no degree": 2,
    "Postsecondary nondegree award": 2,
    "Associate's degree": 3,
    "Bachelor's degree": 4,
    "Master's degree": 5,
    "Doctoral or professional degree": 6,
}
# Candidate education labels emitted by the parser.
_CANDIDATE_EDU_RANK = {
    "Associate's": 3, "Bachelor's": 4, "Master's": 5, "Doctorate": 6,
}


@lru_cache(maxsize=1)
def _load_market() -> Dict[str, Dict[str, Any]]:
    if not MARKET_CSV.exists():
        return {}
    df = pd.read_csv(MARKET_CSV, dtype={"soc_code": str})
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        out[r["soc_code"]] = {
            "typical_education": None if pd.isna(r["typical_education"]) else r["typical_education"],
            "typical_experience": None if pd.isna(r["typical_experience"]) else r["typical_experience"],
            "typical_training": None if pd.isna(r["typical_training"]) else r["typical_training"],
            "growth_pct": None if pd.isna(r["growth_pct"]) else float(r["growth_pct"]),
            "openings_k": None if pd.isna(r["openings_k"]) else float(r["openings_k"]),
            "median_wage": None if pd.isna(r["median_wage"]) else int(r["median_wage"]),
            "current_emp_k": None if pd.isna(r["current_emp_k"]) else float(r["current_emp_k"]),
        }
    return out


def get_market(soc_code: str) -> Optional[Dict[str, Any]]:
    """Return the BLS market row for a SOC code, or None if unavailable."""
    return _load_market().get(soc_code)


# ---------------------------------------------------------------------------
# Fit measures (0-1), grounded in what the occupation actually requires
# ---------------------------------------------------------------------------

def education_fit(candidate_edu: Optional[str], required_edu: Optional[str]) -> Optional[float]:
    """
    1.0 if the candidate meets or exceeds the occupation's typical entry
    education; scaled down for each level short. None if data is missing.
    """
    if required_edu is None:
        return None
    req = _EDU_RANK.get(required_edu)
    if req is None:
        return None
    cand = _CANDIDATE_EDU_RANK.get(candidate_edu or "", 1)  # assume HS if unstated
    if cand >= req:
        return 1.0
    # each level below the requirement costs 0.22, floored at 0.3
    return max(0.3, 1.0 - 0.22 * (req - cand))


def experience_fit(candidate_years: Optional[int], required_exp: Optional[str]) -> Optional[float]:
    """
    Map candidate years against the occupation's typical work-experience
    requirement ("None" / "Less than 5 years" / "5 years or more").
    """
    if required_exp is None:
        return None
    y = candidate_years or 0
    if required_exp == "No experience required":
        return 1.0
    if required_exp == "Less than 5 years":
        # wants some related experience; ~2 yrs satisfies it
        return min(1.0, 0.5 + 0.25 * y)
    if required_exp == "5 years or more":
        return min(1.0, y / 5.0) if y else 0.2
    return None


def outlook_score(market: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Market favorability 0-1 from projected growth, nudged by openings volume.
    Centered so the ~4% all-occupation average lands near 0.5.
    """
    if not market or market.get("growth_pct") is None:
        return None
    growth = market["growth_pct"]
    g = max(0.1, min(1.0, 0.5 + growth / 20.0))  # 0%→0.5, +10%→1.0, -8%→0.1
    openings, emp = market.get("openings_k"), market.get("current_emp_k")
    if openings and emp and emp > 0:
        # openings-to-employment ratio; ~10%+ is healthy churn+growth
        ratio = openings / emp
        o = max(0.0, min(1.0, ratio / 0.12))
        return round(0.75 * g + 0.25 * o, 4)
    return round(g, 4)


def outlook_label(growth_pct: Optional[float]) -> str:
    if growth_pct is None:
        return "unknown"
    if growth_pct >= 8:
        return "much faster than average"
    if growth_pct >= 4:
        return "faster than average"
    if growth_pct >= 1:
        return "about average"
    if growth_pct > -1:
        return "little change"
    return "declining"
