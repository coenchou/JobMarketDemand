"""Compute skill gaps between a user's skills and a target occupation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "onet"


@lru_cache(maxsize=1)
def _load_essential_skills() -> pd.DataFrame:
    """Load and cache Essential Skills importance scores (Scale ID = IM only)."""
    path = DATA_DIR / "Essential Skills.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code_full",
        "Element Name": "element_name",
        "Scale ID": "scale_id",
        "Data Value": "score",
    })
    df["soc_code"] = df["soc_code_full"].str[:7]
    return df[df["scale_id"] == "IM"][["soc_code", "element_name", "score"]].copy()


def compute_skill_gaps(
    user_skills: List[str],
    target_soc: Optional[str] = None,
    max_gaps: int = 25,
    tools: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
) -> Dict:
    """
    Compare the user's skills against what a target requires — an occupation
    (`target_soc`) or an explicit `tools` frame from a pasted job posting.

    Only genuine gaps are reported: tools the resume implies (a Kubernetes user
    runs Linux) count as strengths, and commodity tools nobody lists are dropped
    from both sides rather than held against the candidate — see
    skill_implication.py.

    Returns:
        soc_code: the target SOC
        coverage: fraction of informative occupation tools the user has
        strengths: tool names user has, or clearly implies, that the role values
        gaps: missing tools ranked by demand × relevance to this candidate
        abstract_skills_required: top essential/cognitive skills for the occupation
    """
    from src import posting_demand
    from src.skill_matcher import get_occ_tools  # avoid circular at module level
    from src.skill_implication import classify_occupation_tools, gap_priority

    occ_tools = tools if tools is not None else get_occ_tools(target_soc or "")
    if occ_tools.empty:
        return {
            "soc_code": target_soc,
            "coverage": 0.0,
            "strengths": [],
            "gaps": [],
            "abstract_skills_required": [],
        }

    # Deduplicate occupation tools by normalised name
    deduped = occ_tools.drop_duplicates(subset=["tool_norm"]).copy()
    status = classify_occupation_tools(
        [str(t) for t in deduped["tool_name"]], user_skills)

    strengths: List[Dict] = []
    gaps: List[Dict] = []
    ignored_basic = 0

    for _, row in deduped.iterrows():
        tool_name = str(row["tool_name"])
        cls = status.get(tool_name, {})
        state = cls.get("status", "gap")
        if state in ("held", "implied"):
            # `implied` marks a skill credited from something related rather
            # than claimed outright, so the report can show which is which.
            strengths.append({
                "skill": tool_name,
                "implied": state == "implied",
                "via": cls.get("anchor"),
            })
        elif state == "commodity":
            ignored_basic += 1
        else:
            demand_score = (2.0 if row["is_hot"] else 0.0) + (1.0 if row["in_demand"] else 0.0)
            share = posting_demand.demand_share(tool_name, title)
            gaps.append({
                "skill": tool_name,
                "element_category": str(row["element_name"]),
                "is_hot": bool(row["is_hot"]),
                "in_demand": bool(row["in_demand"]),
                "demand_score": demand_score,
                "market_share": share,
                "relevance": cls.get("relevance", 0.0),
                "priority": gap_priority(demand_score, cls.get("relevance", 0.0), share),
            })

    # Skills the live market asks for that O*NET never lists for this
    # occupation. The survey structurally cannot produce these, and they are
    # often the most current thing a candidate could learn.
    market_only = posting_demand.market_only_skills(
        title, [str(t) for t in deduped["tool_name"]])
    if market_only:
        names = [posting_demand.display_name(m["skill"]) for m in market_only]
        market_status = classify_occupation_tools(names, user_skills)
        for name, entry in zip(names, market_only):
            state = market_status.get(name, {}).get("status", "gap")
            if state != "gap":
                continue
            relevance = market_status.get(name, {}).get("relevance", 0.0)
            gaps.append({
                "skill": name,
                "element_category": "Live market demand",
                "is_hot": entry["share"] >= 0.25,
                "in_demand": True,
                "demand_score": 1.0,
                "market_share": entry["share"],
                "relevance": relevance,
                "priority": gap_priority(1.0, relevance, entry["share"]),
                "source": "postings",
            })

    # Rank by demand weighted by closeness to the candidate's own work, so the
    # list reads as "what to learn next" rather than "everything O*NET lists".
    gaps.sort(key=lambda x: (-x["priority"], x["element_category"], x["skill"]))

    # Deduplicate by element_category — keep the highest-demand tool per category
    seen_cats: Set[str] = set()
    deduped_gaps: List[Dict] = []
    for g in gaps:
        cat = g["element_category"]
        if cat not in seen_cats:
            seen_cats.add(cat)
            deduped_gaps.append(g)

    # Fall back to non-deduped list if we filtered too aggressively
    final_gaps = deduped_gaps if len(deduped_gaps) >= 5 else gaps
    final_gaps = final_gaps[:max_gaps]

    # Abstract essential skills for the occupation
    es = _load_essential_skills()
    occ_abstract = (
        es[es["soc_code"] == target_soc]
        .groupby("element_name")["score"]
        .mean()
        .reset_index()
        .sort_values("score", ascending=False)
    )
    abstract_skills = [
        {"skill": str(r["element_name"]), "importance": round(float(r["score"]), 2)}
        for _, r in occ_abstract.head(8).iterrows()
    ]

    # Commodity tools are excluded from the denominator, not counted as missing.
    total = max(1, len(deduped) - ignored_basic)
    matched_count = len(strengths)
    coverage = round(matched_count / total, 3)

    return {
        "soc_code": target_soc,
        "coverage": coverage,
        "matched_count": matched_count,
        "total_occ_tools": total,
        "ignored_basic": ignored_basic,
        # Claimed skills first, then inferred ones, alphabetical within each.
        "strengths": sorted(strengths, key=lambda s: (s["implied"], s["skill"].lower())),
        "gaps": final_gaps,
        "abstract_skills_required": abstract_skills,
    }
