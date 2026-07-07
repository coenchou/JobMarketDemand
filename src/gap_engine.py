"""Compute skill gaps between a user's skills and a target occupation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

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


def _user_skill_words(skills: List[str]) -> Set[str]:
    """Build the set of significant words across all user skills."""
    import re
    words: Set[str] = set()
    for s in skills:
        for w in re.sub(r"\W+", " ", s).lower().split():
            if len(w) >= 3:
                words.add(w)
    return words


def compute_skill_gaps(
    user_skills: List[str],
    target_soc: str,
    max_gaps: int = 25,
) -> Dict:
    """
    Compare the user's skills against what a target occupation requires.

    Returns:
        soc_code: the target SOC
        coverage: fraction of occupation tools the user already has
        strengths: tool names user has that the occupation values
        gaps: ranked list of missing tools (hot/in-demand first)
        abstract_skills_required: top essential/cognitive skills for the occupation
    """
    from src.skill_matcher import get_occ_tools  # avoid circular at module level

    occ_tools = get_occ_tools(target_soc)
    if occ_tools.empty:
        return {
            "soc_code": target_soc,
            "coverage": 0.0,
            "strengths": [],
            "gaps": [],
            "abstract_skills_required": [],
        }

    user_words = _user_skill_words(user_skills)

    # Deduplicate occupation tools by normalised name
    deduped = occ_tools.drop_duplicates(subset=["tool_norm"]).copy()

    strengths: List[str] = []
    gaps: List[Dict] = []

    for _, row in deduped.iterrows():
        tool_words = set(row["tool_norm"].split())
        # A user "has" a tool if any significant word of the tool name appears
        # in the union of their skill words
        matched = bool(tool_words & user_words)
        if matched:
            strengths.append(str(row["tool_name"]))
        else:
            demand_score = (2.0 if row["is_hot"] else 0.0) + (1.0 if row["in_demand"] else 0.0)
            gaps.append({
                "skill": str(row["tool_name"]),
                "element_category": str(row["element_name"]),
                "is_hot": bool(row["is_hot"]),
                "in_demand": bool(row["in_demand"]),
                "demand_score": demand_score,
            })

    # Sort: hot first, then in-demand, then by element category to group related tools
    gaps.sort(key=lambda x: (-x["demand_score"], x["element_category"], x["skill"]))

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

    total = len(deduped)
    matched_count = len(strengths)
    coverage = round(matched_count / max(1, total), 3)

    return {
        "soc_code": target_soc,
        "coverage": coverage,
        "matched_count": matched_count,
        "total_occ_tools": total,
        "strengths": sorted(strengths),
        "gaps": final_gaps,
        "abstract_skills_required": abstract_skills,
    }
