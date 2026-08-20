"""
Skill-capital scoring.

Plain coverage is economically naive: matching 15 of an occupation's 250 tools
reads as "6%" even when those 15 are the core, high-value, mutually reinforcing
tools of a specialist. This scores skill strength from three
labor-economics-motivated factors:

  breadth         value-weighted coverage — each tool weighted by inverse
                  occupation-frequency (rarer = stronger specialization signal,
                  the TF-IDF idea) and by O*NET hot / in-demand flags.
  specialization  mean rarity of the tools the candidate actually matches.
  complementarity how tightly the candidate's skills cluster in embedding space.
                  Complementary skills that amplify each other (skill
                  complementarity / O-ring intuition) score higher than a
                  scatter of unrelated keywords. This is the ML-driven term.

Coverage is measured over *informative* tools only. Skills the candidate's
resume implies rather than states (a PyTorch user has Python) count as held, and
commodity skills nobody bothers listing leave the denominator entirely instead
of counting against them — see skill_implication.py. Absence of a cheap skill is
a fact about resume writing, not about the candidate.

For the same reason the result is shrunk toward an average candidate when the
resume documents little: a short skills section is weak evidence, not evidence
of weakness.

Returns a 0-1 score plus the components, for transparency in the UI.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import posting_demand
from src.skill_matcher import _load_sw, get_occ_tools
from src.embeddings import _embed
from src.skill_implication import classify_occupation_tools, gap_priority

# Calibration (tuned against the sample resumes so specialists aren't penalized).
# Re-fit after implied-skill credit landed: crediting entailed skills and
# dropping commodity tools raises breadth for everyone, so the curve flattened
# to keep the top of the range unsaturated.
_K_BREADTH = 2.8     # curve on value-weighted coverage (which is fractional)
_A_SPEC = 0.40       # how much specialization amplifies
_A_COMP = 0.30       # how much complementarity amplifies
_N_FOCUS = 5         # priority gaps kept — a short list to act on, not a tally

# Thin evidence is not evidence of weakness. A resume listing a handful of tools
# tells us little either way, so the measured value is shrunk toward an average
# candidate in proportion to how much the resume actually demonstrates
# (empirical-Bayes style). Only well-documented profiles reach the extremes.
_PRIOR = 0.50            # assumed strength of a candidate we can't read
_EVIDENCE_HALF = 12.0    # credited tools at which the resume carries half weight


@lru_cache(maxsize=1)
def _idf_map() -> Tuple[Dict[str, float], float, float]:
    """
    tool_norm -> inverse-occupation-frequency weight, the max for scaling, and
    the value to assume for a skill O*NET has never heard of. Unknown skills
    come from job postings (LangChain, dbt) and are specialised by nature, so
    the default sits high rather than at the mean.
    """
    sw = _load_sw()
    n_occ = sw["soc_code"].nunique()
    counts = sw.groupby("tool_norm")["soc_code"].nunique()
    idf = {tn: math.log((n_occ + 1) / (1 + int(c))) for tn, c in counts.items()}
    values = list(idf.values()) or [1.0]
    return idf, max(values), float(np.percentile(values, 75))


def _demand_mult(
    tool_name: str, is_hot: bool, in_demand: bool, title: Optional[str]
) -> float:
    """
    O*NET's survey flags, scaled by how often live postings actually ask for
    the tool. The two signals disagree often — the survey lags — so posting
    demand multiplies rather than replaces it.
    """
    base = 1.6 if is_hot else (1.3 if in_demand else 1.0)
    return base * posting_demand.demand_multiplier(tool_name, title)


def _complementarity(user_skills: List[str]) -> float:
    """
    Mean *pairwise* cosine similarity across the candidate's (deduped) skills.
    A focused specialist's whole set coheres (high); a scatter of unrelated
    keywords does not (low). Mean-pairwise discriminates far better than
    nearest-neighbour, which saturates because everyone has some close pair.
    Near-duplicate pairs (>0.9) are dropped so "Python/Python3" doesn't inflate
    it. Neutral 0.5 when embeddings are unavailable.
    """
    uniq = [s for s in dict.fromkeys(user_skills) if s and s.strip()][:40]
    if len(uniq) < 3:
        return 0.5
    embs = _embed(uniq)
    if embs is None:
        return 0.5
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = embs @ embs.T
    iu = np.triu_indices(len(uniq), k=1)
    pairs = sims[iu]
    pairs = pairs[pairs < 0.9]  # drop near-duplicate pairs
    if pairs.size == 0:
        return 0.5
    mean_pair = float(np.mean(pairs))
    # observed range ~0.05-0.35 -> 0-1
    return float(np.clip((mean_pair - 0.05) / 0.30, 0.0, 1.0))


def compute_skill_capital(
    user_skills: List[str],
    soc_code: Optional[str] = None,
    tools: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score how strong the candidate's skills are against a target.

    The target is either an occupation (`soc_code`) or an explicit `tools`
    frame — the latter is how a pasted job posting gets scored with the same
    machinery.

    Returns {score, breadth, specialization, complementarity, matched_count,
    implied_count, ignored_basic, focus_skills}.
    """
    occ = tools if tools is not None else get_occ_tools(soc_code or "")
    occ = occ.drop_duplicates(subset=["tool_norm"])
    idf, idf_max, idf_default = _idf_map()
    status = classify_occupation_tools([str(t) for t in occ["tool_name"]], user_skills)

    total_value = matched_value = 0.0
    matched_idfs: List[float] = []
    matched_count = implied_count = ignored_basic = 0
    priority_gaps: List[Dict[str, Any]] = []

    for _, row in occ.iterrows():
        tn = str(row["tool_norm"])
        tool_name = str(row["tool_name"])
        cls = status.get(tool_name, {})
        state = cls.get("status", "gap")
        if state == "commodity":
            # Cheap and ubiquitous: no signal either way, so it never reaches
            # the denominator.
            ignored_basic += 1
            continue

        is_hot, in_demand = bool(row["is_hot"]), bool(row["in_demand"])
        val = idf.get(tn, idf_default) * _demand_mult(tool_name, is_hot, in_demand, title)
        total_value += val

        if state in ("held", "implied"):
            matched_value += val
            matched_idfs.append(idf.get(tn, idf_default))
            matched_count += 1
            if state == "implied":
                implied_count += 1
        else:
            demand_score = (2.0 if is_hot else 0.0) + (1.0 if in_demand else 0.0)
            share = posting_demand.demand_share(tool_name, title)
            if not (is_hot or in_demand or share):
                continue
            priority_gaps.append({
                "skill": tool_name,
                "is_hot": is_hot,
                "in_demand": in_demand,
                "market_share": share,
                "priority": gap_priority(demand_score, cls.get("relevance", 0.0), share),
            })

    breadth = (matched_value / total_value) if total_value else 0.0
    specialization = (float(np.mean(matched_idfs)) / idf_max) if matched_idfs else 0.0
    complementarity = _complementarity(user_skills)

    # Value-weighted breadth, curved up, then amplified by specialization and
    # complementarity — so a focused specialist isn't punished for narrow raw
    # coverage of a huge tool list.
    base = min(1.0, breadth * _K_BREADTH)
    measured = float(np.clip(
        base * (1.0 + _A_SPEC * specialization + _A_COMP * complementarity), 0.0, 1.0))

    # Shrink toward the prior when the resume documents little either way.
    evidence = matched_count / (matched_count + _EVIDENCE_HALF)
    score = evidence * measured + (1.0 - evidence) * _PRIOR

    priority_gaps.sort(key=lambda g: g["priority"], reverse=True)

    return {
        "score": round(score, 3),
        "measured": round(measured, 3),
        "evidence": round(evidence, 3),
        "breadth": round(breadth, 3),
        "specialization": round(specialization, 3),
        "complementarity": round(complementarity, 3),
        "matched_count": matched_count,
        "implied_count": implied_count,
        "ignored_basic": ignored_basic,
        "focus_skills": priority_gaps[:_N_FOCUS],
    }
