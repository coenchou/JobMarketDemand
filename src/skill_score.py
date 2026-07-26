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

Returns a 0-1 score plus the components, for transparency in the UI.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np

from src.skill_matcher import _load_sw, get_occ_tools, _norm
from src.nlp_skills import _embed

# Calibration (tuned against the sample resumes so specialists aren't penalized)
_K_BREADTH = 2.3     # curve on value-weighted coverage (which is fractional)
_A_SPEC = 0.40       # how much specialization amplifies
_A_COMP = 0.30       # how much complementarity amplifies


@lru_cache(maxsize=1)
def _idf_map() -> Tuple[Dict[str, float], float]:
    """tool_norm -> inverse-occupation-frequency weight, plus the max for scaling."""
    sw = _load_sw()
    n_occ = sw["soc_code"].nunique()
    counts = sw.groupby("tool_norm")["soc_code"].nunique()
    idf = {tn: math.log((n_occ + 1) / (1 + int(c))) for tn, c in counts.items()}
    return idf, (max(idf.values()) if idf else 1.0)


def _demand_mult(is_hot: bool, in_demand: bool) -> float:
    return 1.6 if is_hot else (1.3 if in_demand else 1.0)


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
    with np.errstate(invalid="ignore", over="ignore"):
        sims = embs @ embs.T
    iu = np.triu_indices(len(uniq), k=1)
    pairs = sims[iu]
    pairs = pairs[pairs < 0.9]  # drop near-duplicate pairs
    if pairs.size == 0:
        return 0.5
    mean_pair = float(np.mean(pairs))
    # observed range ~0.05-0.35 -> 0-1
    return float(np.clip((mean_pair - 0.05) / 0.30, 0.0, 1.0))


def compute_skill_capital(user_skills: List[str], soc_code: str) -> Dict[str, Any]:
    """
    Score how strong the candidate's skills are for a target occupation.
    Returns {score, breadth, specialization, complementarity, matched_count,
    high_demand_gaps}.
    """
    occ = get_occ_tools(soc_code).drop_duplicates(subset=["tool_norm"])
    idf, idf_max = _idf_map()
    user_words = {w for s in user_skills for w in _norm(s).split() if len(w) >= 3}

    total_value = matched_value = 0.0
    matched_idfs: List[float] = []
    matched_count = high_demand_gaps = 0

    for _, row in occ.iterrows():
        tn = str(row["tool_norm"])
        is_hot, in_demand = bool(row["is_hot"]), bool(row["in_demand"])
        val = idf.get(tn, 1.0) * _demand_mult(is_hot, in_demand)
        total_value += val
        if set(tn.split()) & user_words:
            matched_value += val
            matched_idfs.append(idf.get(tn, 0.0))
            matched_count += 1
        elif is_hot or in_demand:
            high_demand_gaps += 1

    breadth = (matched_value / total_value) if total_value else 0.0
    specialization = (float(np.mean(matched_idfs)) / idf_max) if matched_idfs else 0.0
    complementarity = _complementarity(user_skills)

    # Value-weighted breadth, curved up, then amplified by specialization and
    # complementarity — so a focused specialist isn't punished for narrow raw
    # coverage of a huge tool list.
    base = min(1.0, breadth * _K_BREADTH)
    score = float(np.clip(base * (1.0 + _A_SPEC * specialization + _A_COMP * complementarity), 0.0, 1.0))

    return {
        "score": round(score, 3),
        "breadth": round(breadth, 3),
        "specialization": round(specialization, 3),
        "complementarity": round(complementarity, 3),
        "matched_count": matched_count,
        "high_demand_gaps": high_demand_gaps,
    }
