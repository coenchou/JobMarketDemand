"""Match extracted skills to ONET Software Skills data and score SOC candidates."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "onet"

# Generic words that appear in many tool names and shouldn't drive matching alone.
_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "not", "all", "new", "use", "via",
    "data", "system", "systems", "software", "management", "information",
    "analysis", "tools", "tool", "service", "services", "platform",
    "application", "applications", "program", "programs", "technology",
    "technologies", "solution", "solutions", "cloud", "web", "enterprise",
    "based", "type", "level", "general", "advanced", "basic", "open",
    "source", "free", "online", "digital", "global", "smart", "power",
}


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text)).lower().strip()


def _significant_words(norm_text: str, min_len: int = 3) -> List[str]:
    """Return words that are long enough and not generic stopwords."""
    return [w for w in norm_text.split() if len(w) >= min_len and w not in _STOPWORDS]


@lru_cache(maxsize=1)
def _load_sw() -> pd.DataFrame:
    """Load and normalise Software Skills.xlsx once per process."""
    path = DATA_DIR / "Software Skills.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code_full",
        "Title": "occ_title",
        "Workplace Example": "tool_name",
        "Element Name": "element_name",
        "Hot Technology": "is_hot",
        "In Demand": "in_demand",
    })
    df["soc_code"] = df["soc_code_full"].str[:7]
    df["tool_norm"] = df["tool_name"].apply(_norm)
    df["is_hot"] = df["is_hot"].astype(str).str.strip().eq("Y")
    df["in_demand"] = df["in_demand"].astype(str).str.strip().eq("Y")
    return df[["soc_code", "occ_title", "tool_name", "tool_norm",
               "element_name", "is_hot", "in_demand"]].copy()


@lru_cache(maxsize=1)
def _build_word_index() -> Dict[str, Set[str]]:
    """significant_word → set of tool_norm values that contain that word."""
    sw = _load_sw()
    index: Dict[str, Set[str]] = {}
    for tn in sw["tool_norm"].unique():
        for word in _significant_words(tn):
            index.setdefault(word, set()).add(tn)
    return index


@lru_cache(maxsize=1)
def _all_tool_norms() -> Set[str]:
    return set(_load_sw()["tool_norm"].unique())


def _match_skill_to_tool_norms(skill: str) -> Tuple[Set[str], str]:
    """
    Return the set of tool_norm values that match this skill string,
    plus a match_type label ("exact" | "substring" | "word" | "none").

    Matching rules (in priority order):
    1. Exact normalised match — highest confidence
    2. Skill norm is a contiguous substring of a tool norm (handles "Power BI" → "microsoft power bi")
    3. For multi-word skills: ALL significant words must appear in the same tool norm (intersection)
    4. For single-word skills: that word appears in the tool norm index
    """
    idx = _build_word_index()
    skill_norm = _norm(skill)

    # 1. Exact match
    if skill_norm in _all_tool_norms():
        return {skill_norm}, "exact"

    # 2. Substring match: skill_norm is a phrase inside tool_norm
    substring_hits: Set[str] = {tn for tn in _all_tool_norms() if skill_norm in tn}
    if substring_hits:
        return substring_hits, "substring"

    # 3 & 4. Word-based matching using only significant words
    sig_words = _significant_words(skill_norm)
    if not sig_words:
        return set(), "none"

    if len(sig_words) >= 2:
        # Multi-word: ALL significant words must appear (intersection)
        candidate: Optional[Set[str]] = None
        for word in sig_words:
            hits = idx.get(word, set())
            candidate = hits.copy() if candidate is None else candidate & hits
        if candidate:
            return candidate, "word"
        # Fallback: use only the longest significant word
        longest = max(sig_words, key=len)
        hits = idx.get(longest, set())
        return (hits, "word") if hits else (set(), "none")
    else:
        # Single significant word
        hits = idx.get(sig_words[0], set())
        return (hits, "word") if hits else (set(), "none")


@lru_cache(maxsize=1)
def _token_index() -> Dict[str, Set[str]]:
    """Every token (not just significant ones) → tool_norm values containing it."""
    index: Dict[str, Set[str]] = {}
    for tn in _all_tool_norms():
        for token in tn.split():
            index.setdefault(token, set()).add(tn)
    return index


@lru_cache(maxsize=1)
def _generic_words() -> Set[str]:
    """
    Words that describe a category of software rather than name one: every word
    O*NET uses in its Element Names ("design", "planning", "testing"), plus the
    usual filler. A skill built only from these identifies no specific tool.
    """
    sw = _load_sw()
    words = set(_STOPWORDS)
    for element in sw["element_name"].unique():
        words.update(_significant_words(_norm(element)))
    return words


def match_skill_to_tools_strict(skill: str) -> Set[str]:
    """
    High-precision counterpart to _match_skill_to_tool_norms: the tools this
    skill actually names, for deciding whether to *credit* a candidate with one.

    Ranking wants recall — a near-miss still says something about which
    occupation fits. Crediting wants precision, because a false positive invents
    a strength the candidate never claimed. So every token of the skill must
    appear in the tool's name ("Power BI" → "Microsoft Power BI", but "Apache
    Flink" no longer claims Cassandra, Hadoop, Hive and Pig), and a skill made
    purely of category words ("system design") claims nothing.
    """
    skill_norm = _norm(skill)
    if not skill_norm:
        return set()
    if skill_norm in _all_tool_norms():
        return {skill_norm}

    tokens = set(skill_norm.split())
    if tokens <= _generic_words():
        return set()

    index = _token_index()
    hits: Optional[Set[str]] = None
    for token in tokens:
        matched = index.get(token, set())
        hits = set(matched) if hits is None else hits & matched
        if not hits:
            return set()
    return hits or set()


def match_skills_to_onet(skills: List[str]) -> List[Dict]:
    """
    For each extracted skill, find the best representative ONET tool entry.
    Returns a list of dicts (one per input skill) with match metadata.
    """
    sw = _load_sw()
    results = []

    for skill in skills:
        matched_norms, match_type = _match_skill_to_tool_norms(skill)

        if not matched_norms:
            results.append({
                "original": skill,
                "matched_tool": None,
                "element_name": None,
                "is_hot": False,
                "in_demand": False,
                "match_type": "none",
            })
            continue

        subset = sw[sw["tool_norm"].isin(matched_norms)]
        # Pick the best representative row: prefer hot, then in-demand, then shortest name
        best = (
            subset
            .assign(_priority=subset["is_hot"].astype(int) * 2 + subset["in_demand"].astype(int))
            .sort_values(["_priority", "tool_name"], ascending=[False, True])
            .iloc[0]
        )
        results.append({
            "original": skill,
            "matched_tool": str(best["tool_name"]),
            "element_name": str(best["element_name"]),
            "is_hot": bool(subset["is_hot"].any()),
            "in_demand": bool(subset["in_demand"].any()),
            "match_type": match_type,
        })

    return results


def score_soc_candidates(skills: List[str], top_n: int = 10) -> List[Dict]:
    """
    Score SOC codes by how many of the user's skills appear in each occupation's
    software skill requirements (weighted by hot/in-demand status).
    Returns a ranked list of dicts.
    """
    sw = _load_sw()

    soc_scores: Dict[str, float] = {}
    soc_matched: Dict[str, Set[str]] = {}

    for skill in skills:
        matched_norms, _ = _match_skill_to_tool_norms(skill)
        if not matched_norms:
            continue

        rows = sw[sw["tool_norm"].isin(matched_norms)]
        for soc, group in rows.groupby("soc_code"):
            weight = 2.0 if group["is_hot"].any() else (1.5 if group["in_demand"].any() else 1.0)
            # Count each skill only once per occupation
            if skill not in soc_matched.get(soc, set()):
                soc_scores[soc] = soc_scores.get(soc, 0.0) + weight
                soc_matched.setdefault(soc, set()).add(skill)

    soc_total = sw.groupby("soc_code")["tool_name"].nunique()

    ranked = []
    for soc, raw in soc_scores.items():
        total = int(soc_total.get(soc, 1))
        matched_count = len(soc_matched[soc])
        # Score = raw / total^0.3
        # Using 0.3 (not 0.5) means larger occupations are penalised less,
        # so an occupation with 18 matched skills from 250 tools beats one with
        # 7 matched skills from 25 tools — absolute match count drives ranking.
        score = round(float(raw) / max(1.0, total ** 0.3), 4)
        coverage = round(matched_count / max(1, total), 4)
        ranked.append({
            "soc_code": soc,
            "raw_score": round(float(raw), 2),
            "match_score": score,
            "coverage": coverage,
            "matched_skills": sorted(soc_matched[soc]),
            "total_occ_tools": total,
        })

    # Filter: occupation must have >= 20 tools listed AND user must match >= 4 of them.
    ranked = [
        r for r in ranked
        if r["total_occ_tools"] >= 20 and len(r["matched_skills"]) >= 4
    ]
    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked[:top_n]


def get_occ_tools(soc_code: str) -> pd.DataFrame:
    """Return all Software Skills rows for a given SOC code."""
    sw = _load_sw()
    return sw[sw["soc_code"] == soc_code].copy()


@lru_cache(maxsize=1)
def _tool_occ_counts() -> Dict[str, int]:
    """tool_norm → number of distinct occupations that list it."""
    sw = _load_sw()
    return sw.groupby("tool_norm")["soc_code"].nunique().to_dict()


def count_occupations_for_tool(tool_name: str) -> int:
    """How many O*NET occupations list this tool in their requirements."""
    counts = _tool_occ_counts()
    tn = _norm(tool_name)
    if tn in counts:
        return int(counts[tn])
    hits = [v for k, v in counts.items() if tn in k]
    return int(max(hits)) if hits else 0
