"""
Career options for someone still deciding.

A working professional wants one target scored precisely. A student wants the
opposite: the range of things their current direction opens up, and what each
one costs to reach. The deciding fact at sixteen is rarely the salary — it is
whether a path needs a bachelor's, a master's, or neither, because that is the
choice actually in front of them.

So pathways are drawn from the field the résumé points at, spread deliberately
across education levels rather than ranked purely by fit, and each carries what
the candidate already has for it, what they would need to build, and the
entry requirements straight from BLS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_EDUCATION_ORDER = [
    "No formal educational credential",
    "High school diploma or equivalent",
    "Some college, no degree",
    "Postsecondary nondegree award",
    "Associate's degree",
    "Bachelor's degree",
    "Master's degree",
    "Doctoral or professional degree",
]

_EDUCATION_SHORT = {
    "No formal educational credential": "no credential",
    "High school diploma or equivalent": "high school",
    "Some college, no degree": "some college",
    "Postsecondary nondegree award": "certificate",
    "Associate's degree": "associate's",
    "Bachelor's degree": "bachelor's",
    "Master's degree": "master's",
    "Doctoral or professional degree": "doctorate",
}

MAX_PATHS = 6


def _education_rank(label: Optional[str]) -> int:
    try:
        return _EDUCATION_ORDER.index(label or "")
    except ValueError:
        return len(_EDUCATION_ORDER)


def education_short(label: Optional[str]) -> str:
    return _EDUCATION_SHORT.get(label or "", "varies")


def _is_residual(title: str) -> bool:
    return title.strip().lower().endswith("all other")


def build_pathways(
    candidates: List[Dict],
    skills: List[str],
    *,
    field: Optional[Dict] = None,
    max_paths: int = MAX_PATHS,
) -> List[Dict[str, Any]]:
    """
    Turn shortlisted occupations into career options a student can choose
    between, spread across education levels so the list is not six variations
    of the same requirement.
    """
    from src.gap_engine import compute_skill_gaps
    from src.labor_market import get_market, outlook_label
    from src.naming import pretty_skill
    from src.skill_matcher import match_skill_to_tools_strict
    from src.skill_score import compute_skill_capital
    from src.target_role import occupation_frame

    pool = [c for c in candidates if not _is_residual(c.get("title", ""))]
    if not pool:
        return []

    scored: List[Dict[str, Any]] = []
    for cand in pool[:12]:
        soc = cand["soc_code"]
        title = cand.get("title", soc)
        tools = occupation_frame(soc)
        if tools.empty:
            continue

        cap = compute_skill_capital(skills, tools=tools, title=title)
        gaps = compute_skill_gaps(skills, soc, tools=tools, title=title)
        market = get_market(soc) or {}

        # Name the skills the student actually listed rather than the O*NET
        # entries they matched: "Python" is what they wrote, "CAST SQL Builder"
        # is an implementation detail of the matcher.
        occupation_tools = set(tools["tool_norm"].astype(str))
        have = [
            skill for skill in skills
            if match_skill_to_tools_strict(skill) & occupation_tools
        ][:6]
        build = [pretty_skill(g["skill"]) for g in gaps.get("gaps", [])][:4]

        scored.append({
            "soc_code": soc,
            "title": title,
            "description": (cand.get("description") or "").strip(),
            "fit": round(float(cand.get("blended_score") or cand.get("match_score") or 0.0), 3),
            "skill_strength": round(cap.get("score", 0.0) * 100),
            "median_wage": market.get("median_wage"),
            "growth_pct": market.get("growth_pct"),
            "openings_k": market.get("openings_k"),
            "outlook": outlook_label(market.get("growth_pct")),
            "typical_education": market.get("typical_education"),
            "education_short": education_short(market.get("typical_education")),
            "education_rank": _education_rank(market.get("typical_education")),
            "typical_experience": market.get("typical_experience"),
            "have": have,
            "build": build,
        })

    if not scored:
        return []

    scored.sort(key=lambda p: -p["fit"])

    # One per education level first, so the list shows a ladder rather than six
    # roles that all demand the same degree; best remaining fits fill the rest.
    chosen: List[Dict[str, Any]] = []
    seen_levels = set()
    for path in scored:
        if path["education_rank"] not in seen_levels:
            seen_levels.add(path["education_rank"])
            chosen.append(path)
    for path in scored:
        if len(chosen) >= max_paths:
            break
        if path not in chosen:
            chosen.append(path)

    chosen = chosen[:max_paths]
    chosen.sort(key=lambda p: (p["education_rank"], -p["fit"]))
    return chosen


def paths_for_interest(
    code: str, max_paths: int = 4, min_openings: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Real occupations inside one family, for a student who has interests rather
    than a toolchain.

    Ranked by annual openings, because a sixteen-year-old exploring a field is
    better served by the jobs that actually exist in volume than by the closest
    match to a resume that has not been written yet. Spread across education
    levels so the list shows what opens at each rung.
    """
    from src.labor_market import _load_market, outlook_label
    from src.pipeline import _load_occ_data

    occ = _load_occ_data()
    titles = dict(zip(occ["soc_code"], occ["title"]))
    descriptions = dict(zip(occ["soc_code"], occ["description"]))

    rows = []
    for soc, m in _load_market().items():
        if not soc.startswith(code):
            continue
        title = titles.get(soc)
        if not title or _is_residual(title):
            continue
        if (m.get("openings_k") or 0) < min_openings:
            continue
        rows.append({
            "soc_code": soc,
            "title": title,
            "description": (descriptions.get(soc) or "").strip(),
            "median_wage": m.get("median_wage"),
            "growth_pct": m.get("growth_pct"),
            "openings_k": m.get("openings_k"),
            "outlook": outlook_label(m.get("growth_pct")),
            "typical_education": m.get("typical_education"),
            "education_short": education_short(m.get("typical_education")),
            "education_rank": _education_rank(m.get("typical_education")),
            "typical_experience": m.get("typical_experience"),
        })

    if not rows:
        return []

    rows.sort(key=lambda r: -(r["openings_k"] or 0))
    chosen, seen = [], set()
    for r in rows:
        if r["education_rank"] not in seen:
            seen.add(r["education_rank"])
            chosen.append(r)
    for r in rows:
        if len(chosen) >= max_paths:
            break
        if r not in chosen:
            chosen.append(r)
    chosen = chosen[:max_paths]
    chosen.sort(key=lambda r: (r["education_rank"], -(r["openings_k"] or 0)))
    return chosen


def explore_options(interests: List[Dict[str, Any]], max_fields: int = 3) -> List[Dict[str, Any]]:
    """Career options grouped by the interest that suggested them."""
    out = []
    for interest in interests[:max_fields]:
        paths = paths_for_interest(interest["code"])
        if paths:
            out.append({
                "code": interest["code"],
                "name": interest["name"],
                "label": interest.get("label", ""),
                "evidence": interest.get("evidence", [])[:2],
                "options": paths,
            })
    return out
