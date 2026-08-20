"""
Which field a set of skills points toward.

Naming a single occupation for someone still in school does not work. Every
coverage-style measure favours occupations with short tool lists, so a high
schooler who knows Python, React and SQL gets matched to middle-school
technical education or a social science research assistant post — both of
which list a handful of generic tools that a teenager's toolkit happens to
cover a large share of. Meanwhile Software Developers, with a list of 250
tools, looks like a poor match on exactly the same evidence.

Aggregating to the occupational family removes that artefact. Each skill votes
for every occupation that lists it, weighted by how distinctive the tool is,
and the votes are summed per SOC major group. Individual occupation ranking is
noisy; the family signal is not. For the resume above, computing wins by more
than three to one over education.

A student gets a direction, not a job title.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

FIELDS: Dict[str, str] = {
    "11": "Management",
    "13": "Business and Finance",
    "15": "Computing and Mathematics",
    "17": "Engineering and Architecture",
    "19": "Science and Research",
    "21": "Community and Social Services",
    "23": "Law",
    "25": "Education",
    "27": "Arts, Design and Media",
    "29": "Healthcare",
    "31": "Healthcare Support",
    "33": "Protective Services",
    "35": "Food and Hospitality",
    "37": "Facilities and Grounds",
    "39": "Personal Care and Services",
    "41": "Sales",
    "43": "Office and Administration",
    "45": "Agriculture and Forestry",
    "47": "Construction and Trades",
    "49": "Installation and Repair",
    "51": "Production and Manufacturing",
    "53": "Transport and Logistics",
}


def field_name(code: str) -> str:
    return FIELDS.get(code, "General")


def infer_field(skills: List[str], top_n: int = 3) -> Optional[Dict[str, Any]]:
    """
    Rank occupational families by how much distinctive skill evidence points at
    each. Returns {code, name, score, share, alternatives} or None when the
    skills carry no occupational signal at all.
    """
    from src.skill_matcher import _load_sw, match_skill_to_tools_strict
    from src.skill_score import _idf_map

    if not skills:
        return None

    sw = _load_sw()
    idf, _, idf_default = _idf_map()
    by_tool = defaultdict(list)
    for tool_norm, soc in zip(sw["tool_norm"], sw["soc_code"]):
        by_tool[str(tool_norm)].append(str(soc))

    votes: Dict[str, float] = defaultdict(float)
    for skill in skills:
        for tool_norm in match_skill_to_tools_strict(skill):
            weight = idf.get(tool_norm, idf_default)
            for soc in set(by_tool.get(tool_norm, ())):
                votes[soc[:2]] += weight

    if not votes:
        return None

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    total = sum(votes.values()) or 1.0
    code, score = ranked[0]
    return {
        "code": code,
        "name": field_name(code),
        "score": round(score, 1),
        "share": round(score / total, 3),
        "alternatives": [
            {"code": c, "name": field_name(c), "share": round(v / total, 3)}
            for c, v in ranked[1:top_n]
        ],
    }


def roles_in_field(candidates: List[Dict], code: str) -> List[Dict]:
    """The shortlisted occupations that belong to one family, order preserved."""
    return [c for c in candidates if str(c.get("soc_code", "")).startswith(code)]
