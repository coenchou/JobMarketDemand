"""
Recognising an applicant who is still in school, and judging them on the right
terms.

The occupational model assumes everyone is entering the labour market. A high
school student is not, and forcing one through it produces nonsense: no degree
means education fit fails every bar, club and internship dates read as
professional experience, and the tool matcher hands back whichever occupation
happens to share a few entries with a teenager's toolkit. A real submission
came back "Likely hirable for Career/Technical Education Teachers, Middle
School".

What a student actually wants to know is how their profile compares with other
people applying from the same starting line, and what would strengthen it. So
students are detected before scoring and judged on the dimensions that separate
one application from another rather than on hiring bars they cannot clear yet.

The peer comparison is a rubric, not a measured percentile. There is no dataset
of applicant profiles behind it, and the report says so.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

HIGH_SCHOOL = "high_school"
COLLEGE = "college"

_HS_MARKERS = (
    r"\bhigh school\b", r"\bhigh-school\b", r"\bsecondary school\b",
    r"\bfreshman\b", r"\bsophomore\b", r"\bjunior year\b", r"\bsenior year\b",
    r"\bap [a-z]", r"\badvanced placement\b", r"\bib (?:diploma|higher level|hl|sl)\b",
    r"\bsat\s*:?\s*\d{3,4}\b", r"\bact\s*:?\s*\d{2}\b", r"\bpsat\b",
    r"\bhonor roll\b", r"\bvarsity\b",
    r"\bclass of 20\d\d\b",
)
_COLLEGE_MARKERS = (
    r"\bundergraduate\b", r"\bb\.?s\.? candidate\b", r"\bb\.?a\.? candidate\b",
    r"\bexpected (?:graduation|grad)\b", r"\bin progress\b", r"\bcurrently enrolled\b",
    r"\buniversity\b", r"\bcollege\b", r"\bmajor(?:ing)? in\b", r"\bminor in\b",
    r"\bdean's list\b", r"\bteaching assistant\b", r"\bcoursework\b",
)
_ENROLLED = (
    r"\bexpected (?:graduation|grad)\b", r"\bclass of 20\d\d\b", r"\bin progress\b",
    r"\bcurrently (?:enrolled|attending|a student)\b", r"\banticipated graduation\b",
    r"\bexpected \w+ 20\d\d\b",
)
_DEGREE_DONE = (
    r"\bgraduated\b", r"\bb\.?s\.?\b", r"\bb\.?a\.?\b", r"\bm\.?s\.?\b",
    r"\bph\.?d\.?\b", r"\bmaster'?s\b", r"\bbachelor'?s\b",
)


def _future_grad_year(text: str) -> Optional[int]:
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", text)]
    ahead = [y for y in years if y >= date.today().year]
    return min(ahead) if ahead else None


def detect_student(
    text: str,
    education_lines: Optional[List[str]] = None,
    education_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Decide whether this resume belongs to someone still in school.

    Returns {kind, grad_year, confidence, signals} for a student, or None.
    A completed degree beats every enrolment marker: someone with a bachelor's
    who lists "coursework" is a graduate, not an undergraduate.
    """
    blob = " ".join([text or ""] + list(education_lines or [])).lower()
    if not blob.strip():
        return None

    hs_hits = [p for p in _HS_MARKERS if re.search(p, blob)]
    college_hits = [p for p in _COLLEGE_MARKERS if re.search(p, blob)]
    enrolled = any(re.search(p, blob) for p in _ENROLLED)
    grad_year = _future_grad_year(blob)
    graduating_ahead = grad_year is not None and grad_year > date.today().year

    if education_level in {"Master's", "Doctorate"}:
        return None

    explicit_hs = re.search(r"\bhigh school\b|\bhigh-school\b|\bsecondary school\b", blob)
    if explicit_hs and not education_level:
        return {
            "kind": HIGH_SCHOOL,
            "grad_year": grad_year,
            "confidence": "high" if (enrolled or graduating_ahead) else "medium",
            "signals": hs_hits[:6],
        }

    if len(hs_hits) >= 3 and not education_level and not re.search(
            r"\bbachelor|\bmaster|\bph\.?d", blob):
        return {
            "kind": HIGH_SCHOOL,
            "grad_year": grad_year,
            "confidence": "medium",
            "signals": hs_hits[:6],
        }

    if college_hits and (enrolled or graduating_ahead) and education_level != "Doctorate":
        finished = re.search(r"\bgraduated\b", blob)
        if not finished or graduating_ahead:
            return {
                "kind": COLLEGE,
                "grad_year": grad_year,
                "confidence": "high" if graduating_ahead else "medium",
                "signals": college_hits[:6],
            }
    return None


_RIGOUR = (
    (r"\bap [a-z]", 0.30, "AP coursework"),
    (r"\bib (?:diploma|higher level|hl|sl)\b|\binternational baccalaureate\b", 0.30, "IB programme"),
    (r"\bhonors?\b|\bhonours?\b", 0.15, "honors courses"),
    (r"\bdual enrol|\bdual credit|\bcommunity college\b", 0.20, "college-level courses"),
    (r"\bgpa[:\s]*([0-9.]+)", 0.25, "GPA listed"),
    (r"\bsat\s*:?\s*\d{3,4}\b|\bact\s*:?\s*\d{2}\b", 0.15, "test scores"),
    (r"\bdean's list\b|\bmagna cum laude\b|\bsumma cum laude\b", 0.25, "academic distinction"),
    (r"\bresearch\b|\bthesis\b|\bpublished\b", 0.20, "research work"),
)

_RIGOUR_COLLEGE = (
    (r"\bdean's list\b|\bhonors?\b|\bhonours?\b|\bcum laude\b", 0.25, "academic distinction"),
    (r"\bresearch\b|\bthesis\b|\blab\b", 0.30, "research experience"),
    (r"\bteaching assistant\b|\bta for\b|\bcourse assistant\b|\btutor", 0.25, "taught a course"),
    (r"\bgpa[:\s]*([0-9.]+)", 0.20, "GPA listed"),
    (r"\bcoursework\b|\balgorithms\b|\bdata structures\b|\boperating systems\b|\bmachine learning\b", 0.20, "advanced coursework"),
    (r"\bgraduate[- ]level\b|\bmaster'?s\b", 0.20, "graduate-level work"),
)

_INITIATIVE = (
    (r"\bfounded\b|\bfounder\b|\bco-founded\b|\bstarted\b", 0.35, "started something"),
    (r"\bpresident\b|\bcaptain\b|\bled\b|\bleader\b|\bdirector\b", 0.30, "held a leadership role"),
    (r"\borganiz|\borganis|\bran a\b|\bhosted\b", 0.20, "organised events"),
    (r"\btaught\b|\btutor|\bmentor", 0.20, "taught or mentored"),
    (r"\bvolunteer|\bcommunity\b", 0.15, "community work"),
)

_RECOGNITION = (
    (r"\bfirst place\b|\b1st place\b|\bwon\b|\bwinner\b|\bchampion\b", 0.35, "won a competition"),
    (r"\bsecond place\b|\b2nd place\b|\bthird place\b|\b3rd place\b|\bfinalist\b|\bsemifinalist\b", 0.25, "placed in a competition"),
    (r"\baward|\bscholarship\b|\bprize\b|\bmedal\b|\bscholar\b", 0.25, "awards or scholarships"),
    (r"\busaco\b|\busamo\b|\baime\b|\bolympiad\b|\bscience fair\b|\bhackathon\b|\bdeca\b|\bfbla\b|\brobotics\b", 0.25, "competed at a recognised level"),
    (r"\bpublished\b|\bpublication\b|\bpatent\b|\bconference\b|\bpresented at\b"
     r"|\bco[- ]?authored\b|\bworkshop paper\b|\bpaper\b", 0.25, "presented or published"),
)

_OUTPUT = (
    (r"\bintern(?:ship)?\b", 0.30, "held an internship"),
    (r"\bbuilt\b|\bcreated\b|\bdeveloped\b|\bdesigned\b|\blaunched\b|\bshipped\b", 0.25, "built something"),
    (r"\b\d[\d,.]*\s?(?:\+|k\b|users\b|members\b|people\b|students\b|downloads\b)", 0.25, "reached real users"),
    (r"\bgithub\b|\bportfolio\b|\bopen[- ]source\b|\bdemo\b", 0.20, "work is public"),
    (r"\bfreelance\b|\bpaid\b|\bclient\b|\bcustomer", 0.20, "did paid work"),
)

HEADLINE_WEIGHTS: Dict[str, float] = {
    "skill": 0.28,
    "academics": 0.22,
    "initiative": 0.20,
    "recognition": 0.15,
    "output": 0.15,
}

RUBRIC_WEIGHT = 1.0 - HEADLINE_WEIGHTS["skill"]


def _dimensions(kind: str) -> Tuple[Tuple[str, str, tuple, float], ...]:
    rigour = _RIGOUR_COLLEGE if kind == COLLEGE else _RIGOUR
    w = HEADLINE_WEIGHTS
    return (
        ("academics", "Academic rigour", rigour, w["academics"]),
        ("initiative", "Initiative and leadership", _INITIATIVE, w["initiative"]),
        ("recognition", "Recognition", _RECOGNITION, w["recognition"]),
        ("output", "Things actually made", _OUTPUT, w["output"]),
    )


def score_application(
    text: str,
    highlights: Optional[List[str]] = None,
    kind: str = HIGH_SCHOOL,
) -> Dict[str, Any]:
    """
    Score a student profile on the dimensions that separate one application
    from another, each 0-1 with the specific evidence found and concrete advice
    where it is missing. Academic rigour means different things at the two
    stages: AP and IB for a high schooler, research and teaching for an
    undergraduate.
    """
    blob = " ".join([text or ""] + list(highlights or [])).lower()
    dims = []
    for key, label, rules, weight in _dimensions(kind):
        found, score = [], 0.0
        for pattern, value, description in rules:
            if re.search(pattern, blob):
                score += value
                found.append(description)
        dims.append({
            "key": key,
            "label": label,
            "weight": weight,
            "score": round(min(1.0, score), 3),
            "found": found,
            "missing": [d for _, _, d in rules if d not in found][:2],
        })
    composite = sum(d["score"] * d["weight"] for d in dims) / RUBRIC_WEIGHT
    return {"score": round(min(1.0, composite), 3), "dimensions": dims}


PEER_LABEL = {
    HIGH_SCHOOL: "other high school applicants",
    COLLEGE: "other undergraduates",
}


def competitiveness_level(score: float) -> str:
    if score >= 80:
        return "Very competitive"
    if score >= 60:
        return "Competitive"
    if score >= 40:
        return "Developing"
    if score >= 20:
        return "Early"
    return "Just getting started"


def stage_note(kind: str) -> str:
    peers = PEER_LABEL.get(kind, "other applicants")
    return (
        f"You are still in school, so this is scored as an application against "
        f"{peers} — not as a job candidate against hiring bars you have not "
        f"reached yet. It is a rubric over what your resume shows, not a "
        f"measured percentile against real applicants."
    )
