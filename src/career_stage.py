"""
Weighting the score for where someone actually is in their career.

A fixed blend of skill, experience and education flatters nobody. A new
graduate is scored against experience they could not possibly have yet, so the
strongest thing on their resume — the degree they just earned — gets swamped by
a near-zero on a third of the score. A principal engineer gets the opposite
problem: a decade of shipped work is compressed into one term while their
undergraduate degree still carries a quarter of the verdict, long after any
employer stopped caring what it says.

Labour economics has a name for that second effect. A credential is a *signal*,
valuable exactly while an employer has nothing better to go on, and it decays as
work history accumulates and the employer can observe output directly (employer
learning — Altonji & Pierret). So the weights move with career stage:

    stage    skill  experience  education  track record
    entry     0.45     0.15        0.35        0.05
    early     0.45     0.28        0.17        0.10
    mid       0.42     0.33        0.10        0.15
    senior    0.40     0.35        0.05        0.20

Track record is the fourth term the fixed blend had no room for: not how long
someone worked, but what the work shows — scope, leadership, quantified impact,
outside recognition. It is nearly weightless at entry (where there is rarely
anything to measure) and it is what separates two senior candidates with the
same toolchain.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_ENTRY_MAX = 1
_EARLY_MAX = 4
_MID_MAX = 9

_WEIGHTS: Dict[str, Dict[str, float]] = {
    "entry":  {"skill": 0.45, "experience": 0.15, "education": 0.35, "track_record": 0.05},
    "early":  {"skill": 0.45, "experience": 0.28, "education": 0.17, "track_record": 0.10},
    "mid":    {"skill": 0.42, "experience": 0.33, "education": 0.10, "track_record": 0.15},
    "senior": {"skill": 0.40, "experience": 0.35, "education": 0.05, "track_record": 0.20},
}

_STAGE_LABEL = {
    "entry": "early-career",
    "early": "a few years in",
    "mid": "mid-career",
    "senior": "senior",
}


def career_stage(years: Optional[int]) -> str:
    """Which stage the candidate is at, from years of professional experience."""
    if years is None or years <= _ENTRY_MAX:
        return "entry"
    if years <= _EARLY_MAX:
        return "early"
    if years <= _MID_MAX:
        return "mid"
    return "senior"


def stage_weights(stage: str) -> Dict[str, float]:
    return dict(_WEIGHTS.get(stage, _WEIGHTS["early"]))


def stage_label(stage: str) -> str:
    return _STAGE_LABEL.get(stage, stage)


def stage_explanation(stage: str) -> str:
    """One line on why this profile is weighted the way it is."""
    return {
        "entry": "Weighted for an early-career profile: your degree counts for "
                 "more and the experience you have not had yet counts for less.",
        "early": "Weighted for a few years in: experience is starting to carry "
                 "more than the degree.",
        "mid": "Weighted for a mid-career profile: what you have built now "
               "matters more than where you studied.",
        "senior": "Weighted for a senior profile: your track record carries the "
                  "verdict and the degree barely registers.",
    }.get(stage, "")


_SIGNALS: List[Dict[str, Any]] = [
    {
        "key": "leadership",
        "weight": 0.30,
        "label": "led people or set direction",
        "hint": "If you ran a team, mentored anyone or owned a decision, say so — nothing on the resume shows it.",
        "patterns": (
            r"\bled\b", r"\blead(ing)?\b", r"\bmanag(ed|ing)\b", r"\bmentor",
            r"\bteam of\b", r"\bdirector\b", r"\bhead of\b", r"\bprincipal\b",
            r"\bstaff engineer\b", r"\bsupervis", r"\bcoach", r"\bhired\b",
            r"\bset (the )?technical direction\b", r"\bowner?ship of\b",
        ),
    },
    {
        "key": "impact",
        "weight": 0.25,
        "label": "quantified results",
        "hint": "No outcome has a number attached. Quantify one result — a percentage, dollar figure or time saved.",
        "patterns": (
            r"\d+\s?%", r"\$\s?\d", r"\b\d+x\b", r"\breduc(ed|ing)\b",
            r"\bincreas(ed|ing)\b", r"\bimprov(ed|ing)\b", r"\bsav(ed|ing)\b",
            r"\bgrew\b", r"\braised\b",
        ),
    },
    {
        "key": "scale",
        "weight": 0.20,
        "label": "worked at scale",
        "hint": "Nothing conveys scope. If your work touched real users, records or traffic, say how much.",
        "patterns": (
            r"\b\d[\d,.]*\s?[mkb]\b", r"\bmillion\b", r"\bbillion\b",
            r"\busers\b", r"\brequests\b", r"\brecords\b", r"\btraffic\b",
            r"\bproduction\b", r"\bcustomers\b",
        ),
    },
    {
        "key": "recognition",
        "weight": 0.15,
        "label": "recognised outside the job",
        "hint": "Certifications, talks, publications and open-source work all count — list one if you have it.",
        "patterns": (
            r"\bpublish", r"\bpatent", r"\bspeaker\b", r"\bspoke at\b",
            r"\bconference\b", r"\baward", r"\bopen[- ]source\b",
            r"\bmaintainer\b", r"\bcertifi", r"\bhackathon\b", r"\bpaper\b",
        ),
    },
    {
        "key": "delivery",
        "weight": 0.10,
        "label": "shipped end-to-end work",
        "hint": "Nothing reads as built-and-launched. Describe one thing you carried from start to shipped.",
        "patterns": (
            r"\bbuilt\b", r"\bshipped\b", r"\bdeployed\b", r"\blaunched\b",
            r"\barchitected\b", r"\bdesigned and\b", r"\bend[- ]to[- ]end\b",
            r"\bproject", r"\bdelivered\b",
        ),
    },
]


_FULL_EVIDENCE = 6.0
_PRIOR = 0.5

_HEADER = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}|\b(19|20)\d{2}\s*[-–]")


def score_track_record(
    highlights: List[str],
    experience_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Score what the resume demonstrates, 0-1, from the achievements the parser
    pulled out plus the experience bullets.

    Deliberately about evidence, not duration — years already count under
    experience fit, and counting them twice would punish a career changer twice
    over for the same gap.

    Shrunk toward an average candidate when there is little to read, for the
    same reason skill strength is: a PDF that parses into bare job headers tells
    us nothing about what the person achieved, and scoring that as zero would
    punish them for their resume's formatting.
    """
    lines = [str(l) for l in list(highlights or []) + list(experience_lines or [])]
    text = " ".join(lines).lower()
    achievements = [l for l in lines if len(l) >= 40 and not _HEADER.search(l)]

    if not text.strip():
        return {"score": _PRIOR, "raw": 0.0, "evidence": 0.0,
                "signals": [], "missing": [s["label"] for s in _SIGNALS],
                "checks": [
                    {"key": s["key"], "label": s["label"],
                     "found": False, "hint": s["hint"]}
                    for s in _SIGNALS
                ]}

    found, missing, total = [], [], 0.0
    for signal in _SIGNALS:
        if any(re.search(p, text) for p in signal["patterns"]):
            total += signal["weight"]
            found.append(signal["label"])
        else:
            missing.append(signal["label"])

    raw = min(1.0, total)
    evidence = min(1.0, len(achievements) / _FULL_EVIDENCE)
    return {
        "score": round(evidence * raw + (1.0 - evidence) * _PRIOR, 3),
        "raw": round(raw, 3),
        "evidence": round(evidence, 3),
        "signals": found,
        "missing": missing,
        "checks": [
            {
                "key": sig["key"],
                "label": sig["label"],
                "found": sig["label"] in found,
                "hint": None if sig["label"] in found else sig["hint"],
            }
            for sig in _SIGNALS
        ],
    }
