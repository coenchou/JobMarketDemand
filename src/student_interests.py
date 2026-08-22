"""
Reading a high schooler's resume for what it actually contains.

The occupational matcher runs on O*NET's software lists, which is the right
signal for a working professional and the wrong one for a sixteen-year-old. A
pre-med student who volunteered 120 hours at a hospital, competes in HOSA and
takes AP Biology matches zero software tools, so the matcher returned no
candidates at all and the report told her that her resume "does not point at a
field yet".

What she has is experience, so that is what gets read. Each signal below is a
plain pattern over the resume text tied to an occupational family, and it
carries the phrase that triggered it so the report can say why a suggestion
appeared rather than asking to be trusted.

Semantic similarity is deliberately not the primary route here. A student's
resume is saturated with school vocabulary, so embedding the whole thing
returns teaching and education-administration occupations for everyone.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# (family code, human evidence label, patterns)
_SIGNALS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("29", "health care", (
        r"\bhospital\b", r"\bclinic\b", r"\bpatient", r"\bnurs(e|ing)\b",
        r"\bmedical\b", r"\bmedicine\b", r"\bhosa\b", r"\bpre-?med\b",
        r"\banatomy\b", r"\bphysiolog", r"\bems\b", r"\bemt\b",
        r"\bfirst aid\b", r"\bcpr\b", r"\bpharmac", r"\bdental\b",
        r"\bveterinar", r"\bcaregiv", r"\bhealth\b",
    )),
    ("15", "computing", (
        r"\bcod(e|ing)\b", r"\bprogramming\b", r"\bpython\b", r"\bjava\b",
        r"\bjavascript\b", r"\bapp\b", r"\bwebsite\b", r"\bweb develop",
        r"\bsoftware\b", r"\bcomputer science\b", r"\bcs\b", r"\busaco\b",
        r"\bhackathon\b", r"\bgithub\b", r"\bdatabase\b", r"\bcybersecurity\b",
        r"\bdata (science|analysis)\b", r"\balgorithm",
    )),
    ("17", "engineering and building", (
        r"\brobotics?\b", r"\bengineer", r"\bcad\b", r"\b3d print",
        r"\bdrivetrain\b", r"\bmechanic", r"\belectrical\b", r"\bcircuit",
        r"\bwelding\b", r"\bmachin(e|ing) shop\b", r"\bfrc\b", r"\bvex\b",
        r"\bprototyp", r"\bdesign team\b", r"\bconstruct",
    )),
    ("19", "science and research", (
        r"\bscience fair\b", r"\bresearch\b", r"\blab(oratory)?\b",
        r"\bap (biology|chemistry|physics|environmental)\b", r"\bexperiment",
        r"\bchemistry\b", r"\bbiology\b", r"\bphysics\b", r"\bastronom",
        r"\benvironmental\b", r"\bgeolog", r"\bmarine\b",
    )),
    ("13", "business and finance", (
        r"\bdeca\b", r"\bfbla\b", r"\bbusiness\b", r"\baccounting\b",
        r"\bfinance\b", r"\binvest", r"\beconomics\b", r"\bmarketing\b",
        r"\bentrepreneur", r"\bstartup\b", r"\bbudget",
    )),
    ("41", "sales and customer work", (
        r"\bcashier\b", r"\bretail\b", r"\bcustomer\b", r"\bbarista\b",
        r"\bserver\b", r"\bwaite?r", r"\bsales\b", r"\bregister\b",
        r"\bhost(ess)?\b", r"\bfood service\b",
    )),
    ("25", "teaching and mentoring", (
        r"\btutor", r"\bteach(ing)?\b", r"\bmentor", r"\bcoach(ed|ing)?\b",
        r"\bteaching assistant\b", r"\bcamp counselor\b", r"\binstruct",
    )),
    ("27", "art, design and media", (
        r"\bdesign(er|ing)?\b", r"\bgraphic", r"\bphotograph", r"\bvideo\b",
        r"\bfilm\b", r"\byearbook\b", r"\bnewspaper\b", r"\bjournalism\b",
        r"\bmusic\b", r"\bband\b", r"\btheat(er|re)\b", r"\bart club\b",
        r"\bdrawing\b", r"\banimation\b", r"\bsocial media\b",
    )),
    ("23", "law and debate", (
        r"\bdebate\b", r"\bmock trial\b", r"\bmodel un\b", r"\bmun\b",
        r"\blaw\b", r"\blegal\b", r"\bmoot court\b", r"\bpolitical science\b",
        r"\bgovernment\b",
    )),
    ("21", "community and social work", (
        r"\bvolunteer", r"\bcommunity service\b", r"\bnon-?profit\b",
        r"\bcharity\b", r"\bfundrais", r"\boutreach\b", r"\bshelter\b",
        r"\bsocial work\b", r"\bkey club\b", r"\bnational honor society\b",
    )),
    ("11", "leading and organising", (
        r"\bpresident\b", r"\bcaptain\b", r"\bfounder\b", r"\bfounded\b",
        r"\bstudent (council|government)\b", r"\bled a\b", r"\borganiz(ed|ing)\b",
        r"\bmanag(ed|ing)\b", r"\bteam lead\b",
    )),
    ("31", "care and support work", (
        r"\bbabysit", r"\bchildcare\b", r"\belder", r"\bcaregiver\b",
        r"\bspecial (needs|education)\b", r"\bassisted living\b",
    )),
    ("49", "hands-on repair", (
        r"\bauto repair\b", r"\bengine\b", r"\bdiagnostics\b", r"\bcar\b",
        r"\brepair", r"\bmaintenance\b", r"\bhvac\b", r"\bplumb", r"\belectrician\b",
    )),
    ("45", "outdoors and animals", (
        r"\bfarm\b", r"\bagricultur", r"\bffa\b", r"\bgarden", r"\banimal",
        r"\bhorse", r"\bconservation\b", r"\bforestry\b",
    )),
    ("33", "public safety", (
        r"\bfire(fighter)?\b", r"\bpolice\b", r"\bsecurity\b", r"\blifeguard\b",
        r"\bsearch and rescue\b", r"\bcadet\b", r"\brotc\b",
    )),
)

# Weak on their own — nearly every student resume contains them, so they only
# reinforce a family that some stronger signal already named.
_WEAK = {"11", "25", "21"}


_CLAUSE = re.compile(r"[\n\r|;•·–—]|(?<=[a-z]),\s|\.\s")


def _evidence_phrase(text: str, match: re.Match) -> str:
    """
    A short quote showing where the interest came from. The quote is trimmed to
    the clause holding the match, so it reads as something the person wrote
    rather than a window slid across two unrelated resume lines.
    """
    left = max((m.end() for m in _CLAUSE.finditer(text, 0, match.start())), default=0)
    right = _CLAUSE.search(text, match.end())
    clause = text[left:right.start() if right else len(text)]
    clause = re.sub(r"\s+", " ", clause).strip(" -–—•·|,.:")

    if len(clause) <= 62:
        return clause
    # Still long: keep whole words either side of the match.
    at = clause.lower().find(match.group(0).lower().strip())
    at = at if at >= 0 else 0
    head = clause.rfind(" ", 0, max(0, at - 24))
    tail = clause.find(" ", at + len(match.group(0)) + 24)
    out = clause[head + 1 if head > 0 else 0:tail if tail > 0 else len(clause)]
    return out.strip(" -–—•·|,.:")


def infer_interests(text: str, top_n: int = 4) -> List[Dict[str, Any]]:
    """
    Occupational families the resume points at, strongest first, each with the
    phrases that put it there.
    """
    from src.field_direction import field_name

    if not text or not text.strip():
        return []

    lowered = text.lower()
    found: Dict[str, Dict[str, Any]] = {}

    for code, label, patterns in _SIGNALS:
        hits, quotes = 0, []
        for pattern in patterns:
            for m in re.finditer(pattern, lowered):
                hits += 1
                if len(quotes) < 3:
                    quotes.append(_evidence_phrase(text, m))
                break
        if hits:
            found[code] = {
                "code": code,
                "name": field_name(code),
                "label": label,
                "hits": hits,
                "evidence": quotes,
            }

    strong = {c for c in found if c not in _WEAK}
    ranked = sorted(
        found.values(),
        key=lambda f: (f["code"] not in _WEAK and bool(strong), f["hits"]),
        reverse=True,
    )
    if strong:
        ranked = [f for f in ranked if f["code"] in strong]

    total = sum(f["hits"] for f in ranked) or 1
    for f in ranked:
        f["share"] = round(f["hits"] / total, 3)

    # A single passing mention should not headline a career direction: one
    # summer behind a coffee counter is not a reason to lead with retail.
    strongest = ranked[0]["share"] if ranked else 0.0
    ranked = [f for f in ranked if f["share"] >= 0.15 or f["share"] == strongest]
    return ranked[:top_n]
