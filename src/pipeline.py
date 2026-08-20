"""Full resume analysis pipeline producing a structured JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser import parse_resume
from src.skill_matcher import score_soc_candidates
from src.gap_engine import compute_skill_gaps
from src.skill_score import compute_skill_capital
from src.career_stage import (
    career_stage,
    score_track_record,
    stage_explanation,
    stage_label,
    stage_weights,
)
from src.naming import pretty_skill
from src.simulator import simulate_additions
from src.target_role import occupation_frame, resolve_target
from src.nlp_skills import (
    generate_recommendations,
    score_ai_displacement,
    llm_refine_recommendations,
)
from src.recommendations import _fallback_summary
from src.labor_market import (
    get_market,
    education_fit,
    experience_fit,
    outlook_score,
    outlook_label,
    wage_percentile,
)
from src.semantic_matcher import semantic_score_candidates, blend_scores

DATA_DIR = ROOT / "data" / "raw" / "onet"


# ---------------------------------------------------------------------------
# Occupation metadata
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_occ_data() -> pd.DataFrame:
    path = DATA_DIR / "Occupation Data.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code_full",
        "Title": "title",
        "Description": "description",
    })
    df["soc_code"] = df["soc_code_full"].str[:7]
    return df[["soc_code", "title", "description"]].drop_duplicates(subset=["soc_code"])


def _enrich_candidates(candidates: List[Dict]) -> List[Dict]:
    occ = _load_occ_data()
    enriched = []
    for cand in candidates:
        rows = occ[occ["soc_code"] == cand["soc_code"]]
        if rows.empty:
            continue
        row = rows.iloc[0]
        enriched.append({
            **cand,
            "title": str(row["title"]),
            "description": str(row["description"]),
        })
    return enriched


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Fallback curves — used only when the occupation has no BLS row.
def _edu_score(level: Optional[str]) -> float:
    return {"Associate's": 0.4, "Bachelor's": 0.7, "Master's": 0.9, "Doctorate": 1.0}.get(
        level or "", 0.3
    )


def _exp_score(years: Optional[int]) -> float:
    if not years:
        return 0.25
    if years <= 1:
        return 0.35
    if years <= 3:
        return 0.55
    if years <= 6:
        return 0.75
    if years <= 10:
        return 0.90
    return 1.0


def _component(label: str, value: float, weight: float, detail: Optional[Dict] = None) -> Dict:
    c = {
        "label": label,
        "value": round(value, 3),
        "weight": weight,
        "points": round(weight * value * 100, 1),
    }
    if detail:
        c["detail"] = detail
    return c


def _fits(years: Optional[int], edu: Optional[str], market: Optional[Dict]) -> Tuple[float, float]:
    """(experience fit, education fit) against BLS requirements, else curves."""
    edu_fit = education_fit(edu, market.get("typical_education")) if market else None
    if edu_fit is None:
        edu_fit = _edu_score(edu)
    exp_fit = experience_fit(years, market.get("typical_experience")) if market else None
    if exp_fit is None:
        exp_fit = _exp_score(years)
    return exp_fit, edu_fit


def _hirability(
    skill: float, years: Optional[int], edu: Optional[str],
    market: Optional[Dict], skill_detail: Optional[Dict] = None,
    evidence: Optional[float] = None, track: Optional[Dict] = None,
) -> Dict:
    """
    Weighted blend of skill strength, experience fit, education fit and track
    record — with the weights set by career stage rather than fixed, so a new
    graduate is not scored mostly on experience they cannot have yet and a
    senior engineer is not still being graded on their degree. See
    career_stage.py.

    Experience and education are measured against what the occupation actually
    requires (BLS Employment Projections) when available, else fall back to the
    old curves. Returns {score, level, range, stage, components}.
    """
    exp_fit, edu_fit = _fits(years, edu, market)
    stage = career_stage(years)
    w = stage_weights(stage)
    track = track or {"score": 0.0, "signals": []}

    components = [
        _component("Skill strength", skill, w["skill"], skill_detail),
        _component("Experience fit", exp_fit, w["experience"]),
        _component("Education fit", edu_fit, w["education"]),
        _component("Track record", track.get("score", 0.0), w["track_record"],
                   {"signals": track.get("signals", [])} if track.get("signals") else None),
    ]
    score = round(sum(c["points"] for c in components), 1)
    return {
        "score": score,
        "level": _hirability_level(score),
        "range": _score_range(score, evidence, w["skill"]),
        "stage": stage,
        "stage_label": stage_label(stage),
        "stage_note": stage_explanation(stage),
        "components": components,
    }


def _score_range(
    score: float, evidence: Optional[float], skill_weight: float = 0.45
) -> Optional[List[int]]:
    """
    A plausible band around the headline score.

    The skill term is a shrunken estimate — the less a resume documents, the
    more of it is the prior rather than the candidate. That residual is real
    uncertainty, so the report shows it instead of a false point estimate.
    Width is the unexplained share of the skill weight, which now varies by
    career stage.
    """
    if evidence is None:
        return None
    half = (1.0 - evidence) * 0.5 * skill_weight * 100
    return [max(0, round(score - half)), min(100, round(score + half))]


def _hirability_level(score: float) -> str:
    if score >= 80:
        return "Very likely hirable"
    if score >= 60:
        return "Likely hirable"
    if score >= 40:
        return "Moderately hirable"
    if score >= 20:
        return "Developing"
    return "Early-stage"


def _verdict(hire_level: str, title: Optional[str], growth_pct: Optional[float]) -> str:
    """One-line plain-language headline of the result."""
    if not title:
        return hire_level
    s = f"{hire_level} for {title}"
    if growth_pct is not None:
        if growth_pct >= 4:
            s += " — a field growing faster than average"
        elif growth_pct >= 1:
            s += " — a field growing about as fast as average"
        elif growth_pct > -1:
            s += " — a field holding roughly steady"
        else:
            s += " — though the field is contracting"
    return s + "."


def _competitiveness(
    skill: float,
    years: Optional[int],
    hot_skill_count: int,
    total_matched: int,
    market: Optional[Dict],
    skill_detail: Optional[Dict] = None,
) -> Dict:
    """
    How well-positioned the candidate is: skill coverage + share of hot-tech
    skills + how favorable the occupation's market outlook is (BLS projected
    growth and openings) + experience fit. `skill` is a 0-1 component from the caller.
    """
    hot_ratio = hot_skill_count / max(1, total_matched)

    outlook = outlook_score(market)
    exp_fit = experience_fit(years, market.get("typical_experience")) if market else None
    if exp_fit is None:
        exp_fit = _exp_score(years)

    if outlook is not None:
        components = [
            _component("Skill strength", skill, 0.35, skill_detail),
            _component("In-demand skills", hot_ratio, 0.25),
            _component("Market outlook", outlook, 0.25),
            _component("Experience fit", exp_fit, 0.15),
        ]
    else:
        # no BLS outlook — reweight onto the other three
        components = [
            _component("Skill strength", skill, 0.45, skill_detail),
            _component("In-demand skills", hot_ratio, 0.30),
            _component("Experience fit", exp_fit, 0.25),
        ]
    score = round(sum(c["points"] for c in components), 1)

    if score >= 70:
        level, blurb = "Strong", "Your skill set aligns well with market demand and includes high-value technologies."
    elif score >= 45:
        level, blurb = "Competitive", "Solid foundation — adding a few in-demand skills would significantly boost your profile."
    elif score >= 25:
        level, blurb = "Developing", "Core skills are present; targeted gap-filling will be essential to compete."
    else:
        level, blurb = "Entry-level", "Focus on foundational skills; build a portfolio to compensate for limited experience."
    return {"score": score, "level": level, "explanation": blurb, "components": components}


# ---------------------------------------------------------------------------
# Ranking roles by how hirable the candidate actually is for each
# ---------------------------------------------------------------------------

# A role only gets to lead on hirability if the resume genuinely fits it.
# Hirability rewards a low entry bar, so without a gate the pick drifts to
# whichever occupation asks least: a senior software engineer scores 93 as a
# middle-school teacher, purely because O*NET lists Excel for teachers and BLS
# asks for a bachelor's. Tool overlap alone cannot tell those apart — every
# office job shares a toolchain — so the gate uses the blended fit score, which
# also carries how much the resume *reads* like that occupation.
_MIN_FIT_RATIO = 0.85
_N_ROLES = 5


def _rank_roles(
    candidates: List[Dict],
    skills: List[str],
    years: Optional[int],
    edu: Optional[str],
    track: Dict,
) -> List[Dict]:
    """
    Score the candidate against each of the top occupations and rank by
    hirability rather than by how closely their toolchain matches.

    Someone can match "Computer and Information Research Scientists" most
    closely on tools and still be markedly more hirable as a Data Scientist,
    because the education and experience bars differ. That is the number they
    care about, so it decides which analysis leads.

    The list comes back ordered by how closely the résumé fits each role, which
    is the order that reads naturally; `eligible` marks the ones a fit is close
    enough to be promoted on hirability alone, and is used to pick the default.
    """
    if not candidates:
        return []

    def fit(c: Dict) -> float:
        blended = c.get("blended_score")
        return float(blended) if blended is not None else float(c.get("match_score", 0.0))

    best_fit = max((fit(c) for c in candidates), default=0.0)
    ranked: List[Dict] = []

    for cand in candidates[:_N_ROLES]:
        soc = cand["soc_code"]
        cap = compute_skill_capital(
            skills, tools=occupation_frame(soc), title=cand.get("title"))
        market = get_market(soc)
        hire = _hirability(
            cap.get("score", 0.0), years, edu, market,
            evidence=cap.get("evidence"), track=track,
        )
        ranked.append({
            "soc_code": soc,
            "title": cand.get("title", soc),
            "hirability": hire["score"],
            "level": hire["level"],
            "skill_strength": round(cap.get("score", 0.0) * 100),
            "match_score": round(cand.get("match_score", 0.0), 3),
            "blended_score": cand.get("blended_score"),
            "fit": round(fit(cand), 3),
            # Only a role the resume genuinely fits may be promoted ahead of
            # the closest match on hirability alone.
            "eligible": fit(cand) >= _MIN_FIT_RATIO * best_fit,
        })

    ranked.sort(key=lambda r: r["fit"], reverse=True)
    return ranked


def _pick_default_role(ranked: List[Dict]) -> Optional[str]:
    """The role to lead with: most hirable among those the résumé really fits."""
    eligible = [r for r in ranked if r["eligible"]] or ranked
    return max(eligible, key=lambda r: r["hirability"])["soc_code"] if eligible else None


def _role_view(
    cand: Dict,
    *,
    skills: List[str],
    years: Optional[int],
    edu: Optional[str],
    track: Dict,
    hot_count: int,
    matched_total: int,
    highlights: List[str],
) -> Dict[str, Any]:
    """
    The whole analysis for one occupation, as a patch over the base report.

    Every shortlisted role gets one of these up front so switching between them
    is instant — re-running the pipeline on click would send the reader back
    through an eight-second wait to see numbers we had already computed.

    Deterministic only: no LLM call per role. The recommendations here are the
    dataset-derived ones, which already carry written rationales; the role the
    report opens on keeps its LLM-refined versions.
    """
    soc, title = cand["soc_code"], cand.get("title", cand["soc_code"])
    tools = occupation_frame(soc)

    gaps = compute_skill_gaps(skills, soc, tools=tools, title=title)
    cap = compute_skill_capital(skills, tools=tools, title=title)
    market = get_market(soc)
    detail = {
        "breadth": cap.get("breadth"),
        "specialization": cap.get("specialization"),
        "complementarity": cap.get("complementarity"),
        "evidence": cap.get("evidence"),
    }
    hire = _hirability(
        cap.get("score", 0.0), years, edu, market, detail,
        cap.get("evidence"), track)
    comp = _competitiveness(
        cap.get("score", 0.0), years, hot_count, matched_total, market, detail)

    recs = generate_recommendations(gaps.get("gaps", []), skills, top_n=5)
    recs = [{**r, "skill": pretty_skill(r["skill"])} for r in recs]

    exp_fit, edu_fit = _fits(years, edu, market)
    simulation = simulate_additions(
        skills, [r["skill"] for r in recs], tools=tools,
        exp_fit=exp_fit, edu_fit=edu_fit,
        weights=stage_weights(hire.get("stage", "early")),
        track=track.get("score", 0.0), baseline=cap,
    )

    exposure = score_ai_displacement(cand.get("description", ""), title)

    return {
        "verdict": _verdict(
            hire["level"], title, market.get("growth_pct") if market else None),
        "summary_text": _fallback_summary(title, recs, years),
        "score_breakdown": {
            "hirability": hire["components"],
            "competitiveness": comp.get("components", []),
        },
        "summary": {
            "hirability_score": hire["score"],
            "hirability_range": hire.get("range"),
            "hirability_level": hire["level"],
            "top_match_title": title,
            "top_match_soc": soc,
        },
        "competitiveness": comp,
        "labor_market": _market_snapshot(soc, title, market, years, edu),
        "top_job_matches": [{
            "soc_code": soc, "title": title,
            "description": cand.get("description", ""),
            "match_score": cand.get("match_score", 0.0),
            "blended_score": cand.get("blended_score"),
            "matched_skills": cand.get("matched_skills", []),
        }],
        "skill_gaps": {
            "coverage_of_top_match": gaps.get("coverage", 0.0),
            "strengths": [
                {**s, "skill": pretty_skill(s["skill"])}
                for s in gaps.get("strengths", [])
            ],
            "highlights": highlights,
            "gaps": [
                {**g, "skill": pretty_skill(g["skill"])}
                for g in gaps.get("gaps", [])[:10]
            ],
            "abstract_skills_required": gaps.get("abstract_skills_required", [])[:6],
        },
        "skill_strength": {
            "score": round(cap.get("score", 0.0) * 100),
            "matched": cap.get("matched_count", 0),
            "implied": cap.get("implied_count", 0),
            "ignored_basic": cap.get("ignored_basic", 0),
            "focus_skills": [r["skill"] for r in recs[:3]],
            "breadth": cap.get("breadth"),
            "specialization": cap.get("specialization"),
            "complementarity": cap.get("complementarity"),
        },
        "recommendations": recs,
        "simulation": simulation,
        "ai_displacement_exposure": exposure,
    }


def _market_snapshot(
    soc: str, title: str, market: Optional[Dict],
    years: Optional[int], edu: Optional[str],
) -> Optional[Dict]:
    """BLS figures for one occupation, or None when it has no row."""
    if not market:
        return None
    return {
        "soc_code": soc,
        "title": title,
        "median_wage": market.get("median_wage"),
        "wage_percentile": wage_percentile(market.get("median_wage")),
        "growth_pct": market.get("growth_pct"),
        "openings_k": market.get("openings_k"),
        "outlook": outlook_label(market.get("growth_pct")),
        "typical_education": market.get("typical_education"),
        "typical_experience": market.get("typical_experience"),
        "education_fit": education_fit(edu, market.get("typical_education")),
        "experience_fit": experience_fit(years, market.get("typical_experience")),
    }


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(
    resume_path: str,
    target_soc: Optional[str] = None,
    job_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse a resume against a target.

    The target defaults to the best-matching occupation, but the caller can
    pin one (`target_soc`, e.g. the user picked a different match) or supply a
    pasted posting (`job_description`), in which case the candidate is scored
    against the skills that posting actually asks for.
    """
    resume_path = Path(resume_path)
    parsed = parse_resume(str(resume_path))

    skills: List[str] = parsed["skills"]
    years: Optional[int] = parsed["years_experience"]
    edu: Optional[str] = parsed["education_level"]

    # SOC candidate scoring
    candidates = score_soc_candidates(skills, top_n=10)
    enriched = _enrich_candidates(candidates)

    # Semantic re-ranking: blend tool-based score with NLP similarity
    exp_lines: List[str] = parsed.get("experience_lines", [])
    semantic_scores = semantic_score_candidates(
        exp_lines, skills, [c["soc_code"] for c in enriched]
    )
    enriched = blend_scores(enriched, semantic_scores)

    # What the resume demonstrates, independent of any one role.
    track: Dict = score_track_record(parsed.get("highlights", []), exp_lines)

    # Rank the shortlisted occupations by hirability, not by tool overlap, and
    # lead with the best one — matching an occupation closely is not the same
    # as being hirable for it.
    role_options = _rank_roles(enriched, skills, years, edu, track)
    if role_options and not (target_soc or job_description):
        best = _pick_default_role(role_options)
        enriched = ([c for c in enriched if c["soc_code"] == best]
                    + [c for c in enriched if c["soc_code"] != best])

    # What we are scoring against: the best role, an override, or a posting.
    occ = _load_occ_data()
    titles = dict(zip(occ["soc_code"], occ["title"]))
    target = resolve_target(
        enriched,
        override_soc=target_soc,
        job_description=job_description,
        occ_titles=titles,
    )

    # When a posting names an occupation we didn't rank, surface it as the match.
    top = enriched[0] if enriched else None
    if target and target["source"] != "auto" and target.get("soc_code"):
        pinned = next((c for c in enriched if c["soc_code"] == target["soc_code"]), None)
        if pinned is None:
            pinned = _enrich_candidates([{
                "soc_code": target["soc_code"], "match_score": 0.0,
                "matched_skills": [], "blended_score": 0.0,
            }])
            pinned = pinned[0] if pinned else None
        top = pinned or top

    matched_skills: List[Dict] = parsed.get("matched_skills", [])
    matched_tools = [m for m in matched_skills if m.get("matched_tool")]
    hot_count = sum(1 for m in matched_tools if m.get("is_hot"))

    target_tools = target["tools"] if target else None
    target_title = target["title"] if target else (top["title"] if top else None)
    market_title = (target.get("occupation_title") if target else None) or (
        top["title"] if top else None)

    # Compute skill gaps first — its coverage fraction (matched/total occupation
    # tools) is a real 0-1 measure, far more discriminating for scoring than the
    # raw match_score, which saturates for any decent match.
    skill_gaps: Dict = compute_skill_gaps(
        skills, target["soc_code"] if target else None,
        tools=target_tools, title=market_title,
    ) if target else {}

    # Skill strength — value-weighted coverage + specialization + complementarity
    # (see skill_score.py), not naive fraction-of-toolset. A focused specialist's
    # deep, mutually-reinforcing skills score high even at low raw coverage.
    skill_cap: Dict = compute_skill_capital(
        skills, tools=target_tools, title=market_title) if target else {}
    skill_component = skill_cap.get("score", 0.0)
    skill_detail = {
        "breadth": skill_cap.get("breadth"),
        "specialization": skill_cap.get("specialization"),
        "complementarity": skill_cap.get("complementarity"),
        "evidence": skill_cap.get("evidence"),
    } if skill_cap else None

    # BLS labor-market data for the top-matched occupation grounds the scores
    market = get_market(top["soc_code"]) if top else None

    evidence = skill_cap.get("evidence") if skill_cap else None
    hire = _hirability(
        skill_component, years, edu, market, skill_detail, evidence, track)
    competitiveness = _competitiveness(
        skill_component, years, hot_count, len(matched_tools), market, skill_detail)

    # NLP recommendations for top candidate
    recommendations: List[Dict] = []
    ai_displacement: Optional[Dict] = None
    summary_text: str = ""

    if top:
        dataset_recs = generate_recommendations(
            skill_gaps.get("gaps", []),
            skills,
            top_n=8,
        )
        ai_displacement = score_ai_displacement(
            top.get("description", ""),
            top.get("title", ""),
        )
        # Hand the blunt dataset gap list to the LLM: prune what the candidate
        # obviously already has, add current skills the dataset misses, and
        # produce a concrete next-steps summary.
        refined = llm_refine_recommendations(
            target_title or top["title"], skills, years, exp_lines, dataset_recs,
        )
        summary_text = refined["summary"]
        # Display names from here on, so the recommendation, the focus list and
        # the simulated delta all refer to the same string.
        recommendations = [
            {**r, "skill": pretty_skill(r["skill"])} for r in refined["recommendations"]
        ]

        # Enrich the automation-exposure blurb with a concrete note on how AI is
        # actually used in this field (from the same LLM call — no extra request).
        note = refined.get("automation_note", "")
        if note and ai_displacement:
            ai_displacement["explanation"] = (
                ai_displacement.get("explanation", "").rstrip() + " " + note
            ).strip()

    # A short list of what to actually learn next, not a count of everything the
    # occupation lists. Prefers the refined recommendations (already ranked and
    # pruned of things the candidate obviously has); falls back to the top
    # in-demand true gaps from the skill-capital pass.
    focus_skills: List[str] = [
        r["skill"] for r in recommendations[:3]
    ] or [pretty_skill(g["skill"]) for g in skill_cap.get("focus_skills", [])[:3]]

    skills_from_exp: List[str] = parsed.get("skills_from_experience", [])

    # Notable non-skill strengths: LLM highlights + synthesized experience/education
    highlights: List[str] = list(parsed.get("highlights", []))
    synth: List[str] = []
    if years:
        synth.append(f"{years} year{'s' if years != 1 else ''} of professional experience")
    if edu:
        synth.append(f"{edu} degree")
    # Keep synthesized items only if not already implied by an LLM highlight
    hl_blob = " ".join(highlights).lower()
    for s in synth:
        key = s.split()[0].lower()
        if key not in hl_blob:
            highlights.append(s)

    # What learning each recommended skill would actually move the score by.
    simulation: Optional[Dict] = None
    if target and target_tools is not None and skill_cap:
        exp_fit, edu_fit = _fits(years, edu, market)
        simulation = simulate_additions(
            skills,
            [r["skill"] for r in recommendations] or focus_skills,
            tools=target_tools,
            exp_fit=exp_fit,
            edu_fit=edu_fit,
            weights=stage_weights(hire.get("stage", "early")),
            track=track.get("score", 0.0),
            baseline=skill_cap,
        )
        for entry in simulation["skills"]:
            entry["skill"] = pretty_skill(entry["skill"])


    # The full analysis for every other shortlisted role, precomputed so the
    # reader can switch between them without a round trip. The role currently
    # in view is omitted — it is the report itself.
    role_views: Dict[str, Any] = {}
    if target and target["source"] != "posting":
        for cand in enriched[:_N_ROLES]:
            if cand["soc_code"] == (target.get("soc_code") or ""):
                continue
            role_views[cand["soc_code"]] = _role_view(
                cand, skills=skills, years=years, edu=edu, track=track,
                hot_count=hot_count, matched_total=len(matched_tools),
                highlights=highlights,
            )

    # BLS market snapshot for the matched occupation (None if no data)
    labor_market: Optional[Dict] = (
        _market_snapshot(top["soc_code"], top["title"], market, years, edu)
        if top else None)

    verdict = _verdict(
        hire["level"],
        target_title or (top["title"] if top else None),
        market.get("growth_pct") if market else None,
    )

    return {
        "resume": resume_path.name,
        "parser": parsed.get("parser", "regex"),
        "summary_text": summary_text,
        "verdict": verdict,
        "target": {
            "source": target["source"],
            "title": target["title"],
            "soc_code": target.get("soc_code"),
            "occupation_title": target.get("occupation_title"),
            "posting_skills": target.get("posting_skills", []),
            "role_options": role_options,
            "role_views": role_views,
            "alternatives": [
                {"soc_code": c["soc_code"], "title": c["title"]}
                for c in enriched[:8]
            ],
        } if target else None,
        "score_breakdown": {
            "hirability": hire["components"],
            "competitiveness": competitiveness.get("components", []),
        },
        "simulation": simulation,
        "career_stage": {
            "stage": hire.get("stage"),
            "label": hire.get("stage_label"),
            "note": hire.get("stage_note"),
            "weights": stage_weights(hire.get("stage", "early")),
            "track_record": track,
        },
        "summary": {
            "hirability_score": hire["score"],
            "hirability_range": hire.get("range"),
            "hirability_level": hire["level"],
            "years_experience": years,
            "education_level": edu,
            "top_match_title": top["title"] if top else None,
            "top_match_soc": top["soc_code"] if top else None,
            "skills_extracted": len(skills),
            "skills_from_section": len(parsed.get("skills_from_section", [])),
            "skills_from_experience": len(skills_from_exp),
            "skills_matched_to_onet": len(matched_tools),
        },
        "competitiveness": competitiveness,
        "labor_market": labor_market,
        "top_job_matches": [
            {
                "soc_code": c["soc_code"],
                "title": c["title"],
                "match_score": c["match_score"],
                "semantic_score": c.get("semantic_score"),
                "blended_score": c.get("blended_score", c["match_score"]),
                "matched_skills": c["matched_skills"],
                "description": c["description"],
            }
            for c in enriched[:5]
        ],
        "skills": {
            "extracted": skills,
            "from_section": parsed.get("skills_from_section", []),
            "from_experience": skills_from_exp,
            "matched": matched_tools,
            "unmatched": [m["original"] for m in matched_skills if not m.get("matched_tool")],
        },
        "education": parsed.get("education_lines", []),
        "experience": parsed.get("experience_lines", []),
        "skill_gaps": {
            "coverage_of_top_match": skill_gaps.get("coverage", 0.0),
            "strengths": [
                {**s, "skill": pretty_skill(s["skill"])}
                for s in skill_gaps.get("strengths", [])
            ],
            "highlights": highlights,
            "gaps": [
                {**g, "skill": pretty_skill(g["skill"])}
                for g in skill_gaps.get("gaps", [])[:10]
            ],
            "abstract_skills_required": skill_gaps.get("abstract_skills_required", [])[:6],
        },
        "skill_strength": {
            "score": round(skill_component * 100),
            "matched": skill_cap.get("matched_count", len(skill_gaps.get("strengths", []))),
            "implied": skill_cap.get("implied_count", 0),
            "ignored_basic": skill_cap.get("ignored_basic", 0),
            "focus_skills": focus_skills,
            "breadth": skill_cap.get("breadth"),
            "specialization": skill_cap.get("specialization"),
            "complementarity": skill_cap.get("complementarity"),
        } if skill_cap else None,
        "recommendations": recommendations,
        "ai_displacement_exposure": ai_displacement,
    }


def main_cli() -> None:
    from src.formatter import format_report
    parser = argparse.ArgumentParser(description="Run the full resume analysis pipeline.")
    parser.add_argument("resume", help="Path to a resume file (.txt or .pdf)")
    parser.add_argument("--out", default=None, help="Write report to this file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")
    parser.add_argument("--target-soc", default=None,
                        help="Score against this SOC code instead of the best match")
    parser.add_argument("--job-description", default=None,
                        help="Path to a job posting to score against")
    args = parser.parse_args()

    jd = None
    if args.job_description:
        jd = Path(args.job_description).read_text(encoding="utf-8", errors="ignore")

    report = build_report(args.resume, target_soc=args.target_soc, job_description=jd)

    if args.json:
        output = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        output = format_report(report)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main_cli()
