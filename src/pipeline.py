"""Full resume analysis pipeline producing a structured JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser import parse_resume
from src.skill_matcher import score_soc_candidates
from src.gap_engine import compute_skill_gaps
from src.nlp_skills import (
    generate_recommendations,
    score_ai_displacement,
    llm_refine_recommendations,
)
from src.labor_market import (
    get_market,
    education_fit,
    experience_fit,
    outlook_score,
    outlook_label,
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


def _hirability(
    skill: float, years: Optional[int], edu: Optional[str],
    market: Optional[Dict],
) -> float:
    """
    Weighted blend of skill coverage, experience fit, and education fit.
    `skill` is a 0-1 coverage component computed by the caller. Experience and
    education are measured against what the occupation actually requires (BLS
    Employment Projections) when available, else fall back to the old curves.
    """
    edu_fit = education_fit(edu, market.get("typical_education")) if market else None
    if edu_fit is None:
        edu_fit = _edu_score(edu)
    exp_fit = experience_fit(years, market.get("typical_experience")) if market else None
    if exp_fit is None:
        exp_fit = _exp_score(years)

    return round((0.45 * skill + 0.30 * exp_fit + 0.25 * edu_fit) * 100, 1)


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


def _competitiveness(
    skill: float,
    years: Optional[int],
    hot_skill_count: int,
    total_matched: int,
    market: Optional[Dict],
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
        score = round((0.35 * skill + 0.25 * hot_ratio + 0.25 * outlook + 0.15 * exp_fit) * 100, 1)
    else:
        # no BLS outlook — reweight onto the other three
        score = round((0.45 * skill + 0.30 * hot_ratio + 0.25 * exp_fit) * 100, 1)

    if score >= 70:
        level, blurb = "Strong", "Your skill set aligns well with market demand and includes high-value technologies."
    elif score >= 45:
        level, blurb = "Competitive", "Solid foundation — adding a few in-demand skills would significantly boost your profile."
    elif score >= 25:
        level, blurb = "Developing", "Core skills are present; targeted gap-filling will be essential to compete."
    else:
        level, blurb = "Entry-level", "Focus on foundational skills; build a portfolio to compensate for limited experience."
    return {"score": score, "level": level, "explanation": blurb}


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(resume_path: str) -> Dict[str, Any]:
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

    top = enriched[0] if enriched else None

    matched_skills: List[Dict] = parsed.get("matched_skills", [])
    matched_tools = [m for m in matched_skills if m.get("matched_tool")]
    hot_count = sum(1 for m in matched_tools if m.get("is_hot"))

    # Compute skill gaps first — its coverage fraction (matched/total occupation
    # tools) is a real 0-1 measure, far more discriminating for scoring than the
    # raw match_score, which saturates for any decent match.
    skill_gaps: Dict = compute_skill_gaps(skills, top["soc_code"]) if top else {}
    coverage_frac = skill_gaps.get("coverage", 0.0)
    # Knee at ~33% coverage = full marks; covering a third of an occupation's
    # entire tool list is exceptional, so only then does the skill term saturate.
    skill_component = min(1.0, coverage_frac * 3.0)

    # BLS labor-market data for the top-matched occupation grounds the scores
    market = get_market(top["soc_code"]) if top else None

    hirability = _hirability(skill_component, years, edu, market)
    competitiveness = _competitiveness(skill_component, years, hot_count, len(matched_tools), market)

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
            top["title"], skills, years, exp_lines, dataset_recs,
        )
        summary_text = refined["summary"]
        recommendations = refined["recommendations"]

        # Enrich the automation-exposure blurb with a concrete note on how AI is
        # actually used in this field (from the same LLM call — no extra request).
        note = refined.get("automation_note", "")
        if note and ai_displacement:
            ai_displacement["explanation"] = (
                ai_displacement.get("explanation", "").rstrip() + " " + note
            ).strip()

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

    # BLS market snapshot for the matched occupation (None if no data)
    labor_market: Optional[Dict] = None
    if top and market:
        labor_market = {
            "soc_code": top["soc_code"],
            "title": top["title"],
            "median_wage": market.get("median_wage"),
            "growth_pct": market.get("growth_pct"),
            "openings_k": market.get("openings_k"),
            "outlook": outlook_label(market.get("growth_pct")),
            "typical_education": market.get("typical_education"),
            "typical_experience": market.get("typical_experience"),
            "education_fit": education_fit(edu, market.get("typical_education")),
            "experience_fit": experience_fit(years, market.get("typical_experience")),
        }

    return {
        "resume": resume_path.name,
        "parser": parsed.get("parser", "regex"),
        "summary_text": summary_text,
        "summary": {
            "hirability_score": hirability,
            "hirability_level": _hirability_level(hirability),
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
            "strengths": skill_gaps.get("strengths", []),
            "highlights": highlights,
            "gaps": skill_gaps.get("gaps", [])[:10],
            "abstract_skills_required": skill_gaps.get("abstract_skills_required", [])[:6],
        },
        "recommendations": recommendations,
        "ai_displacement_exposure": ai_displacement,
    }


def main_cli() -> None:
    from src.formatter import format_report
    parser = argparse.ArgumentParser(description="Run the full resume analysis pipeline.")
    parser.add_argument("resume", help="Path to a resume file (.txt or .pdf)")
    parser.add_argument("--out", default=None, help="Write report to this file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")
    args = parser.parse_args()

    report = build_report(args.resume)

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
