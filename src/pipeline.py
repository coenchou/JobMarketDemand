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
    get_emerging_recommendations,
    is_technical_occupation,
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


def _hirability(coverage: float, years: Optional[int], edu: Optional[str]) -> float:
    # coverage is sparse (most resumes cover <15% of all occ tools), so amplify it
    skill_c = 0.5 * min(1.0, coverage * 6)
    exp_c = 0.3 * _exp_score(years)
    edu_c = 0.2 * _edu_score(edu)
    return round((skill_c + exp_c + edu_c) * 100, 1)


def _competitiveness(
    coverage: float,
    years: Optional[int],
    hot_skill_count: int,
    total_matched: int,
) -> Dict:
    hot_ratio = hot_skill_count / max(1, total_matched)
    score = round(
        (0.4 * min(1.0, coverage * 6) + 0.3 * _exp_score(years) + 0.3 * hot_ratio) * 100, 1
    )
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
    coverage = top["match_score"] if top else 0.0

    matched_skills: List[Dict] = parsed.get("matched_skills", [])
    matched_tools = [m for m in matched_skills if m.get("matched_tool")]
    hot_count = sum(1 for m in matched_tools if m.get("is_hot"))

    hirability = _hirability(coverage, years, edu)
    competitiveness = _competitiveness(coverage, years, hot_count, len(matched_tools))

    # Skill gap analysis and NLP recommendations for top candidate
    skill_gaps: Dict = {}
    recommendations: List[Dict] = []
    ai_displacement: Optional[Dict] = None
    relevant_ai_skills: List[Dict] = []

    if top:
        skill_gaps = compute_skill_gaps(skills, top["soc_code"])
        recommendations = generate_recommendations(
            skill_gaps.get("gaps", []),
            skills,
            top_n=5,
        )
        ai_displacement = score_ai_displacement(
            top.get("description", ""),
            top.get("title", ""),
        )
        # AI/ML additions to the gap list — only for technical occupations
        if is_technical_occupation(top["soc_code"]):
            relevant_ai_skills = get_emerging_recommendations(skills, top_n=3)

    skills_from_exp: List[str] = parsed.get("skills_from_experience", [])

    return {
        "resume": resume_path.name,
        "parser": parsed.get("parser", "regex"),
        "summary": {
            "hirability_score": hirability,
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
            "gaps": skill_gaps.get("gaps", [])[:10],
            "abstract_skills_required": skill_gaps.get("abstract_skills_required", [])[:6],
            "relevant_ai_skills": relevant_ai_skills,
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
