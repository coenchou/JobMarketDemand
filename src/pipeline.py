from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser import load_master_tables, parse_resume_file

PROCESSED = ROOT / "data" / "processed"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _score_candidate(match_score: float, wage: float | None, growth: float | None) -> float:
    # Lightweight heuristic: reward strong skill overlap and a healthy market signal.
    wage_component = 0.0 if pd.isna(wage) else min(0.2, wage / 200000.0)
    growth_component = 0.0 if pd.isna(growth) else min(0.2, growth / 50.0)
    return round(min(1.0, 0.6 * min(1.0, match_score) + wage_component + growth_component), 3)


def build_match_summary(resume_path: str) -> Dict[str, Any]:
    resume_path = Path(resume_path)
    parsed = parse_resume_file(str(resume_path))
    master_skills, master_occ = load_master_tables()

    skills = parsed.get("skills", [])
    mapped_skills = parsed.get("mapped_skills", [])
    mapped_count = sum(1 for item in mapped_skills if item.get("mapped_element"))
    skill_match_ratio = round(mapped_count / len(skills), 3) if skills else 0.0

    candidates = []
    for item in parsed.get("soc_candidates", []):
        soc_code = item.get("soc_code")
        if not soc_code:
            continue

        occ_rows = master_occ.loc[master_occ["soc_code"] == soc_code]
        if occ_rows.empty:
            continue

        occ = occ_rows.iloc[0]
        wage = _safe_float(occ.get("median_wage"))
        growth = _safe_float(occ.get("growth")) if "growth" in occ else float("nan")

        match_score = item.get("score", 0) / max(1, len(master_skills.loc[master_skills["soc_code"] == soc_code]))
        candidate_score = _score_candidate(match_score, wage, growth)

        candidates.append(
            {
                "soc_code": soc_code,
                "title": occ.get("Title") or occ.get("title") or soc_code,
                "score": round(float(item.get("score", 0)), 2),
                "match_score": round(float(candidate_score), 3),
                "median_wage": None if pd.isna(wage) else round(wage, 2),
                "growth": None if pd.isna(growth) else round(growth, 2),
                "education": occ.get("typical_education_level"),
                "experience": occ.get("typical_experience"),
                "top_work_style": occ.get("top_work_style"),
            }
        )

    candidates = sorted(candidates, key=lambda x: x["match_score"], reverse=True)

    top_candidate = candidates[0] if candidates else None
    hirability_score = 0
    if top_candidate:
        hirability_score = round(min(100, max(0, 100 * top_candidate["match_score"])), 1)

    return {
        "resume": resume_path.name,
        "skills": skills,
        "mapped_skills": mapped_skills,
        "education": parsed.get("education", []),
        "experience_lines": parsed.get("experience_lines", []),
        "years_experience_est": parsed.get("years_experience_est"),
        "skill_match_ratio": skill_match_ratio,
        "hirability_score": hirability_score,
        "top_candidate": top_candidate,
        "candidate_matches": candidates[:10],
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the resume pipeline on one resume file.")
    parser.add_argument("resume", help="Path to a resume text file")
    parser.add_argument("--out", default=None, help="Optional path to write JSON output")
    args = parser.parse_args()

    summary = build_match_summary(args.resume)
    output = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main_cli()
