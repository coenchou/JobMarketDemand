from __future__ import annotations

import json
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts.resume_parser import extract_resume_sections


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load_master_tables():
    skills_path = PROCESSED / "master_skills.csv"
    occ_path = PROCESSED / "master_occupations.csv"
    if not skills_path.exists() or not occ_path.exists():
        raise FileNotFoundError("Processed master files not found in data/processed/")
    ms = pd.read_csv(skills_path)
    mo = pd.read_csv(occ_path)
    return ms, mo


def map_skills_to_onet(skills: List[str], master_skills: pd.DataFrame) -> List[Dict]:
    """Map extracted free-text skills to ONET element_name using exact/fuzzy matching."""
    candidates = sorted(master_skills['element_name'].dropna().unique())
    mapped = []
    for s in skills:
        # try exact case-insensitive match
        match = next((c for c in candidates if c.lower() == s.lower()), None)
        if not match:
            # fuzzy match on normalized strings
            norm_candidates = {c: re.sub(r"\W+", " ", c).lower() for c in candidates}
            norm_skill = re.sub(r"\W+", " ", s).lower()
            choices = list(norm_candidates.values())
            close = get_close_matches(norm_skill, choices, n=1, cutoff=0.85)
            if close:
                # find original candidate
                match = next(orig for orig, norm in norm_candidates.items() if norm == close[0])

        mapped.append({
            'original': s,
            'mapped_element': match,
        })
    return mapped


def propose_soc_candidates(mapped_elements: List[Dict], master_skills: pd.DataFrame, top_n=5) -> List[Dict]:
    """Propose SOC codes ranked by frequency of the matched elements appearing in occupations."""
    soc_scores = {}
    for item in mapped_elements:
        elem = item.get('mapped_element')
        if not elem:
            continue
        rows = master_skills[master_skills['element_name'] == elem]
        for soc, count in rows['soc_code'].value_counts().items():
            soc_scores[soc] = soc_scores.get(soc, 0) + int(count)

    ranked = sorted(soc_scores.items(), key=lambda x: x[1], reverse=True)
    return [{'soc_code': soc, 'score': score} for soc, score in ranked[:top_n]]


def parse_resume_file(path: str) -> Dict:
    text = Path(path).read_text(encoding='utf-8')
    sections = extract_resume_sections(text)
    master_skills, master_occ = load_master_tables()

    skills = sections.get('skills', [])
    mapped = map_skills_to_onet(skills, master_skills)
    socs = propose_soc_candidates(mapped, master_skills)

    # simple years of experience extractor (looks for 'X years')
    years = None
    m = re.search(r"(\d+)\+?\s+years", text, re.I)
    if m:
        years = int(m.group(1))

    result = {
        'resume': Path(path).name,
        'skills': skills,
        'mapped_skills': mapped,
        'soc_candidates': socs,
        'education': sections.get('education', []),
        'experience_lines': sections.get('experience', []),
        'years_experience_est': years,
    }

    return result


def main_cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('resume', help='Path to resume file')
    parser.add_argument('--out', help='Write JSON output to file', default=None)
    args = parser.parse_args()

    out = parse_resume_file(args.resume)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print('Wrote', args.out)
    else:
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main_cli()
