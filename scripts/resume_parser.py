#!/usr/bin/env python3
"""Extract ONET skills and abilities mentioned in the example resumes.

This is an initial parser that uses simple normalized phrase matching against
ONET vocabulary from the downloaded datasets to find skills, software skills,
transferable skills, knowledge, and abilities that appear in the five sample
resumes under resumes/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Prefer the new data layout under data/raw/onet, fall back to legacy datasets/onet
if (ROOT / "data" / "raw" / "onet").exists():
    DATA_DIR = ROOT / "data" / "raw" / "onet"
else:
    DATA_DIR = ROOT / "datasets" / "onet"
RESUME_DIR = ROOT / "resumes"
OUTPUT_DIR = ROOT / "outputs" / "resume_parser"
OUTPUT_FILE = OUTPUT_DIR / "resume_skill_matches.csv"
SUMMARY_FILE = OUTPUT_DIR / "resume_skill_matches_summary.md"


def normalize_text(text: str) -> str:
    """Normalize text for reliable phrase matching."""
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_vocabulary() -> list[str]:
    """Collect candidate ONET terms from the skill/ability-related files."""
    files = [
        "Abilities.xlsx",
        "Essential Skills.xlsx",
        "Software Skills.xlsx",
        "Transferable Skills.xlsx",
        "Knowledge.xlsx",
    ]

    terms = set()
    for file_name in files:
        path = DATA_DIR / file_name
        if not path.exists():
            continue

        df = pd.read_excel(path)

        for column in ("Element Name", "Workplace Example"):
            if column in df.columns:
                terms.update(str(v).strip() for v in df[column].dropna().tolist())

    filtered = []
    for term in terms:
        if not term or len(term.split()) > 5:
            continue
        normalized = normalize_text(term)
        if len(normalized) < 3:
            continue
        filtered.append(term)

    return sorted(set(filtered), key=len, reverse=True)


def find_matches(resume_text: str, vocabulary: list[str]) -> list[str]:
    """Return ONET terms found in the resume text using normalized phrase matching."""
    normalized_resume = normalize_text(resume_text)
    matches = []

    for term in vocabulary:
        normalized_term = normalize_text(term)
        pattern = r"\b" + re.escape(normalized_term) + r"\b"
        if re.search(pattern, normalized_resume):
            matches.append(term)

    unique = []
    seen = set()
    for term in matches:
        if term not in seen:
            unique.append(term)
            seen.add(term)
    return unique


def _clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned = []
    for line in lines:
        text = " ".join(line.split())
        if text:
            cleaned.append(text)
    return cleaned


def extract_list_items(section_text: str) -> list[str]:
    """Split a section into discrete items using common resume delimiters."""
    items = []
    for raw in section_text.splitlines():
        for part in re.split(r"[,;]\s*", raw):
            item = part.strip(" -•")
            if item and item.lower() not in {"skills", "summary", "experience", "education"}:
                items.append(item)
    return list(dict.fromkeys(items))


def find_section(resume_text: str, *headings: str) -> str:
    """Return lines between a heading and the next all-caps heading."""
    lines = resume_text.splitlines()
    targets = {heading.upper() for heading in headings}
    found = False

    for index, raw in enumerate(lines):
        label = raw.strip().upper()
        if label in targets or any(label.startswith(target) for target in targets):
            found = True
            start = index + 1
            break

    if not found:
        return ""

    section_lines = []
    for raw in lines[start:]:
        text = raw.strip()
        label = text.upper()
        if not text:
            section_lines.append(raw)
            continue
        if label.endswith(":"):
            continue

        is_heading = (
            text == label
            and len(text.split()) <= 4
            and label not in targets
        )
        if is_heading:
            break

        section_lines.append(raw)

    return "\n".join(_clean_lines(section_lines))


def extract_resume_sections(resume_text: str) -> dict[str, list[str]]:
    """Return a lightweight analysis of skills, education, and experience from a resume."""
    vocabulary = load_vocabulary()

    skills_section = find_section(resume_text, "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS")
    education_section = find_section(resume_text, "EDUCATION", "EDUCATION & TRAINING")
    experience_section = find_section(resume_text, "EXPERIENCE", "WORK EXPERIENCE")

    explicit_skills = extract_list_items(skills_section)
    explicit_education = _clean_lines(education_section.splitlines())
    explicit_experience = [
        re.sub(r"^[-•]\s*", "", line).strip()
        for line in _clean_lines(experience_section.splitlines())
    ]

    onet_matches = find_matches(resume_text, vocabulary)
    skills = list(dict.fromkeys([*explicit_skills, *onet_matches]))

    education = list(dict.fromkeys(explicit_education))
    if not education:
        for line in resume_text.splitlines():
            if re.search(r"\b(Bachelor|Master|Associate|Doctorate|PhD|MBA|BS|BA|MS|MA)\b", line, re.I):
                education.append(" ".join(line.split()))

    experience = list(dict.fromkeys(explicit_experience))
    if not experience:
        for line in resume_text.splitlines():
            if re.search(r"\b(Experience|Worked|Built|Developed|Led|Managed|Analyzed)\b", line, re.I):
                experience.append(" ".join(line.split()))

    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "matched_terms": onet_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one or more example resumes for skills, education, and experience.")
    parser.add_argument("--resume", type=str, default=None, help="Optional path to a resume file to analyze.")
    args = parser.parse_args()

    vocabulary = load_vocabulary()
    rows = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [Path(args.resume)] if args.resume else sorted(RESUME_DIR.glob("*.txt"))

    for resume_path in targets:
        text = resume_path.read_text(encoding="utf-8")
        section_analysis = extract_resume_sections(text)
        matches = section_analysis["matched_terms"]
        rows.append({
            "resume": resume_path.name,
            "match_count": len(matches),
            "matched_terms": "; ".join(matches),
            "skills": "; ".join(section_analysis["skills"]),
            "education": "; ".join(section_analysis["education"]),
            "experience": "; ".join(section_analysis["experience"]),
        })

    df = pd.DataFrame(rows).sort_values(["match_count", "resume"], ascending=[False, True])
    df.to_csv(OUTPUT_FILE, index=False)

    summary_lines = [
        "# Resume parser",
        "",
        "This parser extracts skills, education, and experience from a resume file and also highlights ONET vocabulary matches from the downloaded datasets.",
        "",
        "## Matches by resume",
        "",
    ]

    for _, row in df.iterrows():
        summary_lines.append(f"- {row['resume']}: {int(row['match_count'])} matched ONET terms")
        if row['skills']:
            summary_lines.append(f"  - Skills: {row['skills']}")
        if row['education']:
            summary_lines.append(f"  - Education: {row['education']}")
        if row['experience']:
            summary_lines.append(f"  - Experience: {row['experience']}")
        if row['matched_terms']:
            summary_lines.append(f"  - ONET term matches: {row['matched_terms']}")

    SUMMARY_FILE.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Loaded {len(vocabulary)} ONET vocabulary terms.")
    print(f"Wrote {len(df)} resume match summaries to {OUTPUT_FILE}")
    print(f"Summary written to {SUMMARY_FILE}")
    print("\nTop results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
