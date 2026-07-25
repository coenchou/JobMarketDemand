"""
Resume parser: extracts skills, education, and experience from a .txt resume.

All text-extraction logic lives here (skill_extractor merged in per project convention).
The skill-to-ONET matching lives in skill_matcher.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env so GOOGLE_API_KEY is available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.skill_matcher import _load_sw, _STOPWORDS, match_skills_to_onet

# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

_SECTION_HEADINGS = {
    "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS", "KEY SKILLS",
    "TECHNOLOGIES", "TOOLS", "COMPETENCIES",
}

# All known resume section names — used to stop section extraction regardless of case
_ALL_KNOWN_HEADINGS = {
    "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS", "KEY SKILLS",
    "TECHNOLOGIES", "TOOLS", "COMPETENCIES",
    "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT",
    "EDUCATION", "EDUCATION & TRAINING", "ACADEMIC BACKGROUND",
    "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "PROFILE", "ABOUT",
    "CERTIFICATIONS", "LICENSES", "PROJECTS", "PORTFOLIO",
    "AWARDS", "HONORS", "PUBLICATIONS", "PRESENTATIONS",
    "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES", "VOLUNTEER",
    "ACHIEVEMENTS", "ACCOMPLISHMENTS", "TRAINING",
}

_QUALIFIER_WORDS = {
    "basic", "limited", "intermediate", "advanced", "proficient",
    "beginner", "familiar", "working knowledge", "exposure",
}

_EDU_PATTERNS: List[tuple] = [
    (r"\b(PhD|Doctorate|Doctor of)\b", "Doctorate"),
    (r"\b(Master['']?s?|MBA|MS|MA|M\.S\.|M\.A\.)\b", "Master's"),
    (r"\b(Bachelor['']?s?|BS|BA|B\.S\.|B\.A\.)\b", "Bachelor's"),
    (r"\b(Associate['']?s?|AA|AS)\b", "Associate's"),
]


def _find_section(text: str, *headings: str) -> str:
    targets = {h.upper() for h in headings}
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        label = line.strip().upper()
        if label in targets or any(label.startswith(t) for t in targets):
            start = i + 1
            break
    if start is None:
        return ""
    section_lines: List[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        upper = stripped.upper()
        # Stop at any known resume section heading (case-insensitive)
        is_known_heading = upper in _ALL_KNOWN_HEADINGS and upper not in targets
        # Also stop at short all-caps lines (catches custom headings like "CERTIFICATIONS")
        is_allcaps_heading = (
            stripped
            and stripped == upper
            and len(stripped.split()) <= 4
            and upper not in targets
        )
        if (is_known_heading or is_allcaps_heading) and section_lines:
            break
        section_lines.append(stripped)
    return "\n".join(ln for ln in section_lines if ln)


# ---------------------------------------------------------------------------
# Skill parsing
# ---------------------------------------------------------------------------

def _split_top_level(line: str) -> List[str]:
    """Split a comma/semicolon line while respecting parenthesis depth."""
    items: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in line + ",":
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in (",", ";") and depth == 0:
            token = "".join(current).strip()
            if token:
                items.append(token)
            current = []
        else:
            current.append(ch)
    return items


def _expand_item(item: str) -> List[str]:
    """
    Expand "Python (pandas, NumPy)" → ["Python", "pandas", "NumPy"].
    Single-word qualifiers like "(basic)" are discarded.
    """
    item = item.strip()
    m = re.match(r"^([^(]+?)(?:\s*\(([^)]*)\))?$", item)
    if not m:
        return [item] if item else []

    main = m.group(1).strip()
    paren = m.group(2)
    result = [main] if main else []

    if paren:
        paren_lower = paren.strip().lower()
        is_qualifier = (
            paren_lower in _QUALIFIER_WORDS
            or (len(paren_lower.split()) <= 2 and "," not in paren_lower)
        )
        if not is_qualifier:
            for sub in paren.split(","):
                sub = sub.strip()
                if sub:
                    result.append(sub)

    return result


def _parse_skills_section(section_text: str) -> List[str]:
    skills: List[str] = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-•*]\s*", "", line)
        # Strip "Category: " prefixes (e.g. "Programming: ", "Databases: ")
        line = re.sub(r"^[A-Za-z][A-Za-z /&-]+:\s*", "", line)
        for item in _split_top_level(line):
            for skill in _expand_item(item):
                skill = skill.strip()
                if 2 <= len(skill) <= 60:
                    skills.append(skill)
    return skills


# ---------------------------------------------------------------------------
# Experience-text NER (dictionary-based)
# ---------------------------------------------------------------------------

def _build_ner_vocab():
    """
    Build a vocabulary for experience-text scanning: {tool_norm: canonical_name}.

    Restricted to Hot Technology or In Demand tools to keep precision high.
    Tools with only stopwords or very short names are excluded because they
    match too broadly in free text (e.g. "R", "C", "Go").
    """
    sw = _load_sw()
    priority = sw[sw["is_hot"] | sw["in_demand"]].drop_duplicates(subset=["tool_norm"])
    vocab: Dict[str, str] = {}
    for _, row in priority.iterrows():
        tn = str(row["tool_norm"])
        # Require at least one non-stopword word of length >= 4
        sig = [w for w in tn.split() if len(w) >= 4 and w not in _STOPWORDS]
        if not sig:
            continue
        vocab[tn] = str(row["tool_name"])
    return vocab


# Cached at module level so it's built once per process
_NER_VOCAB: Dict[str, str] = {}


def _get_ner_vocab() -> Dict[str, str]:
    global _NER_VOCAB
    if not _NER_VOCAB:
        _NER_VOCAB = _build_ner_vocab()
    return _NER_VOCAB


def extract_tools_from_experience(
    experience_lines: List[str],
    existing_skill_norms: set,
) -> List[Dict]:
    """
    Scan experience bullet points for ONET tool names not already in the
    skills section. Uses the Software Skills vocabulary as a named-entity
    dictionary (dictionary-based NER).

    Returns a list of dicts: {original, matched_tool, element_name,
    is_hot, in_demand, match_type, source}.
    """
    if not experience_lines:
        return []

    vocab = _get_ner_vocab()
    sw = _load_sw()

    # Normalise the full experience block once
    exp_norm = re.sub(r"\W+", " ", " ".join(experience_lines)).lower()

    found: List[Dict] = []
    seen_norms: set = set(existing_skill_norms)

    for tool_norm, tool_name in vocab.items():
        if tool_norm in seen_norms:
            continue

        # Multi-word tool: require contiguous phrase match ("apache spark" in text)
        # Single-word tool: require word-boundary match to avoid "r" matching "router"
        words = tool_norm.split()
        if len(words) > 1:
            matched = tool_norm in exp_norm
        else:
            matched = bool(re.search(r"\b" + re.escape(tool_norm) + r"\b", exp_norm))

        if not matched:
            continue

        seen_norms.add(tool_norm)

        # Look up metadata for this tool
        rows = sw[sw["tool_norm"] == tool_norm]
        if rows.empty:
            continue
        row = rows.iloc[0]
        found.append({
            "original": tool_name,
            "matched_tool": tool_name,
            "element_name": str(row["element_name"]),
            "is_hot": bool(rows["is_hot"].any()),
            "in_demand": bool(rows["in_demand"].any()),
            "match_type": "experience_ner",
            "source": "experience",
        })

    return found


# ---------------------------------------------------------------------------
# Groq LLM parser (Llama 3.1)
# ---------------------------------------------------------------------------

_GROQ_PROMPT = """You are a resume parser. Extract structured information from the resume below.
Return ONLY a valid JSON object — no explanation, no markdown fences.

Required fields:
- "skills": array of all skills, tools, and technologies EXPLICITLY mentioned anywhere in the resume
- "implied_skills": array of skills the candidate almost certainly has based on their roles, seniority, and projects, even if never written out. Be conservative and specific: a software engineer who shipped production code implies "Git" and "GitHub"; a data analyst who built dashboards implies "data visualization"; a backend engineer implies "REST APIs". Only high-confidence implications a hiring manager would take for granted. Do NOT repeat items already in "skills".
- "highlights": array of 2 to 5 short strings naming the candidate's most impressive, resume-defining credentials — notable employers, selective schools, scale of impact, leadership, or standout achievements. Examples: "Software Engineer at Google (3 yrs)", "B.S. Computer Science, MIT", "Led a team of 8 engineers", "Scaled platform to 2M users". Empty array if nothing genuinely stands out. Do not invent.
- "years_experience": total years of professional experience as an integer, or null
- "education_level": highest degree, must be exactly one of: "Doctorate", "Master's", "Bachelor's", "Associate's", or null
- "education_lines": array of lines from the education section
- "experience_lines": array of job titles, company names, dates, and bullet points from work experience

Resume:
{text}"""


def _parse_with_groq(text: str) -> Optional[Dict]:
    """Call Groq (Llama 3.3 70B) to extract structured resume data. Returns None if unavailable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": _GROQ_PROMPT.format(text=text)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        data.setdefault("skills", [])
        data.setdefault("implied_skills", [])
        data.setdefault("highlights", [])
        data.setdefault("years_experience", None)
        data.setdefault("education_level", None)
        data.setdefault("education_lines", [])
        data.setdefault("experience_lines", [])
        if isinstance(data["years_experience"], str):
            m = re.search(r"\d+", data["years_experience"])
            data["years_experience"] = int(m.group()) if m else None
        return data
    except Exception as e:
        print(f"[parser] Groq extraction failed ({type(e).__name__}: {e}), falling back to regex parser.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_resume(path: str) -> Dict:
    """
    Parse a resume file and return structured metadata.

    Returns:
        skills              — deduplicated skill strings
        matched_skills      — ONET match results for each skill
        years_experience    — estimated years (int or None)
        education_level     — highest degree label (str or None)
        education_lines     — raw education section lines
        experience_lines    — raw experience section lines
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            import PyPDF2  # type: ignore
            with open(p, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
        except Exception as e:
            raise ValueError(f"Could not read PDF: {e}") from e
    else:
        text = p.read_text(encoding="utf-8")

    # --- Implicit skills from profile URLs ---
    implicit_skills: List[str] = []
    if re.search(r"github\.com/", text, re.I):
        implicit_skills += ["Git", "GitHub"]
    if re.search(r"gitlab\.com/", text, re.I):
        implicit_skills += ["Git", "GitLab"]

    # --- Try Groq LLM parser first ---
    llm = _parse_with_groq(text)

    implied_skills: List[str] = []
    highlights: List[str] = []
    if llm:
        skills: List[str] = list(llm["skills"])
        implied_skills = [str(s) for s in llm.get("implied_skills", []) if s]
        highlights = [str(h) for h in llm.get("highlights", []) if h]
        years: Optional[int] = llm["years_experience"]
        edu_level: Optional[str] = llm["education_level"]
        edu_lines: List[str] = llm["education_lines"]
        exp_lines: List[str] = llm["experience_lines"]
        skills_from_section = list(skills)  # LLM extracts from whole resume
    else:
        # --- Regex fallback ---
        skills_section = _find_section(text, *_SECTION_HEADINGS)
        raw_skills = _parse_skills_section(skills_section)

        seen: set = set()
        skills = []
        for s in raw_skills:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                skills.append(s)

        exp_section = _find_section(text, "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE")
        exp_lines = [ln.strip() for ln in exp_section.splitlines() if ln.strip()]

        years = None
        for m in re.finditer(r"(\d+)\+?\s+years?", text, re.I):
            val = int(m.group(1))
            if val < 40:
                years = max(years or 0, val)

        edu_level = None
        for pattern, label in _EDU_PATTERNS:
            if re.search(pattern, text, re.I):
                edu_level = label
                break

        edu_section = _find_section(text, "EDUCATION", "EDUCATION & TRAINING")
        edu_lines = [ln.strip() for ln in edu_section.splitlines() if ln.strip()]
        skills_from_section = skills

    # --- Merge implicit (URL) + implied (LLM-inferred) skills ---
    # These suppress false gaps: a senior engineer who didn't list "Git" still
    # shouldn't be told to learn it. Deduped against already-known skills.
    known_lower = {s.lower() for s in skills}
    for imp in implicit_skills + implied_skills:
        if imp.lower() not in known_lower:
            skills.append(imp)
            known_lower.add(imp.lower())

    # --- ONET matching (always runs regardless of parser path) ---
    matched_skills = match_skills_to_onet(skills)

    # --- NER scan of experience lines for additional tool mentions ---
    existing_norms = {re.sub(r"\W+", " ", s).lower().strip() for s in skills}
    ner_skills = extract_tools_from_experience(exp_lines, existing_norms)

    all_matched = matched_skills + ner_skills
    all_skills = skills + [n["original"] for n in ner_skills]

    return {
        "resume": Path(path).name,
        "skills": all_skills,
        "skills_from_section": skills_from_section,
        "skills_from_experience": [n["original"] for n in ner_skills],
        "implied_skills": implied_skills,
        "highlights": highlights,
        "matched_skills": all_matched,
        "years_experience": years,
        "education_level": edu_level,
        "education_lines": edu_lines,
        "experience_lines": exp_lines,
        "parser": "groq/llama-3.3-70b" if llm else "regex",
    }


# ---------------------------------------------------------------------------
# CLI (parser-only mode, for quick inspection)
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse a resume file (extraction only).")
    ap.add_argument("resume", help="Path to resume .txt file")
    ap.add_argument("--out", default=None, help="Write JSON output to file")
    args = ap.parse_args()

    result = parse_resume(args.resume)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    _cli()
