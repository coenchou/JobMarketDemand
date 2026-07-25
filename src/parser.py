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

# Boundaries use (?<![A-Za-z]) / (?![A-Za-z]) so "B.S." (trailing period, then
# space) still matches and abbreviations don't fire inside words. Bare two-letter
# forms (BS, MA, MS) are omitted — they collide with "MS Office", state codes, etc.
_EDU_PATTERNS: List[tuple] = [
    (r"(?<![A-Za-z])(Ph\.?\s?D\.?|Doctorate|Doctoral|Doctor of)(?![A-Za-z])", "Doctorate"),
    (r"(?<![A-Za-z])(Master['']?s|MBA|M\.B\.A\.?|M\.S\.|M\.A\.|M\.?Sc\.?|M\.Eng\.?)(?![A-Za-z])", "Master's"),
    (r"(?<![A-Za-z])(Bachelor['']?s|B\.S\.|B\.A\.|B\.?Sc\.?|B\.Eng\.?)(?![A-Za-z])", "Bachelor's"),
    (r"(?<![A-Za-z])(Associate['']?s|A\.A\.|A\.S\.)(?![A-Za-z])", "Associate's"),
]


def _find_section(text: str, *headings: str) -> str:
    targets = {h.upper() for h in headings}
    lines = text.splitlines()
    start = None
    inline = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        label = stripped.upper()
        matched = next((t for t in targets if label == t or label.startswith(t)), None)
        if matched:
            start = i + 1
            # capture content on the heading line itself ("SKILLS | SQL | Python")
            rest = stripped[len(matched):].lstrip(" :|-–—\t")
            if rest:
                inline = rest
            break
    if start is None:
        return ""
    section_lines: List[str] = [inline] if inline else []
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
        elif ch in (",", ";", "|") and depth == 0:
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


def _strip_fences(s: str) -> str:
    """Remove ```json ... ``` fences a model may wrap around JSON."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _coerce_years(val) -> Optional[int]:
    """Coerce a years value to a sane int in [0, 60], else None."""
    if isinstance(val, (int, float)):
        y = int(val)
    elif isinstance(val, str):
        m = re.search(r"\d+", val)
        y = int(m.group()) if m else None
    else:
        y = None
    if y is None or not (0 <= y <= 60):
        return None
    return y


def _parse_with_groq(text: str) -> Optional[Dict]:
    """
    Call Groq (Llama 3.3 70B) for structured resume data. Retries once on a
    transient failure, tolerates fenced JSON, and validates every field.
    Returns None if the key is missing or both attempts fail.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    # Cap payload — very long resumes waste tokens and risk truncation.
    prompt = _GROQ_PROMPT.format(text=text[:14000])

    last_err = None
    for attempt in range(2):
        try:
            from groq import Groq  # type: ignore
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(_strip_fences(response.choices[0].message.content))
            if not isinstance(data, dict):
                raise ValueError("model did not return a JSON object")

            data.setdefault("skills", [])
            data.setdefault("implied_skills", [])
            data.setdefault("highlights", [])
            data.setdefault("education_level", None)
            data.setdefault("education_lines", [])
            data.setdefault("experience_lines", [])
            # Coerce list-ish fields defensively (model may return a string)
            for k in ("skills", "implied_skills", "highlights", "education_lines", "experience_lines"):
                if not isinstance(data[k], list):
                    data[k] = [data[k]] if data[k] else []
            data["years_experience"] = _coerce_years(data.get("years_experience"))
            if data["education_level"] not in {"Doctorate", "Master's", "Bachelor's", "Associate's", None}:
                data["education_level"] = None
            return data
        except Exception as e:
            last_err = e
            continue

    print(f"[parser] Groq extraction failed ({type(last_err).__name__}: {last_err}), "
          "falling back to regex parser.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Robust text extraction, skill cleaning, and date-range experience inference
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """Extract PDF text with pdfplumber (better layout), falling back to PyPDF2."""
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import PyPDF2  # type: ignore
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(pg.extract_text() or "" for pg in reader.pages)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}") from e


def _extract_text(path: Path) -> str:
    """Read a resume file to text, robust to format and encoding."""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


# URLs, emails, and long digit runs are never skills
_SKILL_JUNK_RE = re.compile(r"https?://|www\.|@|\d{4,}")


def _clean_skills(raw: List[str]) -> List[str]:
    """
    Normalize and filter a raw skill list: strip bullets/whitespace, drop
    junk (URLs, emails, sentence fragments, number noise), dedup case-insensitively.
    """
    seen: set = set()
    out: List[str] = []
    for item in raw:
        s = re.sub(r"\s+", " ", str(item)).strip().strip("•-*·,;:|")
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        if not (2 <= len(s) <= 50):
            continue
        if len(s.split()) > 6:               # a phrase/sentence, not a skill
            continue
        if _SKILL_JUNK_RE.search(low):
            continue
        if not re.search(r"[a-zA-Z]", s):    # must contain a letter
            continue
        seen.add(low)
        out.append(s)
    return out


_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATE_RANGE_RE = re.compile(
    r"(?:(?:" + _MONTHS + r")\w*\.?\s+)?((?:19|20)\d{2})\s*(?:-|–|—|to|until)\s*"
    r"(?:(?:" + _MONTHS + r")\w*\.?\s+)?((?:19|20)\d{2}|present|current|now|ongoing)",
    re.I,
)


def _infer_years_from_dates(text: str) -> Optional[int]:
    """
    Estimate total professional experience by unioning employment date ranges
    (e.g. "2019 – 2023", "Jan 2020 - Present"). Overlapping spans are merged so
    concurrent roles are not double-counted. Used only when years isn't stated.
    """
    import datetime
    this_year = datetime.date.today().year
    spans: List[tuple] = []
    for m in _DATE_RANGE_RE.finditer(text):
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = this_year if end_raw in ("present", "current", "now", "ongoing") else int(end_raw)
        if 1950 <= start <= this_year and start <= end <= this_year + 1:
            spans.append((start, min(end, this_year)))
    if not spans:
        return None
    spans.sort()
    total, cur_s, cur_e = 0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total if total > 0 else None


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
    text = _extract_text(p)

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

    # --- Normalize skills from either path: drop junk, dedup case-insensitively ---
    skills = _clean_skills(skills)
    implied_skills = _clean_skills(implied_skills)
    skills_from_section = _clean_skills(skills_from_section)

    # --- Infer experience from employment date ranges if not explicitly stated ---
    if not years:
        years = _infer_years_from_dates("\n".join(exp_lines)) or _infer_years_from_dates(text)

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
