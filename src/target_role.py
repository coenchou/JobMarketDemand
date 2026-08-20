"""
What the candidate is being scored *against*.

Every number in a report is relative to one target. Until now that target was
whatever the matcher guessed, which is fragile: pick the wrong SOC and the
skill score, the gaps and the market data are all confidently wrong. This
module makes the target explicit and supports three ways of choosing it:

  auto       the top blended match (previous behaviour)
  override   a SOC the user picked from the ranked alternatives
  posting    a pasted job description — scored against the skills that posting
             actually asks for, with the nearest occupation supplying wage,
             growth and education data

A target is a `tools` frame (tool_name / tool_norm / element_name / is_hot /
in_demand) plus display metadata, so the scoring code does not care which of
the three produced it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.embeddings import _embed, _norm
from src.skill_matcher import _generic_words, _load_sw, get_occ_tools

_HOT_MENTIONS = 2
_MAX_NGRAM = 4


def occupation_frame(soc_code: str) -> pd.DataFrame:
    """The O*NET tool list for an occupation, deduped by normalised name."""
    return get_occ_tools(soc_code).drop_duplicates(subset=["tool_norm"]).copy()


_TOOL_WORDS: Set[str] = {
    "python", "ruby", "scala", "django", "flask", "angular", "react", "excel",
    "tableau", "oracle", "jira", "slack", "notion", "figma", "jenkins",
    "ansible", "terraform", "docker", "kubernetes", "linux", "unix", "git",
    "sql", "nosql", "java", "javascript", "typescript", "kotlin", "swift",
    "spark", "kafka", "hadoop", "hive", "airflow", "snowflake", "databricks",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "kibana",
    "grafana", "prometheus", "salesforce", "wordpress", "photoshop", "looker",
}


_COMMON_WORDS: Set[str] = {
    "go", "access", "word", "project", "analyze", "analysis", "build", "design",
    "test", "monitor", "process", "report", "review", "support", "train",
    "plan", "track", "scale", "ship", "lead", "own", "drive", "swift", "agile",
    "impact", "insight", "solution", "platform", "pipeline", "model", "sketch",
    "jest", "storm", "pig", "primer", "cloud", "spring", "chef", "puppet",
}


_DISPLAY_OVERRIDES: Dict[str, str] = {
    "aws": "AWS", "gcp": "GCP", "sql": "SQL", "nosql": "NoSQL", "css": "CSS",
    "html": "HTML", "nlp": "NLP", "mlops": "MLOps", "dbt": "dbt", "ci cd": "CI/CD",
    "etl": "ETL", "api": "API", "rest apis": "REST APIs", "gis": "GIS",
    "r": "R", "c": "C", "c++": "C++", "javascript": "JavaScript",
    "typescript": "TypeScript", "postgresql": "PostgreSQL", "mysql": "MySQL",
    "mongodb": "MongoDB", "graphql": "GraphQL", "pytorch": "PyTorch",
    "tensorflow": "TensorFlow", "github": "GitHub", "gitlab": "GitLab",
    "power bi": "Power BI", "bigquery": "BigQuery", "openai api": "OpenAI API",
}


def _display(key: str) -> str:
    return _DISPLAY_OVERRIDES.get(key, key.title() if key.islower() else key)


def _is_english(word: str, english: Set[str]) -> bool:
    """Dictionary check that also catches plurals ("pipelines" → "pipeline")."""
    return word in english or (word.endswith("s") and word[:-1] in english)


_VENDORS: Set[str] = {
    "apache", "ibm", "microsoft", "oracle", "google", "amazon", "adobe", "esri",
    "sap", "salesforce", "cisco", "intel", "nvidia", "jetbrains", "atlassian",
    "mathworks", "qlik", "splunk", "elastic", "hashicorp", "autodesk",
    "siemens", "dassault", "ptc", "altair", "ansys", "wolfram", "minitab",
    "statacorp", "mozilla", "canonical", "redhat", "vmware", "citrix", "sas",
    "aws", "hewlett", "packard", "corel", "intuit", "quest", "tibco",
}

_CATEGORY_SUFFIXES: Tuple[str, ...] = (
    "ing", "ion", "ions", "ment", "ments", "ance", "ence", "ity", "ities",
)

_ALIAS_BLOCKLIST: Set[str] = {
    "lifecycle", "workstation", "workspace", "connect", "insight", "insights",
    "discovery", "foundation", "essentials", "enterprise", "professional",
    "standard", "premium", "studio", "server", "office", "teams", "onenote",
}


@lru_cache(maxsize=1)
def _english_words() -> Set[str]:
    """The system word list, used to keep prose out of the skill vocabulary."""
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if len(w.strip()) >= 2}
        except OSError:
            continue
    return _COMMON_WORDS


@lru_cache(maxsize=1)
def _skill_vocabulary() -> Dict[str, str]:
    """
    normalised phrase → canonical display name, covering every O*NET tool plus
    the skills the curated tables know about.

    Single-word entries are filtered hard: O*NET category words ("design",
    "testing") describe no product, and ordinary English words produce constant
    false hits when matched against posting prose.
    """
    from src.recommendations import EMERGING_AI_SKILLS
    from src.skill_difficulty import SKILL_DIFFICULTY

    generic = _generic_words()
    english = _english_words()
    vocab: Dict[str, str] = {}

    for name in SKILL_DIFFICULTY:
        vocab.setdefault(_norm(name), _display(name))
    for entry in EMERGING_AI_SKILLS:
        vocab.setdefault(_norm(entry["skill"]), entry["skill"])

    sw = _load_sw()
    tool_names = {str(tn): str(name) for tn, name in zip(sw["tool_norm"], sw["tool_name"])}
    for tool_norm, tool_name in tool_names.items():
        vocab.setdefault(tool_norm, tool_name)

    for tool_norm, tool_name in tool_names.items():
        words = tool_norm.split()
        if not set(words) & _VENDORS:
            continue
        tokens = [t for t in words
                  if t not in generic and t not in _VENDORS and len(t) > 3]
        if len(tokens) != 1:
            continue
        alias = tokens[0]
        if alias in _ALIAS_BLOCKLIST:
            continue
        if alias in _TOOL_WORDS:
            vocab.setdefault(alias, tool_name)
            continue
        if _is_english(alias, english) or alias.endswith(_CATEGORY_SUFFIXES):
            continue
        vocab.setdefault(alias, tool_name)

    out: Dict[str, str] = {}
    for phrase, name in vocab.items():
        if not phrase or len(phrase) < 2:
            continue
        if len(phrase.split()) == 1:
            if phrase in generic:
                continue
            if _is_english(phrase, english) and phrase not in _TOOL_WORDS:
                continue
        out[phrase] = name
    return out


def extract_posting_skills(text: str) -> List[Tuple[str, int]]:
    """
    Skills a job posting asks for, as (display name, mention count).

    Longest phrase wins, so "Amazon Web Services" is one hit rather than three,
    and a token already consumed by a longer match is not reused.
    """
    vocab = _skill_vocabulary()
    tokens = _norm(text).split()
    counts: Dict[str, int] = {}
    consumed = [False] * len(tokens)

    for size in range(_MAX_NGRAM, 0, -1):
        for i in range(len(tokens) - size + 1):
            if any(consumed[i:i + size]):
                continue
            phrase = " ".join(tokens[i:i + size])
            name = vocab.get(phrase)
            if name is None:
                continue
            counts[name] = counts.get(name, 0) + 1
            for j in range(i, i + size):
                consumed[j] = True

    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def posting_frame(text: str) -> pd.DataFrame:
    """
    Turn a pasted posting into a tools frame. O*NET metadata is carried over
    where the skill is a known tool; anything the posting asks for counts as
    in-demand by definition, and repeated mentions count as hot.
    """
    mentions = extract_posting_skills(text)
    if not mentions:
        return pd.DataFrame(
            columns=["tool_name", "tool_norm", "element_name", "is_hot", "in_demand"])

    sw = _load_sw().drop_duplicates(subset=["tool_norm"])
    by_norm = {str(r.tool_norm): r for r in sw.itertuples()}

    rows = []
    for name, count in mentions:
        norm = _norm(name)
        known = by_norm.get(norm)
        rows.append({
            "tool_name": str(known.tool_name) if known is not None else name,
            "tool_norm": norm,
            "element_name": str(known.element_name) if known is not None else "Posting requirement",
            "is_hot": bool(known.is_hot) if known is not None else count >= _HOT_MENTIONS,
            "in_demand": True,
            "mentions": count,
        })
    return pd.DataFrame(rows)


def _posting_title(text: str) -> str:
    """Best guess at the role name — the first short, title-ish line."""
    for line in text.splitlines():
        line = line.strip(" \t-–—•*#")
        if not line or len(line) > 80:
            continue
        if re.search(r"\b(we|you|our|the team|about)\b", line, re.I):
            continue
        return line
    return ""


def nearest_occupation(
    text: str, top_n: int = 1, skills: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Rank O*NET occupations against a block of text.

    Text similarity alone is unreliable on job postings — an ML platform
    posting reads a lot like Computer Hardware Engineers to an embedding model.
    When the posting's skills are known, the occupation must also actually use
    those tools; the intersection is far more stable, and the answer feeds the
    wage, growth and education figures in the report.
    """
    from src.semantic_matcher import load_occ_embeddings
    from src.skill_matcher import score_soc_candidates

    soc_codes, occ_embs = load_occ_embeddings()
    embs = _embed([text[:4000]])
    if not soc_codes or embs is None:
        return []

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = np.nan_to_num(occ_embs @ embs[0], nan=0.0, posinf=0.0, neginf=0.0)
    by_soc = {soc: float(sims[i]) for i, soc in enumerate(soc_codes)}

    tool_ranked = score_soc_candidates(skills or [], top_n=15) if skills else []
    pool = [c["soc_code"] for c in tool_ranked if c["soc_code"] in by_soc]
    if not pool:
        pool = sorted(by_soc, key=lambda s: -by_soc[s])[:top_n]

    pool.sort(key=lambda s: -by_soc[s])
    return [{"soc_code": s, "similarity": round(by_soc[s], 4)} for s in pool[:top_n]]


def resolve_target(
    candidates: List[Dict[str, Any]],
    *,
    override_soc: Optional[str] = None,
    job_description: Optional[str] = None,
    occ_titles: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Decide what to score against.

    Returns {source, soc_code, title, tools, posting_skills} or None when there
    is nothing to score against at all. `source` is "posting" | "override" |
    "auto" so the report can say which it used.
    """
    titles = occ_titles or {}

    if job_description and job_description.strip():
        frame = posting_frame(job_description)
        near = nearest_occupation(
            job_description,
            skills=[str(t) for t in frame["tool_name"]] if not frame.empty else None,
        )
        soc = near[0]["soc_code"] if near else None
        return {
            "source": "posting",
            "soc_code": soc,
            "title": _posting_title(job_description) or titles.get(soc or "", "this posting"),
            "occupation_title": titles.get(soc or ""),
            "tools": frame,
            "posting_skills": [
                {"skill": r["tool_name"], "mentions": int(r["mentions"])}
                for _, r in frame.iterrows()
            ][:25],
        }

    if override_soc:
        frame = occupation_frame(override_soc)
        if not frame.empty:
            return {
                "source": "override",
                "soc_code": override_soc,
                "title": titles.get(override_soc, override_soc),
                "occupation_title": titles.get(override_soc),
                "tools": frame,
                "posting_skills": [],
            }

    if candidates:
        top = candidates[0]
        return {
            "source": "auto",
            "soc_code": top["soc_code"],
            "title": top.get("title") or titles.get(top["soc_code"], top["soc_code"]),
            "occupation_title": top.get("title"),
            "tools": occupation_frame(top["soc_code"]),
            "posting_skills": [],
        }

    return None
