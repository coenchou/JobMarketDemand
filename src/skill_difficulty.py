"""
How long a skill takes to reach basic working proficiency.

A curated month estimate for ~120 common skills, a token-subset lookup for
names that differ only by vendor prefix or suffix, and a batched embedding
fallback for everything else. Difficulty is what lets the rest of the system
tell a commodity skill from a real one.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.embeddings import _embed, _norm

# ---------------------------------------------------------------------------
# Skill difficulty lookup  (estimated months to reach basic working proficiency)
# ---------------------------------------------------------------------------
SKILL_DIFFICULTY: Dict[str, float] = {
    # Days to a week
    "git": 0.5,
    "github": 0.5,
    "gitlab": 0.5,
    "slack": 0.1,
    "jira": 0.5,
    "confluence": 0.5,
    "trello": 0.2,
    "notion": 0.2,
    "google sheets": 0.5,
    # Consumer apps. Using one is not a skill, and treating them as gaps
    # produces reports that tell an engineer to go learn WhatsApp.
    "gmail": 0.1,
    "whatsapp": 0.1,
    "instagram": 0.2,
    "facebook": 0.2,
    "tiktok": 0.2,
    "linkedin": 0.2,
    "youtube": 0.2,
    "zoom": 0.1,
    "microsoft teams": 0.2,
    "microsoft excel": 0.5,
    "excel": 0.5,
    "powerpoint": 0.5,
    "word": 0.3,
    # Weeks to 1 month
    "sql": 1.5,
    "structured query language sql": 1.5,
    "tableau": 1.5,
    "power bi": 1.5,
    "looker": 2.0,
    "metabase": 1.5,
    "html": 1.0,
    "css": 1.5,
    "bash": 1.0,
    "linux": 2.0,
    "postgresql": 2.0,
    "mysql": 2.0,
    "sqlite": 1.5,
    # 2-4 months
    "python": 3.0,
    "r": 3.0,
    "pandas": 2.0,
    "numpy": 2.0,
    "matplotlib": 2.0,
    "seaborn": 2.0,
    "plotly": 2.0,
    "scikit-learn": 3.0,
    "scikit learn": 3.0,
    "sklearn": 3.0,
    "javascript": 4.0,
    "typescript": 4.0,
    "docker": 3.0,
    "aws": 4.0,
    "amazon web services": 4.0,
    "azure": 4.0,
    "google cloud": 4.0,
    "gcp": 4.0,
    "mongodb": 3.0,
    "redis": 2.0,
    "dbt": 3.0,
    "airflow": 4.0,
    "apache airflow": 4.0,
    "fastapi": 2.0,
    "flask": 2.0,
    "django": 4.0,
    "react": 5.0,
    "vue": 4.0,
    # Modern data / platform tooling the survey-derived tables miss
    "terraform": 4.0,
    "ansible": 3.0,
    "kafka": 4.0,
    "apache kafka": 4.0,
    "snowflake": 3.0,
    "databricks": 4.0,
    "redshift": 3.0,
    "bigquery": 2.5,
    "elasticsearch": 4.0,
    "grafana": 2.0,
    "prometheus": 3.0,
    "graphql": 2.5,
    "nosql": 2.0,
    "ci cd": 3.0,
    "github actions": 1.5,
    "great expectations": 2.0,
    "looker": 2.0,
    # 5-9 months
    "spark": 5.0,
    "apache spark": 5.0,
    "kubernetes": 6.0,
    "tensorflow": 6.0,
    "pytorch": 6.0,
    "java": 7.0,
    "scala": 8.0,
    "golang": 7.0,
    "machine learning": 9.0,
    "statistics": 6.0,
    "statistical analysis": 6.0,
    "data engineering": 8.0,
    "mlops": 10.0,
    "devops": 8.0,
    # 10+ months
    "deep learning": 12.0,
    "neural networks": 12.0,
    "natural language processing": 10.0,
    "computer vision": 12.0,
    "reinforcement learning": 18.0,
    "c++": 12.0,
    "c": 10.0,
    "rust": 14.0,
    "system design": 18.0,
    "distributed systems": 18.0,
}


DEFAULT_DIFFICULTY_MONTHS = 4.0  # unknown skill: assume a moderate lift


def _difficulty_from_table(key: str) -> Optional[float]:
    """
    Look a normalised skill up in SKILL_DIFFICULTY, allowing one name to be a
    token-subset of the other ("spark" → "apache spark", "Microsoft Word" →
    "word"). Token subsets rather than raw substrings: the substring rule let
    one-letter keys like "r" swallow every tool with an r in its name.
    Prefers the most specific matching key.
    """
    if key in SKILL_DIFFICULTY:
        return SKILL_DIFFICULTY[key]
    toks = set(key.split())
    if not toks:
        return None
    best: Optional[Tuple[int, float]] = None
    for known, val in SKILL_DIFFICULTY.items():
        ktoks = set(known.split())
        if ktoks and (ktoks <= toks or toks <= ktoks):
            if best is None or len(ktoks) > best[0]:
                best = (len(ktoks), val)
    return best[1] if best else None


@lru_cache(maxsize=2)
def _difficulty_key_embeddings(n_keys: int) -> Optional[Tuple[Tuple[str, ...], np.ndarray]]:
    """
    Embeddings of every difficulty-table key, cached so the NLP fallback costs
    one matrix multiply per skill instead of re-embedding the table.
    Keyed on table size so runtime additions (emerging skills) invalidate it.
    """
    keys = tuple(SKILL_DIFFICULTY.keys())
    embs = _embed(list(keys))
    return (keys, embs) if embs is not None else None


def batch_difficulty_months(skills: List[str]) -> List[float]:
    """
    Estimate months to basic working proficiency for many skills at once.
    Table lookup first, then a single batched NLP-similarity pass over whatever
    is left. Values land in roughly [0.1, 24.0].
    """
    keys = [_norm(s) for s in skills]
    out: List[Optional[float]] = [_difficulty_from_table(k) for k in keys]

    unknown = [i for i, v in enumerate(out) if v is None]
    if unknown:
        table = _difficulty_key_embeddings(len(SKILL_DIFFICULTY))
        embs = _embed([keys[i] for i in unknown]) if table else None
        if table and embs is not None:
            known_keys, known_embs = table
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = np.nan_to_num(
                    embs @ known_embs.T, nan=0.0, posinf=0.0, neginf=0.0)
            for row, i in enumerate(unknown):
                j = int(np.argmax(sims[row]))
                if sims[row][j] > 0.55:
                    out[i] = SKILL_DIFFICULTY[known_keys[j]]

    return [DEFAULT_DIFFICULTY_MONTHS if v is None else float(v) for v in out]


def get_difficulty_months(skill: str) -> float:
    """
    Estimate months to basic working proficiency for a skill.
    Checks the lookup table first, then uses NLP similarity as a fallback.
    Returns a value in roughly [0.1, 24.0].
    """
    return batch_difficulty_months([skill])[0]


def difficulty_label(months: float) -> str:
    if months <= 0.5:
        return "days"
    if months <= 1.0:
        return "1–2 weeks"
    if months <= 1.5:
        return "2–4 weeks"
    if months <= 2.0:
        return "4–6 weeks"
    if months <= 3.0:
        return "1–3 months"
    if months <= 6.0:
        return "3–6 months"
    if months <= 12.0:
        return "6–12 months"
    return "1+ years"
