"""
Backwards-compatible facade over the NLP skill modules.

This module used to hold everything — the embedding model, the difficulty
table, ~450 lines of curated skill prose, feasibility scoring, automation
exposure and the LLM refinement layer. Those now live in focused modules:

    embeddings.py        sentence-transformers access, text normalisation
    skill_difficulty.py  months-to-proficiency estimates
    skill_notes.py       curated per-skill and per-category prose
    recommendations.py   feasibility, ranking, emerging skills, LLM refinement
    ai_exposure.py       automation displacement scoring

Import from those directly in new code. Everything is re-exported here so
existing call sites keep working.
"""

from __future__ import annotations

from src.embeddings import _embed, _load_model, _norm, embed, normalise
from src.skill_difficulty import (
    DEFAULT_DIFFICULTY_MONTHS,
    SKILL_DIFFICULTY,
    batch_difficulty_months,
    difficulty_label,
    get_difficulty_months,
)
from src.skill_notes import build_skill_description
from src.recommendations import (
    EMERGING_AI_SKILLS,
    compute_complementarity,
    generate_recommendations,
    get_emerging_recommendations,
    is_technical_occupation,
    llm_refine_recommendations,
    score_feasibility,
)
from src.ai_exposure import score_ai_displacement

__all__ = [
    "DEFAULT_DIFFICULTY_MONTHS",
    "EMERGING_AI_SKILLS",
    "SKILL_DIFFICULTY",
    "_embed",
    "_load_model",
    "_norm",
    "batch_difficulty_months",
    "build_skill_description",
    "compute_complementarity",
    "difficulty_label",
    "embed",
    "generate_recommendations",
    "get_difficulty_months",
    "get_emerging_recommendations",
    "is_technical_occupation",
    "llm_refine_recommendations",
    "normalise",
    "score_ai_displacement",
    "score_feasibility",
]
