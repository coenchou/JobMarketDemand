"""
Automation / AI displacement exposure for an occupation.

Blends embedding similarity against two anchor descriptions (routine work vs.
human-centric work) with a keyword count over the occupation description.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.embeddings import _embed

# ---------------------------------------------------------------------------
# AI / automation displacement signals
# ---------------------------------------------------------------------------
_AI_RISK_PHRASES = [
    "data entry", "clerical", "routine", "repetitive", "filing", "sorting",
    "scheduling", "processing forms", "check", "verify", "compile records",
    "transcribe", "calculate", "tabulate", "monitor gauges", "inspect",
    "assemble", "package", "load", "unload",
]

_AI_SAFE_PHRASES = [
    "creative", "novel", "negotiate", "persuade", "counsel", "lead",
    "mentor", "innovate", "design strategies", "empathy", "social",
    "interpersonal", "complex judgment", "unstructured", "physical dexterity",
    "patient care", "client relationship", "diagnosis", "ethical",
    "manage conflict", "coach", "inspire",
]

_AI_DISPLACEMENT_ANCHOR = (
    "routine repetitive rule-based data-entry calculation sorting filing clerical"
)
_AI_SAFE_ANCHOR = (
    "creative leadership social empathy complex judgment design innovation coaching"
)


def score_ai_displacement(description: str, title: str) -> Dict:
    """
    Estimate automation / AI displacement risk for an occupation.

    Uses NLP semantic similarity to two anchor phrases
    (automatable tasks vs. human-centric tasks) plus keyword signals.

    Returns:
        score  — 0.0 (low risk) to 1.0 (high risk)
        level  — "Low" | "Medium" | "High"
        explanation — one-sentence summary
    """
    text = (description + " " + title).lower()

    # Keyword signals
    risk_hits = sum(1 for p in _AI_RISK_PHRASES if p in text)
    safe_hits = sum(1 for p in _AI_SAFE_PHRASES if p in text)
    kw_score = 0.5
    if risk_hits + safe_hits > 0:
        kw_score = risk_hits / (risk_hits + safe_hits)

    # NLP signal
    nlp_score = 0.5
    embs = _embed([description[:512], _AI_DISPLACEMENT_ANCHOR, _AI_SAFE_ANCHOR])
    if embs is not None:
        desc_emb, risk_emb, safe_emb = embs[0], embs[1], embs[2]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            risk_sim = float(np.nan_to_num(desc_emb @ risk_emb))
            safe_sim = float(np.nan_to_num(desc_emb @ safe_emb))
        nlp_score = float(np.clip(0.5 + (risk_sim - safe_sim) * 2.0, 0.0, 1.0))

    combined = round(0.6 * nlp_score + 0.4 * kw_score, 3)

    if combined < 0.35:
        level = "Low"
        explanation = (
            "This occupation involves complex judgment, creativity, or social skills "
            "that are difficult to automate."
        )
    elif combined < 0.60:
        level = "Medium"
        explanation = (
            "This role contains a mix of routine tasks and tasks requiring human judgment. "
            "AI will augment rather than replace most of the work."
        )
    else:
        level = "High"
        explanation = (
            "Many tasks in this occupation are routine, rule-based, or data-processing in nature "
            "— areas where AI and automation are advancing rapidly."
        )

    return {"score": combined, "level": level, "explanation": explanation}
