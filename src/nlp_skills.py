"""
NLP-powered skill analysis: feasibility scoring, complementarity detection,
and AI/automation displacement estimation.

Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings.
Falls back to keyword-based heuristics if the model is unavailable.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

# ---------------------------------------------------------------------------
# Emerging AI/ML skills not yet in ONET (curated, high-demand in tech roles)
# ---------------------------------------------------------------------------

EMERGING_AI_SKILLS: List[Dict] = [
    # Prompt & API layer — very quick to learn
    {"skill": "Prompt Engineering",          "difficulty_months": 1.0,  "tags": ["LLM", "AI"]},
    {"skill": "OpenAI API",                  "difficulty_months": 1.5,  "tags": ["LLM", "AI"]},
    {"skill": "Anthropic Claude API",        "difficulty_months": 1.5,  "tags": ["LLM", "AI"]},
    # Orchestration frameworks
    {"skill": "LangChain",                   "difficulty_months": 2.0,  "tags": ["LLM", "AI"]},
    {"skill": "LlamaIndex",                  "difficulty_months": 2.0,  "tags": ["LLM", "AI"]},
    # Vector databases
    {"skill": "Pinecone",                    "difficulty_months": 1.5,  "tags": ["vector-db", "AI"]},
    {"skill": "Chroma",                      "difficulty_months": 1.5,  "tags": ["vector-db", "AI"]},
    {"skill": "Weaviate",                    "difficulty_months": 2.0,  "tags": ["vector-db", "AI"]},
    # RAG & retrieval
    {"skill": "Retrieval-Augmented Generation (RAG)", "difficulty_months": 3.0, "tags": ["LLM", "AI"]},
    # Hugging Face ecosystem
    {"skill": "Hugging Face Transformers",   "difficulty_months": 3.0,  "tags": ["LLM", "AI", "ML"]},
    {"skill": "Hugging Face Hub",            "difficulty_months": 1.5,  "tags": ["LLM", "AI", "ML"]},
    # Fine-tuning & training
    {"skill": "LLM Fine-tuning",             "difficulty_months": 6.0,  "tags": ["LLM", "AI", "ML"]},
    {"skill": "LoRA / QLoRA",               "difficulty_months": 5.0,  "tags": ["LLM", "AI", "ML"]},
    # AI infra
    {"skill": "AI Agent Development",        "difficulty_months": 4.0,  "tags": ["LLM", "AI"]},
    {"skill": "MLflow",                      "difficulty_months": 3.0,  "tags": ["MLOps", "AI"]},
    {"skill": "Weights & Biases",            "difficulty_months": 2.0,  "tags": ["MLOps", "AI"]},
]

# Anchor skill sets that indicate a tech profile (eligible for AI recommendations)
_TECH_ANCHORS = {
    "python", "javascript", "typescript", "java", "go", "rust", "c++",
    "react", "node", "sql", "aws", "azure", "gcp", "docker", "kubernetes",
    "tensorflow", "pytorch", "scikit", "pandas", "numpy", "spark",
}


def get_emerging_recommendations(
    user_skills: List[str],
    top_n: int = 3,
    max_months: float = 12.0,
) -> List[Dict]:
    """
    Recommend high-demand AI/ML skills not covered by ONET.
    Only runs for tech-oriented profiles (detected by anchor skill overlap).
    """
    skill_norms = {re.sub(r"\W+", " ", s).lower().strip() for s in user_skills}

    # Only recommend AI skills to tech profiles
    if not any(anchor in s for s in skill_norms for anchor in _TECH_ANCHORS):
        return []

    # Also add emerging skill difficulties to the main lookup so feasibility works
    for entry in EMERGING_AI_SKILLS:
        SKILL_DIFFICULTY[entry["skill"].lower()] = entry["difficulty_months"]

    results: List[Dict] = []
    for entry in EMERGING_AI_SKILLS:
        skill_name = entry["skill"]
        if skill_name.lower() in skill_norms:
            continue
        # Also skip if any close variant already known
        norm = re.sub(r"\W+", " ", skill_name).lower().strip()
        if any(norm in s or s in norm for s in skill_norms):
            continue

        fs = score_feasibility(skill_name, user_skills, max_months=max_months)
        if fs["difficulty_months"] > max_months * 1.5:
            continue

        results.append({
            "skill": skill_name,
            "category": "Emerging AI/ML",
            "tags": entry["tags"],
            "is_hot": True,
            "in_demand": True,
            "feasibility": fs["feasibility"],
            "complementarity": fs["complementarity"],
            "effort_label": fs["effort_label"],
            "difficulty_months": fs["difficulty_months"],
            "priority_score": round(fs["feasibility"] * (1.0 + fs["complementarity"]), 4),
            "rationale": _build_rationale(skill_name, True, True, fs["complementarity"], fs["effort_label"]),
        })

    results.sort(key=lambda x: x["priority_score"], reverse=True)
    return results[:top_n]


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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model():
    """Load sentence-transformers model once per process; return None on failure."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _embed(texts: List[str]) -> Optional[np.ndarray]:
    """Return (N, D) float32 embedding matrix, or None if model unavailable."""
    model = _load_model()
    if model is None or not texts:
        return None
    try:
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embs, dtype=np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Skill difficulty
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text).lower().strip()


def get_difficulty_months(skill: str) -> float:
    """
    Estimate months to basic working proficiency for a skill.
    Checks the lookup table first, then uses NLP similarity as a fallback.
    Returns a value in roughly [0.1, 24.0].
    """
    key = _norm(skill)
    if key in SKILL_DIFFICULTY:
        return SKILL_DIFFICULTY[key]

    # Partial match in lookup table
    for known, val in SKILL_DIFFICULTY.items():
        if key in known or known in key:
            return val

    # NLP fallback: find closest known skill by embedding similarity
    embs = _embed([skill] + list(SKILL_DIFFICULTY.keys()))
    if embs is not None:
        skill_emb = embs[0]
        known_embs = embs[1:]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.nan_to_num(known_embs @ skill_emb, nan=0.0, posinf=0.0, neginf=0.0)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] > 0.55:
            return list(SKILL_DIFFICULTY.values())[best_idx]

    return 4.0  # default: ~4 months if completely unknown


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


# ---------------------------------------------------------------------------
# Complementarity scoring
# ---------------------------------------------------------------------------

def compute_complementarity(gap_skill: str, user_skills: List[str]) -> float:
    """
    Score how complementary a gap skill is to the user's existing skill set.
    Complementary = semantically similar (builds on what you know).
    Returns 0.0 (unrelated) to 1.0 (very close to existing skills).
    """
    if not user_skills:
        return 0.0

    # Try NLP embeddings first
    all_texts = [gap_skill] + user_skills
    embs = _embed(all_texts)
    if embs is not None:
        gap_emb = embs[0]
        user_embs = embs[1:]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.nan_to_num(user_embs @ gap_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.clip(np.max(sims), 0.0, 1.0))

    # Keyword fallback: Jaccard similarity on word tokens
    gap_words = set(_norm(gap_skill).split())
    best = 0.0
    for s in user_skills:
        s_words = set(_norm(s).split())
        union = gap_words | s_words
        if union:
            best = max(best, len(gap_words & s_words) / len(union))
    return best


# ---------------------------------------------------------------------------
# Full feasibility scoring for a gap skill
# ---------------------------------------------------------------------------

def score_feasibility(
    gap_skill: str,
    user_skills: List[str],
    max_months: float = 12.0,
) -> Dict:
    """
    Score how feasible it is for this user to learn a gap skill.

    Complementary skills are weighted as effectively easier because the learner
    has a relevant foundation to build on (transfer learning in a human sense).

    Returns:
        difficulty_months  — raw estimated months to proficiency
        complementarity    — 0-1 semantic proximity to user's skills
        effective_months   — difficulty adjusted for complementarity
        feasibility        — 0-1 (1 = very feasible within max_months)
        effort_label       — human-readable effort estimate
    """
    difficulty = get_difficulty_months(gap_skill)
    complementarity = compute_complementarity(gap_skill, user_skills)

    # Complementarity reduces effective difficulty by up to 50%
    effective = difficulty * (1.0 - 0.5 * complementarity)
    feasibility = float(np.clip(1.0 - effective / max_months, 0.0, 1.0))

    return {
        "difficulty_months": round(difficulty, 1),
        "complementarity": round(complementarity, 3),
        "effective_months": round(effective, 1),
        "feasibility": round(feasibility, 3),
        "effort_label": difficulty_label(effective),
    }


# ---------------------------------------------------------------------------
# Skill recommendations
# ---------------------------------------------------------------------------

def _build_rationale(skill: str, is_hot: bool, in_demand: bool,
                     complementarity: float, effort: str) -> str:
    if is_hot:
        demand_note = "hot technology in the current market"
    elif in_demand:
        demand_note = "frequently requested by employers"
    else:
        demand_note = "valued in this occupation"

    if complementarity >= 0.65:
        comp_note = "closely related to your existing skills — you have a strong foundation to build on"
    elif complementarity >= 0.35:
        comp_note = "complements your current skill set"
    else:
        comp_note = "introduces a new area; plan dedicated study time"

    return f"{demand_note.capitalize()}, {comp_note}. Estimated learning time: {effort}."


def generate_recommendations(
    gaps: List[Dict],
    user_skills: List[str],
    top_n: int = 5,
    max_feasibility_months: float = 12.0,
) -> List[Dict]:
    """
    Score and rank gap skills as recommendations.
    Filters out skills that take longer than max_feasibility_months * 1.5 to learn.
    Prioritises: market demand × feasibility.
    """
    scored: List[Dict] = []

    for gap in gaps:
        skill_name = gap["skill"]
        fs = score_feasibility(skill_name, user_skills, max_months=max_feasibility_months)

        # Skip skills far beyond the feasibility window (> 1.5× threshold)
        if fs["difficulty_months"] > max_feasibility_months * 1.5:
            continue

        demand = gap.get("demand_score", 1.0)
        priority = round(demand * fs["feasibility"], 4)

        scored.append({
            "skill": skill_name,
            "element_category": gap.get("element_category", ""),
            "is_hot": gap.get("is_hot", False),
            "in_demand": gap.get("in_demand", False),
            "feasibility": fs["feasibility"],
            "complementarity": fs["complementarity"],
            "effort_label": fs["effort_label"],
            "difficulty_months": fs["difficulty_months"],
            "priority_score": priority,
            "rationale": _build_rationale(
                skill_name,
                gap.get("is_hot", False),
                gap.get("in_demand", False),
                fs["complementarity"],
                fs["effort_label"],
            ),
        })

    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# AI displacement scoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LLM-powered skill suggestions (Groq / Llama 3.3) — beyond dataset coverage
# ---------------------------------------------------------------------------

_LLM_SUGGESTION_PROMPT = """You are a senior career advisor specialising in tech hiring trends.

The candidate is targeting the role: {role}
Their current skills: {skills}

Suggest exactly 4 specific skills or tools that:
1. Are highly valued for {role} in 2025-2026
2. Are NOT already in the candidate's skill list
3. May not appear in standard job databases (ONET etc.) but are gaining rapid adoption

Return ONLY a JSON array, no explanation, no markdown:
[
  {{"skill": "...", "reason": "...", "time_to_learn": "..."}},
  ...
]

Where time_to_learn is a short estimate like "1-2 weeks", "1-3 months", etc."""


def get_llm_skill_suggestions(
    user_skills: List[str],
    target_role: str,
) -> List[Dict]:
    """
    Ask Groq (Llama 3.3) for skill suggestions beyond dataset coverage.
    Returns [] if GROQ_API_KEY is not set or the call fails.
    """
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not target_role:
        return []
    try:
        from groq import Groq  # type: ignore
        import json as _json
        client = Groq(api_key=api_key)
        skills_text = ", ".join(user_skills[:30])
        prompt = _LLM_SUGGESTION_PROMPT.format(role=target_role, skills=skills_text)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        raw = response.choices[0].message.content
        parsed = _json.loads(raw)
        # Handle both {"suggestions": [...]} and [...]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        if not isinstance(parsed, list):
            return []
        results = []
        for item in parsed[:4]:
            if isinstance(item, dict) and "skill" in item:
                results.append({
                    "skill": str(item.get("skill", "")),
                    "reason": str(item.get("reason", "")),
                    "time_to_learn": str(item.get("time_to_learn", "varies")),
                })
        return results
    except Exception as e:
        print(f"[nlp_skills] LLM suggestions unavailable: {e}", file=sys.stderr)
        return []
