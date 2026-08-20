"""
Turning skill gaps into a ranked, explained list of what to learn next.

Three layers:
  1. feasibility  — difficulty discounted by how much the candidate's existing
                    skills transfer (complementarity).
  2. dataset recs — O*NET gaps scored by demand x feasibility.
  3. LLM refine   — the dataset list is blunt (it will tell a senior engineer to
                    learn Git), so the candidate's real context is handed to an
                    LLM to prune, add current skills the datasets miss, and
                    write concrete guidance. Falls back to the dataset list.

Also carries the curated emerging-AI skill list, which no occupational dataset
tracks yet.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from src.embeddings import _embed, _norm
from src.skill_difficulty import (
    SKILL_DIFFICULTY,
    difficulty_label,
    get_difficulty_months,
)
from src.skill_notes import _closest_user_skill, build_skill_description


EMERGING_AI_SKILLS: List[Dict] = [
    {"skill": "Prompt Engineering",          "difficulty_months": 1.0,  "tags": ["LLM", "AI"]},
    {"skill": "OpenAI API",                  "difficulty_months": 1.5,  "tags": ["LLM", "AI"]},
    {"skill": "Anthropic Claude API",        "difficulty_months": 1.5,  "tags": ["LLM", "AI"]},
    {"skill": "LangChain",                   "difficulty_months": 2.0,  "tags": ["LLM", "AI"]},
    {"skill": "LlamaIndex",                  "difficulty_months": 2.0,  "tags": ["LLM", "AI"]},
    {"skill": "Pinecone",                    "difficulty_months": 1.5,  "tags": ["vector-db", "AI"]},
    {"skill": "Chroma",                      "difficulty_months": 1.5,  "tags": ["vector-db", "AI"]},
    {"skill": "Weaviate",                    "difficulty_months": 2.0,  "tags": ["vector-db", "AI"]},
    {"skill": "Retrieval-Augmented Generation (RAG)", "difficulty_months": 3.0, "tags": ["LLM", "AI"]},
    {"skill": "Hugging Face Transformers",   "difficulty_months": 3.0,  "tags": ["LLM", "AI", "ML"]},
    {"skill": "Hugging Face Hub",            "difficulty_months": 1.5,  "tags": ["LLM", "AI", "ML"]},
    {"skill": "LLM Fine-tuning",             "difficulty_months": 6.0,  "tags": ["LLM", "AI", "ML"]},
    {"skill": "LoRA / QLoRA",               "difficulty_months": 5.0,  "tags": ["LLM", "AI", "ML"]},
    {"skill": "AI Agent Development",        "difficulty_months": 4.0,  "tags": ["LLM", "AI"]},
    {"skill": "MLflow",                      "difficulty_months": 3.0,  "tags": ["MLOps", "AI"]},
    {"skill": "Weights & Biases",            "difficulty_months": 2.0,  "tags": ["MLOps", "AI"]},
]

_TECHNICAL_SOC_PREFIXES = ("15-", "17-")
_TECHNICAL_SOC_EXTRAS = {"11-3021"}


def is_technical_occupation(soc_code: str) -> bool:
    """True if the SOC code falls in a technical field (engineering, data, software, IT)."""
    if not soc_code:
        return False
    return soc_code.startswith(_TECHNICAL_SOC_PREFIXES) or soc_code in _TECHNICAL_SOC_EXTRAS


def get_emerging_recommendations(
    user_skills: List[str],
    top_n: int = 3,
    max_months: float = 12.0,
) -> List[Dict]:
    """
    Recommend AI/ML skills not covered by ONET's technology data.
    Callers should gate this on is_technical_occupation() for the top match.
    """
    skill_norms = {re.sub(r"\W+", " ", s).lower().strip() for s in user_skills}

    for entry in EMERGING_AI_SKILLS:
        SKILL_DIFFICULTY[entry["skill"].lower()] = entry["difficulty_months"]

    results: List[Dict] = []
    for entry in EMERGING_AI_SKILLS:
        skill_name = entry["skill"]
        if skill_name.lower() in skill_norms:
            continue
        norm = re.sub(r"\W+", " ", skill_name).lower().strip()
        if any(norm in s or s in norm for s in skill_norms):
            continue

        fs = score_feasibility(skill_name, user_skills, max_months=max_months)
        if fs["difficulty_months"] > max_months * 1.5:
            continue

        closest, sim = _closest_user_skill(skill_name, user_skills)
        rationale, trend = build_skill_description(
            skill_name,
            is_hot=True,
            closest=closest,
            closest_sim=sim,
            months=fs["difficulty_months"],
            n_occupations=0,
        )

        results.append({
            "skill": skill_name,
            "category": "Emerging AI/ML",
            "tags": entry["tags"],
            "is_hot": True,
            "in_demand": True,
            "demand_trend": trend,
            "feasibility": fs["feasibility"],
            "complementarity": fs["complementarity"],
            "effort_label": fs["effort_label"],
            "difficulty_months": fs["difficulty_months"],
            "priority_score": round(fs["feasibility"] * (1.0 + fs["complementarity"]), 4),
            "rationale": rationale,
        })

    results.sort(key=lambda x: x["priority_score"], reverse=True)
    return results[:top_n]


def compute_complementarity(gap_skill: str, user_skills: List[str]) -> float:
    """
    Score how complementary a gap skill is to the user's existing skill set.
    Complementary = semantically similar (builds on what you know).
    Returns 0.0 (unrelated) to 1.0 (very close to existing skills).
    """
    if not user_skills:
        return 0.0

    all_texts = [gap_skill] + user_skills
    embs = _embed(all_texts)
    if embs is not None:
        gap_emb = embs[0]
        user_embs = embs[1:]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.nan_to_num(user_embs @ gap_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.clip(np.max(sims), 0.0, 1.0))

    gap_words = set(_norm(gap_skill).split())
    best = 0.0
    for s in user_skills:
        s_words = set(_norm(s).split())
        union = gap_words | s_words
        if union:
            best = max(best, len(gap_words & s_words) / len(union))
    return best


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

    effective = difficulty * (1.0 - 0.5 * complementarity)
    feasibility = float(np.clip(1.0 - effective / max_months, 0.0, 1.0))

    return {
        "difficulty_months": round(difficulty, 1),
        "complementarity": round(complementarity, 3),
        "effective_months": round(effective, 1),
        "feasibility": round(feasibility, 3),
        "effort_label": difficulty_label(effective),
    }


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
    from src.skill_matcher import count_occupations_for_tool

    scored: List[Dict] = []

    for gap in gaps:
        skill_name = gap["skill"]
        fs = score_feasibility(skill_name, user_skills, max_months=max_feasibility_months)

        if fs["difficulty_months"] > max_feasibility_months * 1.5:
            continue

        demand = gap.get("demand_score", 1.0)
        priority = round(demand * fs["feasibility"], 4)

        closest, sim = _closest_user_skill(skill_name, user_skills)
        rationale, trend = build_skill_description(
            skill_name,
            category=gap.get("element_category", ""),
            is_hot=gap.get("is_hot", False),
            in_demand=gap.get("in_demand", False),
            closest=closest,
            closest_sim=sim,
            months=fs["difficulty_months"],
            n_occupations=count_occupations_for_tool(skill_name),
        )

        scored.append({
            "skill": skill_name,
            "element_category": gap.get("element_category", ""),
            "is_hot": gap.get("is_hot", False),
            "in_demand": gap.get("in_demand", False),
            "demand_trend": trend,
            "feasibility": fs["feasibility"],
            "complementarity": fs["complementarity"],
            "effort_label": fs["effort_label"],
            "difficulty_months": fs["difficulty_months"],
            "priority_score": priority,
            "rationale": rationale,
        })

    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored[:top_n]


_SKILL_FAMILIES: List[set] = [
    {"git", "github", "gitlab", "version control", "bitbucket"},
    {"amazon web services", "aws"},
    {"microsoft azure", "azure"},
    {"google cloud", "gcp", "google cloud platform"},
    {"postgresql", "postgres"},
    {"javascript", "js"},
]


def _family_key(skill: str) -> str:
    n = _norm(skill)
    for fam in _SKILL_FAMILIES:
        if n in fam or any(m in n for m in fam):
            return sorted(fam)[0]
    return n


def _dedupe_by_family(recs: List[Dict]) -> List[Dict]:
    seen: set = set()
    out: List[Dict] = []
    for r in recs:
        k = _family_key(r.get("skill", ""))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _fallback_summary(role: str, recs: List[Dict], years: Optional[int]) -> str:
    role = role or "your target roles"
    if not recs:
        return (
            f"Your profile already covers the core tools for {role}. Focus on "
            "deepening what you have and building portfolio evidence rather than "
            "adding new tools."
        )
    top = [r["skill"] for r in recs[:3]]
    lead = top[0]
    rest = ", ".join(top[1:]) if len(top) > 1 else ""
    exp_note = ""
    if years is not None:
        exp_note = f" With {years} year{'s' if years != 1 else ''} of experience, you can "
        exp_note += "target senior postings once these land." if years >= 5 else "close these gaps and move up quickly."
    return (
        f"For {role}, the highest-leverage next step is {lead}"
        + (f", followed by {rest}" if rest else "")
        + f".{exp_note}"
    ).strip()


_REFINE_PROMPT = """You are a career advisor. Refine a skill-gap list for one candidate.

CANDIDATE
Target role: {role}
Years of experience: {years}
Skills they already have (explicit + safely implied): {skills}
Recent experience:
{experience}

DRAFT GAP LIST (from an occupational database; may be blunt or wrong):
{gaps}

Return ONLY JSON:
{{
  "summary": "2-3 sentences of concrete next-steps guidance. Answer: what direction fits them, what to learn next, and how quickly they can improve. Reference their actual background. No fluff, no 'leverage'/'unlock'.",
  "automation_note": "1-2 plain sentences on how AI and automation are actually being used in the {role} field today (concrete examples of tools or tasks), and what that means for someone working in it. Factual and specific, not alarmist.",
  "recommendations": [
    {{"skill": "...", "rationale": "one specific sentence tying this skill to THIS candidate's background and target role", "effort": "e.g. 2-4 weeks", "source": "dataset" or "new"}}
  ]
}}

Rules:
- DROP any gap the candidate almost certainly already has given their seniority and experience (e.g. never tell an experienced software engineer to learn Git or version control).
- KEEP the most valuable remaining gaps from the draft list.
- You MAY ADD up to 2 current, high-value skills this specific person needs that the database does not track (these can be modern frameworks, AI/ML tools, cloud/platform skills — whatever actually fits THIS resume and role; do not force AI skills onto a non-technical candidate). Mark added ones "source": "new".
- Return at most 5 recommendations, ordered by priority.
- Every rationale must be specific to this candidate, not generic."""


def llm_refine_recommendations(
    role: str,
    skills: List[str],
    years: Optional[int],
    experience_lines: List[str],
    dataset_recs: List[Dict],
) -> Dict[str, Any]:
    """
    Refine the dataset gap list with the candidate's real context.
    Returns {"summary": str, "recommendations": [...]}.
    Falls back to the dataset list (deduped) plus a templated summary.
    """
    deduped = _dedupe_by_family(dataset_recs)
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {"summary": _fallback_summary(role, deduped, years), "automation_note": "", "recommendations": deduped[:5]}

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        gap_lines = "\n".join(
            f"- {r['skill']}: {r.get('rationale', '')}" for r in deduped[:8]
        ) or "- (none found)"
        exp = "\n".join(f"- {ln}" for ln in experience_lines[:6]) or "- (not provided)"
        prompt = _REFINE_PROMPT.format(
            role=role or "unspecified",
            years="unknown" if years is None else years,
            skills=", ".join(skills[:40]) or "none listed",
            experience=exp,
            gaps=gap_lines,
        )
        from src.llm_cache import MODEL, complete
        data = json.loads(
            complete(client, MODEL, prompt, temperature=0.3))
        summary = str(data.get("summary", "")).strip() or _fallback_summary(role, deduped, years)
        automation_note = str(data.get("automation_note", "")).strip()

        by_name = {_norm(r["skill"]): r for r in dataset_recs}
        out: List[Dict] = []
        for item in data.get("recommendations", [])[:5]:
            name = str(item.get("skill", "")).strip()
            if not name:
                continue
            base = by_name.get(_norm(name))
            source = item.get("source", "dataset" if base else "new")
            if base:
                effort_label = base.get("effort_label", "")
                demand_trend = base.get("demand_trend", "")
            else:
                fs = score_feasibility(name, skills)
                effort_label = fs["effort_label"]
                demand_trend = "growing"
            out.append({
                "skill": name,
                "rationale": str(item.get("rationale", "")).strip(),
                "effort_label": item.get("effort") or effort_label,
                "demand_trend": demand_trend,
                "source": source,
                "is_hot": bool(base.get("is_hot")) if base else True,
            })

        out = _dedupe_by_family(out)
        if not out:
            out = deduped[:5]
        return {"summary": summary, "automation_note": automation_note, "recommendations": out}
    except Exception as e:
        print(f"[nlp_skills] recommendation refinement failed: {e}", file=sys.stderr)
        return {"summary": _fallback_summary(role, deduped, years), "automation_note": "", "recommendations": deduped[:5]}
