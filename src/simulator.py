"""
What a skill is actually worth to this candidate.

The report can tell someone that Terraform is the highest-priority gap without
telling them what closing it buys. This re-runs the skill-capital model with
the skill added and reports the movement in the headline score, so a
recommendation carries a number: "+3 hirability".

The delta is not just the tool's own weight. Adding a skill can also raise
specialization and complementarity, and it can flip other tools from gap to
implied — learning Kubernetes credits Docker and Linux too. Re-running the
whole model captures all of that; approximating it would not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.skill_score import _PRIOR, compute_skill_capital

# Simulating is a full classification pass per skill, so cap how many the
# report offers to try.
MAX_SIMULATED = 6


def _hirability(
    skill_component: float,
    exp_fit: float,
    edu_fit: float,
    weights: Dict[str, float],
    track: float,
) -> float:
    """
    Same blend as pipeline._hirability, with the fits already resolved. The
    weights are passed in rather than hard-coded so a simulated delta always
    moves the score the reader is actually looking at — these vary by career
    stage, and a stale copy here would quietly disagree with the headline.
    """
    return round(100 * (
        weights["skill"] * skill_component
        + weights["experience"] * exp_fit
        + weights["education"] * edu_fit
        + weights["track_record"] * track
    ), 1)


def simulate_additions(
    user_skills: List[str],
    candidate_skills: List[str],
    *,
    tools: pd.DataFrame,
    exp_fit: float,
    edu_fit: float,
    weights: Dict[str, float],
    track: float = 0.0,
    baseline: Optional[Dict[str, Any]] = None,
    max_skills: int = MAX_SIMULATED,
) -> Dict[str, Any]:
    """
    Score the effect of learning each candidate skill, one at a time.

    Returns {baseline_score, skills: [{skill, score, delta, skill_strength,
    also_unlocks}]}, ranked by delta. `also_unlocks` counts the extra tools that
    became credited beyond the skill itself — the transfer effect.
    """
    base = baseline or compute_skill_capital(user_skills, tools=tools)
    base_score = _hirability(base.get("score", 0.0), exp_fit, edu_fit, weights, track)
    base_matched = base.get("matched_count", 0)
    base_evidence = base.get("evidence", 1.0)

    results: List[Dict[str, Any]] = []
    for skill in candidate_skills[:max_skills]:
        if not skill or not skill.strip():
            continue
        with_skill = compute_skill_capital(user_skills + [skill], tools=tools)

        # Hold the evidence weight at the baseline. Learning a skill also makes
        # the resume document more, which un-shrinks the estimate — for a thin
        # resume that can drag the score *down* and produce "learn this, lose a
        # point". The question here is what the skill is worth, not how much
        # better documented the candidate would be, so only the measured term
        # is allowed to move.
        measured = with_skill.get("measured", with_skill.get("score", 0.0))
        adjusted = base_evidence * measured + (1.0 - base_evidence) * _PRIOR
        score = _hirability(adjusted, exp_fit, edu_fit, weights, track)
        gained = with_skill.get("matched_count", 0) - base_matched

        if gained <= 0 or score - base_score < 0.05:
            # Either the target never mentions this skill, or the movement is
            # below what the model can resolve. Report no movement rather than
            # a fraction of a point that would read as "learning this makes you
            # worse" — not a claim the model is entitled to make.
            results.append({
                "skill": skill,
                "score": base_score,
                "delta": 0.0,
                "skill_strength": round(base.get("score", 0.0) * 100),
                "also_unlocks": 0,
                "measurable": False,
            })
            continue

        results.append({
            "skill": skill,
            "score": score,
            "delta": round(score - base_score, 1),
            "skill_strength": round(with_skill.get("score", 0.0) * 100),
            "also_unlocks": max(0, gained - 1),
            "measurable": True,
        })

    results.sort(key=lambda r: r["delta"], reverse=True)
    return {
        "baseline_score": base_score,
        "baseline_skill_strength": round(base.get("score", 0.0) * 100),
        "skills": results,
    }
