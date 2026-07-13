"""
Human-readable report formatter for the resume analysis pipeline.
Converts the build_report() dict into a clean, structured text report.
"""

from __future__ import annotations

from typing import Any, Dict, List


_WIDTH = 60


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> str:
    return f"\n{'─' * _WIDTH}\n  {title}\n{'─' * _WIDTH}"


def _score_line(label: str, score: float, suffix: str = "") -> str:
    return f"  {label:<24} {score:>5.1f} / 100  {_bar(score)}{suffix}"


def _wrap(text: str, indent: str = "  ", width: int = _WIDTH - 4) -> List[str]:
    """Word-wrap a paragraph to the report width."""
    words = text.split()
    out, buf = [], []
    for w in words:
        if buf and len(" ".join(buf + [w])) > width:
            out.append(indent + " ".join(buf))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        out.append(indent + " ".join(buf))
    return out


def _rec_block(i: int, rec: Dict, index_width: int = 2) -> List[str]:
    trend = rec.get("demand_trend", "")
    meta = "  ·  ".join(
        p for p in [rec.get("effort_label", ""), f"demand {trend}" if trend else ""] if p
    )
    lines = [f"\n  {i}. {rec['skill']}    [{meta}]"]
    lines += _wrap(rec.get("rationale", ""), indent="     ")
    return lines


def format_report(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    s = r.get("summary", {})
    comp = r.get("competitiveness", {})
    ai_disp = r.get("ai_displacement_exposure") or {}
    top_matches = r.get("top_job_matches", [])
    gaps = r.get("skill_gaps", {})
    recs = r.get("recommendations", [])
    ai_skills = gaps.get("relevant_ai_skills", [])

    # ── Header ────────────────────────────────────────────────
    lines += [
        "═" * _WIDTH,
        "  RESUME ANALYSIS".center(_WIDTH),
        f"  {r.get('resume', '')}".center(_WIDTH),
        "═" * _WIDTH,
    ]

    # ── Scores ────────────────────────────────────────────────
    lines.append(_section("SCORES"))
    lines.append(_score_line("Hirability", s.get("hirability_score", 0)))
    lines.append(_score_line("Competitiveness", comp.get("score", 0),
                             f"  {comp.get('level', '')}"))
    if ai_disp:
        lvl = ai_disp.get("level", "")
        lines.append(f"  {'Automation exposure':<24} {lvl}  (index {ai_disp.get('score', 0):.2f})")
    lines.append(f"\n  {comp.get('explanation', '')}")

    # ── Profile ───────────────────────────────────────────────
    lines.append(_section("PROFILE"))
    edu = s.get("education_level") or "Not detected"
    exp = s.get("years_experience")
    exp_str = f"{exp} years" if exp else "Not detected"
    n_section = s.get("skills_from_section", 0)
    n_exp = s.get("skills_from_experience", 0)
    lines += [
        f"  Education    {edu}",
        f"  Experience   {exp_str}",
        f"  Skills       {s.get('skills_extracted', 0)} identified  "
        f"({n_section} listed, {n_exp} from experience text)",
    ]

    # ── Occupation match ──────────────────────────────────────
    if top_matches:
        top = top_matches[0]
        lines.append(_section(f"OCCUPATION MATCH: {top['title'].upper()}"))
        lines.append(f"  SOC {top['soc_code']}  ·  match index {top.get('blended_score', 0):.2f}\n")
        lines += _wrap(top.get("description", ""))

        if len(top_matches) > 1:
            lines.append("\n  Alternative matches:")
            for m in top_matches[1:5]:
                score = m.get("blended_score", 0)
                lines.append(f"    {m['title']:<38} {score:.2f}")

    # ── Strengths ─────────────────────────────────────────────
    strengths = gaps.get("strengths", [])
    if strengths:
        lines.append(_section("STRENGTHS"))
        lines.append("  Skills on this resume that the matched occupation lists:")
        names = [g["skill"] if isinstance(g, dict) else str(g) for g in strengths]
        lines += _wrap("  ·  ".join(names))

    # ── Skill gaps ────────────────────────────────────────────
    top_title = top_matches[0]["title"] if top_matches else "the matched occupation"
    if recs:
        lines.append(_section("SKILL GAPS"))
        lines.append(f"  Missing tools for {top_title}, ranked by demand and feasibility:")
        for i, rec in enumerate(recs, 1):
            lines += _rec_block(i, rec)

        if ai_skills:
            lines.append(f"\n  Relevant AI/ML skills to consider")
            lines.append(f"  (shown because the matched occupation is a technical field):")
            for i, rec in enumerate(ai_skills, 1):
                lines += _rec_block(i, rec)

    # ── Automation exposure ───────────────────────────────────
    if ai_disp:
        lines.append(_section("AUTOMATION EXPOSURE"))
        lvl = ai_disp.get("level", "")
        score = ai_disp.get("score", 0)
        lines.append(f"  Level   {lvl.upper()}  (index {score:.2f})\n")
        lines += _wrap(ai_disp.get("explanation", ""))
        abstract = gaps.get("abstract_skills_required", [])
        if abstract:
            lines.append("\n  Task characteristics that lower exposure for this role:")
            names = [a["skill"] if isinstance(a, dict) else str(a) for a in abstract[:5]]
            lines += _wrap("  ·  ".join(names))

    # ── Footer ────────────────────────────────────────────────
    lines += [
        f"\n{'═' * _WIDTH}",
        "  Sources: O*NET occupational database".center(_WIDTH),
        "═" * _WIDTH,
    ]

    return "\n".join(lines)
