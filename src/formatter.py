"""
Human-readable report formatter for the resume analysis pipeline.
Converts the build_report() dict into a clean, structured text report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_WIDTH = 60


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> str:
    return f"\n{'─' * _WIDTH}\n  {title}\n{'─' * _WIDTH}"


def _score_line(label: str, score: float, suffix: str = "") -> str:
    return f"  {label:<24} {score:>5.1f} / 100  {_bar(score)}{suffix}"


def _level_badge(level: str) -> str:
    return {"Strong": "●●●●●", "Competitive": "●●●●○",
            "Developing": "●●●○○", "Entry-level": "●●○○○",
            "Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(level, level)


def format_report(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    s = r.get("summary", {})
    comp = r.get("competitiveness", {})
    ai_disp = r.get("ai_displacement_exposure") or {}
    top_matches = r.get("top_job_matches", [])
    gaps = r.get("skill_gaps", {})
    recs = r.get("recommendations", [])
    emerging = r.get("emerging_ai_recommendations", [])
    llm_sugg = r.get("llm_skill_suggestions", [])
    skills_block = r.get("skills", {})

    # ── Header ────────────────────────────────────────────────
    lines += [
        "═" * _WIDTH,
        "  CAREER INTELLIGENCE REPORT".center(_WIDTH),
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
        lines.append(f"  {'AI Displacement Risk':<24} {_level_badge(lvl)}  {lvl}")
    lines.append(f"\n  {comp.get('explanation', '')}")

    # ── Profile ───────────────────────────────────────────────
    lines.append(_section("YOUR PROFILE"))
    edu = s.get("education_level") or "Not detected"
    exp = s.get("years_experience")
    exp_str = f"{exp} years" if exp else "Not detected"
    n_section = s.get("skills_from_section", 0)
    n_exp = s.get("skills_from_experience", 0)
    lines += [
        f"  Education    {edu}",
        f"  Experience   {exp_str}",
        f"  Skills       {s.get('skills_extracted', 0)} total  "
        f"({n_section} from resume, {n_exp} found in experience text)",
    ]

    # ── Top job match ─────────────────────────────────────────
    if top_matches:
        top = top_matches[0]
        lines.append(_section(f"TOP JOB MATCH: {top['title'].upper()}"))
        snippet = top.get("description", "")
        if snippet:
            # Word-wrap to width
            words = snippet.replace("...", "").split()
            row, buf = [], []
            for w in words:
                buf.append(w)
                if len(" ".join(buf)) > _WIDTH - 4:
                    row.append("  " + " ".join(buf[:-1]))
                    buf = [w]
            if buf:
                row.append("  " + " ".join(buf))
            lines += row[:3]   # max 3 lines of description

        if len(top_matches) > 1:
            lines.append("\n  Other strong fits:")
            for m in top_matches[1:5]:
                score = m.get("blended_score", 0)
                lines.append(f"    · {m['title']:<35} match: {score:.2f}")

    # ── Strengths ─────────────────────────────────────────────
    strengths = gaps.get("strengths", [])
    if strengths:
        lines.append(_section("STRENGTHS"))
        lines.append("  Skills you have that align with your top match:")
        names = [g["skill"] if isinstance(g, dict) else str(g) for g in strengths[:12]]
        lines.append("  " + "  ·  ".join(names))

    unmatched = skills_block.get("unmatched", [])
    if unmatched:
        lines.append(f"\n  Skills not yet in the dataset (still valuable):")
        lines.append("  " + "  ·  ".join(unmatched[:8]))

    # ── Skill gaps ────────────────────────────────────────────
    top_title = top_matches[0]["title"] if top_matches else "your target role"
    gap_list = gaps.get("gaps", [])
    if gap_list or recs:
        lines.append(_section(f"SKILL GAPS  (vs. {top_title})"))
        lines.append("  Prioritised skills to consider adding:\n")
        for i, rec in enumerate(recs, 1):
            hot = "★ " if rec.get("is_hot") else "  "
            lines.append(f"  {i}. {hot}{rec['skill']:<30} {rec['effort_label']}")
            lines.append(f"     {rec['rationale']}")

    # ── Emerging AI ───────────────────────────────────────────
    if emerging:
        lines.append(_section("EMERGING AI / ML SKILLS"))
        lines.append("  High-demand skills for tech roles not yet in ONET:\n")
        for i, rec in enumerate(emerging, 1):
            lines.append(f"  {i}. ★ {rec['skill']:<30} {rec['effort_label']}")
            lines.append(f"     {rec['rationale']}")

    # ── LLM suggestions ───────────────────────────────────────
    if llm_sugg:
        lines.append(_section("AI ADVISOR SUGGESTIONS  (beyond standard datasets)"))
        lines.append("  Skills gaining rapid adoption — sourced from Groq / Llama 3.3:\n")
        for i, s_item in enumerate(llm_sugg, 1):
            lines.append(f"  {i}. ★ {s_item['skill']:<30} {s_item['time_to_learn']}")
            lines.append(f"     {s_item['reason']}")

    # ── AI displacement ───────────────────────────────────────
    if ai_disp:
        lines.append(_section("AI DISPLACEMENT EXPOSURE"))
        lvl = ai_disp.get("level", "")
        score = ai_disp.get("score", 0)
        lines += [
            f"  Risk Level   {_level_badge(lvl)}  {lvl.upper()}  (score: {score:.2f})",
            f"\n  {ai_disp.get('explanation', '')}",
        ]
        abstract = gaps.get("abstract_skills_required", [])
        if abstract:
            lines.append("\n  Human skills that protect this role:")
            names = [a["skill"] if isinstance(a, dict) else str(a) for a in abstract[:5]]
            lines.append("  " + "  ·  ".join(names))

    # ── Footer ────────────────────────────────────────────────
    parser_used = r.get("parser", "regex")
    lines += [
        f"\n{'═' * _WIDTH}",
        f"  parser: {parser_used}".center(_WIDTH),
        "═" * _WIDTH,
    ]

    return "\n".join(lines)
