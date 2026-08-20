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
    parts = [rec.get("effort_label", "")]
    if rec.get("demand_trend") == "declining":
        parts.append("declining demand")
    meta = "  ·  ".join(p for p in parts if p)
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

    lines += [
        "═" * _WIDTH,
        "  RESUME ANALYSIS".center(_WIDTH),
        f"  {r.get('resume', '')}".center(_WIDTH),
        "═" * _WIDTH,
    ]

    verdict = r.get("verdict", "")
    if verdict:
        lines.append("")
        lines += _wrap(verdict)

    summary_text = r.get("summary_text", "")
    if summary_text:
        lines.append(_section("NEXT STEPS"))
        lines += _wrap(summary_text)

    breakdown = r.get("score_breakdown", {})
    lines.append(_section("SCORES"))
    lines.append(_score_line("Hirability", s.get("hirability_score", 0),
                             f"  {s.get('hirability_level', '')}"))
    for c in breakdown.get("hirability", []):
        lines.append(f"        {c['label']:<18} {c['points']:>5} pts   "
                     f"({int(c['weight'] * 100)}% weight)")
    lines.append(_score_line("Competitiveness", comp.get("score", 0),
                             f"  {comp.get('level', '')}"))
    for c in breakdown.get("competitiveness", []):
        lines.append(f"        {c['label']:<18} {c['points']:>5} pts   "
                     f"({int(c['weight'] * 100)}% weight)")
    if ai_disp:
        lvl = ai_disp.get("level", "")
        lines.append(f"  {'Automation exposure':<24} {lvl}  (index {ai_disp.get('score', 0):.2f})")
    lines.append(f"\n  {comp.get('explanation', '')}")

    if top_matches:
        top = top_matches[0]
        lines.append(_section(f"OCCUPATION MATCH: {top['title'].upper()}"))
        lines.append(f"  SOC {top['soc_code']}  ·  match index {top.get('blended_score', 0):.2f}\n")
        lines += _wrap(top.get("description", ""))

        lm = r.get("labor_market")
        if lm:
            stats = []
            if lm.get("median_wage"):
                stats.append(f"median ${lm['median_wage']:,}")
            if lm.get("growth_pct") is not None:
                stats.append(f"{lm['growth_pct']:+.1f}% outlook ({lm['outlook']})")
            if lm.get("openings_k") is not None:
                stats.append(f"~{lm['openings_k'] * 1000:,.0f} openings/yr")
            if stats:
                lines.append("\n  " + "   ".join(stats))
            if lm.get("typical_education"):
                lines.append(f"  Typically requires {lm['typical_education'].lower()}"
                             f" · {(lm.get('typical_experience') or 'no experience').lower()}"
                             "   (BLS)")

        if len(top_matches) > 1:
            lines.append("\n  Alternative matches:")
            for m in top_matches[1:5]:
                score = m.get("blended_score", 0)
                lines.append(f"    {m['title']:<38} {score:.2f}")

    highlights = gaps.get("highlights", [])
    strengths = gaps.get("strengths", [])
    if highlights or strengths:
        lines.append(_section("STRENGTHS"))
        for h in highlights:
            lines.append(f"  • {h}")
        if strengths:
            if highlights:
                lines.append("")
            lines.append("  Skills the occupation lists that you have or clearly imply:")
            names = [g["skill"] if isinstance(g, dict) else str(g) for g in strengths]
            lines += _wrap("  ·  ".join(names))

    top_title = top_matches[0]["title"] if top_matches else "the matched occupation"
    if recs:
        lines.append(_section("SKILL GAPS"))
        lines.append(f"  Prioritized skills to add for {top_title}:")
        for i, rec in enumerate(recs, 1):
            lines += _rec_block(i, rec)

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

    lines += [
        f"\n{'═' * _WIDTH}",
        "  Sources: O*NET occupational database".center(_WIDTH),
        "═" * _WIDTH,
    ]

    return "\n".join(lines)
