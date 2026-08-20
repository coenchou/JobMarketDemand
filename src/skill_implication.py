"""
Implied-skill credit and commodity-skill neutralisation.

A resume is a signalling document, not an inventory. Candidates list what
differentiates them and silently drop what everyone already has ("Microsoft
Word") or what is obviously entailed by something they did list — nobody who
ships PyTorch models adds a separate bullet for Python. Scoring the literal
presence of occupation tool names therefore punishes omissions that carry no
information, and since cheap skills earn no wage premium, their absence should
not move a score in either direction.

Every tool an occupation requires is classified against the candidate:

  held       their own skill list covers it (word overlap).
  implied    entailed by something they do have — a prerequisite of a listed
             advanced skill, an interchangeable substitute for one, or a close
             semantic neighbour that is no harder to learn. Credited as held.
  commodity  learnable in about two weeks, or ubiquitous office software.
             No signal either way, so it leaves the denominator rather than
             counting against the candidate.
  gap        a genuine, non-trivial missing skill.

Only `gap` tools may lower a score or surface as something to learn.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.nlp_skills import _embed, _norm, batch_difficulty_months

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

COMMODITY_MONTHS = 1.0       # <= ~2 weeks to working proficiency = no signal
_NONTRIVIAL_MONTHS = 2.0     # candidate needs one skill this hard before we
                             # forgive missing basics ("a very advanced skill
                             # or something related is there")
_SIM_IMPLIED = 0.62          # embedding proximity that counts as entailment
_SIM_SAME_TOOL = 0.80        # at this range it is the same tool under another
                             # spelling ("Hugging Face" / "HuggingFace
                             # Transformers"), so the difficulty guard is moot
_SLACK_MONTHS = 1.0          # an implied tool may be marginally harder than
                             # the skill that vouches for it

# Ubiquitous tools whose absence from a resume says nothing at all. Difficulty
# already catches most of these; these are the ones O*NET names in ways the
# difficulty table can't resolve ("Microsoft Office software").
_UBIQUITOUS: Tuple[str, ...] = (
    "microsoft office", "office suite", "microsoft word", "microsoft excel",
    "microsoft powerpoint", "microsoft outlook", "microsoft onenote",
    "microsoft teams", "google docs", "google sheets", "google slides",
    "google workspace", "email software", "electronic mail software",
    "web browser", "internet browser", "spreadsheet software",
    "word processing software", "presentation software", "calendar software",
    "adobe acrobat", "zoom", "slack", "webex", "skype",
)

# Prerequisites: holding the key implies working familiarity with the values.
# Deliberately conservative — only genuine "you cannot do X without Y" links.
_PREREQUISITES: Dict[str, Tuple[str, ...]] = {
    "pytorch": ("python", "numpy", "linux", "machine learning"),
    "tensorflow": ("python", "numpy", "linux", "machine learning"),
    "keras": ("python", "numpy", "machine learning"),
    "scikit learn": ("python", "numpy", "pandas", "machine learning"),
    "xgboost": ("python", "pandas", "machine learning"),
    "lightgbm": ("python", "pandas", "machine learning"),
    "pandas": ("python",),
    "numpy": ("python",),
    "matplotlib": ("python",),
    "seaborn": ("python", "matplotlib"),
    "django": ("python", "html", "sql"),
    "flask": ("python", "html"),
    "fastapi": ("python",),
    "machine learning": ("python", "statistics"),
    "deep learning": ("python", "machine learning", "linux"),
    "natural language processing": ("python", "machine learning"),
    "nlp": ("python", "machine learning"),
    "computer vision": ("python", "machine learning"),
    "mlops": ("python", "docker", "git", "linux"),
    "kubeflow": ("kubernetes", "docker", "python"),
    "mlflow": ("python",),
    "hugging face transformers": ("python", "pytorch"),
    "kubernetes": ("docker", "linux", "bash", "yaml"),
    "docker": ("linux", "bash"),
    "terraform": ("linux", "yaml"),
    "ansible": ("linux", "yaml", "bash"),
    "jenkins": ("bash", "linux", "git"),
    "devops": ("linux", "bash", "git", "docker"),
    "apache airflow": ("python", "sql", "bash"),
    "apache spark": ("sql", "linux"),
    "apache flink": ("java", "sql", "linux"),
    "kafka": ("linux", "java"),
    "dbt": ("sql", "git"),
    "snowflake": ("sql",),
    "bigquery": ("sql",),
    "redshift": ("sql",),
    "postgresql": ("sql",),
    "mysql": ("sql",),
    "sqlite": ("sql",),
    "tableau": ("microsoft excel", "sql"),
    "power bi": ("microsoft excel", "sql"),
    "looker": ("sql",),
    "react": ("javascript", "html", "css"),
    "vue": ("javascript", "html", "css"),
    "angular": ("javascript", "typescript", "html", "css"),
    "typescript": ("javascript",),
    "node js": ("javascript",),
    "github": ("git",),
    "gitlab": ("git",),
    "bitbucket": ("git",),
    "aws sagemaker": ("aws", "python"),
    "amazon web services": ("linux",),
    "data engineering": ("sql", "python", "linux"),
    "statistical analysis": ("statistics",),
    "hypothesis testing": ("statistics",),
    "scala": ("java",),
}

# Interchangeable substitutes: holding one member makes the others a matter of
# weeks, so a resume naming only one is not evidence of a gap.
_SUBSTITUTES: Tuple[Tuple[str, ...], ...] = (
    ("git", "github", "gitlab", "bitbucket", "subversion", "apache subversion"),
    ("aws", "amazon web services", "azure", "microsoft azure", "google cloud",
     "google cloud platform", "gcp"),
    ("postgresql", "mysql", "sqlite", "mariadb", "oracle database",
     "microsoft sql server", "microsoft access"),
    ("snowflake", "redshift", "bigquery", "databricks"),
    ("tableau", "power bi", "looker", "qlik", "metabase", "microstrategy"),
    ("pytorch", "tensorflow", "keras"),
    ("matplotlib", "seaborn", "plotly", "ggplot", "bokeh"),
    ("jira", "trello", "asana", "microsoft project", "monday"),
    ("jenkins", "circleci", "github actions", "gitlab ci", "travis ci",
     "teamcity", "bamboo"),
    ("docker", "podman", "containerd"),
    ("terraform", "ansible", "puppet", "chef", "cloudformation"),
    ("prometheus", "grafana", "datadog", "new relic", "splunk", "nagios"),
    ("elasticsearch", "solr", "opensearch"),
    ("kafka", "rabbitmq", "activemq", "pulsar"),
    ("linux", "unix", "ubuntu", "red hat enterprise linux", "centos", "debian"),
    ("slack", "microsoft teams", "zoom", "webex", "skype"),
    ("microsoft excel", "google sheets"),
    ("r", "stata", "sas", "spss", "matlab"),
    ("flask", "fastapi", "django"),
    ("react", "vue", "angular", "svelte"),
)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_MAX_VENDOR_TOKENS = 2  # "Terraform" still names the same thing as
                        # "IBM Terraform"; "Java" does not name "Oracle Java 2
                        # Platform Enterprise Edition J2EE"


def _tokens(text: str) -> FrozenSet[str]:
    return frozenset(_norm(text).split())


def _covers(skill: FrozenSet[str], phrase: FrozenSet[str]) -> bool:
    """
    True when a candidate skill contains a curated phrase outright. Deliberately
    one-directional: "AWS SageMaker" covers "aws", but plain "GitHub" must not
    count as "GitHub Actions".
    """
    return bool(phrase) and phrase <= skill


def _names_same_tool(phrase: FrozenSet[str], tool: FrozenSet[str]) -> bool:
    """
    True when a curated phrase and an O*NET tool name refer to the same thing,
    allowing for O*NET's vendor prefixes and "software" suffixes.
    """
    if not phrase or not tool:
        return False
    if tool <= phrase:
        return True
    return phrase <= tool and len(tool - phrase) <= _MAX_VENDOR_TOKENS


def _matches_any(tool: FrozenSet[str], phrases: Iterable[str]) -> bool:
    return any(_names_same_tool(_tokens(p), tool) for p in phrases)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _entailed_by(user_skills: List[str]) -> Dict[str, str]:
    """Skill phrases the candidate's listed skills presuppose → the voucher."""
    entailed: Dict[str, str] = {}
    for skill in user_skills:
        st = _tokens(skill)
        for key, prereqs in _PREREQUISITES.items():
            if _covers(st, _tokens(key)):
                for p in prereqs:
                    entailed.setdefault(p, skill)
    return entailed


def _substituted_by(user_skills: List[str]) -> Dict[str, str]:
    """Substitute-family members the candidate already holds a member of."""
    covered: Dict[str, str] = {}
    for family in _SUBSTITUTES:
        held = next(
            (s for s in user_skills
             if any(_covers(_tokens(s), _tokens(m)) for m in family)),
            None,
        )
        if held:
            for m in family:
                covered.setdefault(m, held)
    return covered


def _vouching_skill(tool: FrozenSet[str], phrases: Dict[str, str]) -> Optional[str]:
    for phrase, skill in phrases.items():
        if _names_same_tool(_tokens(phrase), tool):
            return skill
    return None


def _held_tool_norms(user_skills: List[str]) -> Set[str]:
    """Tools the candidate literally claims (precision-first matching)."""
    from src.skill_matcher import match_skill_to_tools_strict

    held: Set[str] = set()
    for skill in user_skills:
        held |= match_skill_to_tools_strict(skill)
    return held


def _max_similarity(
    tool_names: List[str], user_skills: List[str]
) -> Tuple[np.ndarray, List[Optional[str]]]:
    """
    Per tool: its highest cosine similarity to any user skill, and which skill
    that was. Zeros and Nones when embeddings are unavailable.
    """
    zeros = np.zeros(len(tool_names), dtype=np.float32)
    if not tool_names or not user_skills:
        return zeros, [None] * len(tool_names)
    embs = _embed(tool_names + user_skills)
    if embs is None:
        return zeros, [None] * len(tool_names)
    tool_embs, user_embs = embs[: len(tool_names)], embs[len(tool_names):]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = np.nan_to_num(tool_embs @ user_embs.T, nan=0.0, posinf=0.0, neginf=0.0)
    best = sims.argmax(axis=1)
    return sims.max(axis=1), [user_skills[i] for i in best]


def classify_occupation_tools(
    tool_names: List[str], user_skills: List[str]
) -> Dict[str, Dict]:
    """
    Classify each occupation tool against the candidate.

    Returns tool_name -> {status, difficulty_months, relevance, anchor} where
    status is one of "held" | "implied" | "commodity" | "gap", relevance is the
    tool's semantic proximity to the candidate's skill set (0-1, useful for
    ranking real gaps), and anchor names the skill that vouched for an implied
    tool. Batched throughout: one embedding pass and one difficulty pass for the
    whole occupation.
    """
    tools = list(dict.fromkeys(tool_names))
    if not tools:
        return {}

    user_skills = [s for s in dict.fromkeys(user_skills) if s and s.strip()]
    held_norms = _held_tool_norms(user_skills)
    entailed = _entailed_by(user_skills)
    substituted = _substituted_by(user_skills)

    tool_months = batch_difficulty_months(tools)
    user_months = batch_difficulty_months(user_skills) if user_skills else [0.0]
    has_nontrivial = max(user_months) >= _NONTRIVIAL_MONTHS
    months_by_skill = dict(zip(user_skills, user_months))

    sims, anchors = _max_similarity(tools, user_skills)

    out: Dict[str, Dict] = {}
    for i, tool in enumerate(tools):
        tt = _tokens(tool)
        months = tool_months[i]
        sim = float(sims[i])
        nearest = anchors[i]
        nearest_months = months_by_skill.get(nearest or "", 0.0)
        voucher = _vouching_skill(tt, entailed) or _vouching_skill(tt, substituted)
        anchor: Optional[str] = None

        if _norm(tool) in held_norms:
            status = "held"
        elif voucher:
            status, anchor = "implied", voucher
        elif nearest is not None and (
            sim >= _SIM_SAME_TOOL
            or (sim >= _SIM_IMPLIED and months <= nearest_months + _SLACK_MONTHS)
        ):
            status, anchor = "implied", nearest
        elif has_nontrivial and (
            months <= COMMODITY_MONTHS or _matches_any(tt, _UBIQUITOUS)
        ):
            status = "commodity"
        else:
            status = "gap"

        out[tool] = {
            "status": status,
            "difficulty_months": months,
            "relevance": round(sim, 3),
            "anchor": anchor,
        }
    return out


def gap_priority(
    demand_score: float, relevance: float, market_share: float = 0.0
) -> float:
    """
    Rank real gaps by how much closing them would matter to *this* candidate:
    market demand, scaled by how close the skill sits to the work they already
    do. Keeps "Snowflake" above "PTC Creo Parametric" for an ML engineer.

    `demand_score` is O*NET's hot/in-demand weighting (0-3); `market_share` is
    the fraction of live postings that ask for the skill. Where postings
    disagree with the survey the higher signal wins — O*NET under-reports new
    tooling far more often than it over-reports it.
    """
    demand_norm = max(
        min(1.0, demand_score / 3.0),
        min(1.0, market_share / 0.4),
    )
    return round((0.35 + 0.65 * demand_norm) * (0.3 + 0.7 * relevance), 4)
