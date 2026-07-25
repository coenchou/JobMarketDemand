"""
NLP-powered skill analysis: feasibility scoring, complementarity detection,
and AI/automation displacement estimation.

Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings.
Falls back to keyword-based heuristics if the model is unavailable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import zlib
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

# SOC major groups considered technical fields (computer/mathematical,
# architecture/engineering) — used to decide whether AI/ML skill suggestions
# are relevant to the user's top matched occupation.
_TECHNICAL_SOC_PREFIXES = ("15-", "17-")
_TECHNICAL_SOC_EXTRAS = {"11-3021"}  # Computer and Information Systems Managers


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

    # Add emerging skill difficulties to the main lookup so feasibility works
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
# Skill descriptions
#
# Each recommended skill gets a distinct description built from three parts:
#   1. what the skill is and where it sits in the labour market (curated per
#      skill where possible, category-derived otherwise)
#   2. how it relates to the specific skills already on the resume
#   3. what learning it actually involves
# Demand statements are grounded in the O*NET flags (hot technology /
# in-demand) and the number of occupations listing the tool.
# ---------------------------------------------------------------------------

# Curated notes: normalised-substring key → (what, learning, optional trend override)
# Keys are matched longest-first against the normalised tool name.
_SKILL_NOTES: Dict[str, Tuple[str, str, Optional[str]]] = {
    "structured query language": (
        "SQL remains the single most common data requirement in job postings; nearly every role that touches stored data assumes it.",
        "The core select–join–group pattern takes two or three weeks of practice; window functions and query optimisation add another month.",
        None),
    "microsoft sql server": (
        "SQL Server anchors data work inside Microsoft-stack enterprises — a large and stable segment of the market.",
        "If you already write SQL, the T-SQL specifics and tooling take a few weeks; administration is a separate track.",
        None),
    "postgresql": (
        "PostgreSQL is the most widely deployed open-source relational database and the default choice for new backend systems.",
        "The query layer transfers from any SQL dialect within days; indexing, permissions, and performance work are the longer arc.",
        None),
    "mysql": (
        "MySQL powers a large share of existing web backends, so it appears constantly in full-stack and maintenance listings.",
        "Query skills carry over from other SQL dialects in days; replication and tuning are the deeper layer.",
        None),
    "mongodb": (
        "MongoDB is the most used document database, common in product stacks that outgrew relational-only storage.",
        "CRUD operations take days; designing schemas without joins is the actual conceptual shift.",
        None),
    "amazon web services": (
        "AWS is the largest cloud platform, and cloud experience is now assumed in most engineering and data infrastructure roles.",
        "About a month of hands-on use gets you productive with the core services (EC2, S3, IAM, Lambda); breadth accumulates over years.",
        None),
    "microsoft azure": (
        "Azure is the default cloud inside Microsoft-centric enterprises and second in overall market share, with strong corporate IT demand.",
        "Core services take a few weeks if you have touched any cloud; the certification paths give the learning useful structure.",
        None),
    "oracle java": (
        "Java still underpins most large enterprise backends and remains among the top languages by total job volume.",
        "It rewards patience — expect several months to internalise the type system, JVM ecosystem, and standard tooling.",
        None),
    "javascript": (
        "JavaScript is unavoidable in web-facing work: every front end and much of modern tooling runs on it.",
        "Working proficiency takes one to two months; asynchronous patterns and ecosystem churn are what extend the curve.",
        None),
    "typescript": (
        "TypeScript has overtaken plain JavaScript in new professional front-end work, and postings increasingly name it directly.",
        "For someone with JavaScript it is a few weeks of learning the type system; without it, learn JavaScript first.",
        None),
    "node.js": (
        "Node.js runs a large share of modern web APIs and is the standard way to take JavaScript server-side.",
        "With JavaScript in hand, a few weeks of building services covers the core; streams and performance tuning come later.",
        None),
    "react": (
        "React is the most demanded front-end framework by a wide margin and a fixture in full-stack listings.",
        "Component basics take weeks; state management and performance discipline push real fluency to a few months.",
        None),
    "angular": (
        "Angular holds a durable niche in enterprise front ends, where its all-in-one structure is preferred over lighter libraries.",
        "It has a steeper initial slope than React — TypeScript, dependency injection, and RxJS arrive as a package — so plan a few months.",
        None),
    "jquery": (
        "jQuery demand has been declining for years as frameworks replaced it, but a long tail of existing sites still needs maintenance.",
        "For anyone with JavaScript it is days of learning; treat it as a compatibility skill, not an investment.",
        "declining"),
    "python": (
        "Python is the most requested programming language across data, machine learning, and automation roles.",
        "Basic scripting comes in weeks, but professional fluency — idiomatic code, packaging, testing — is a several-month build.",
        None),
    "docker": (
        "Containers are the standard unit of deployment, and Docker literacy is expected in nearly all backend and DevOps postings.",
        "A week gets you building images and running containers; networking and multi-stage builds fill out the next month.",
        None),
    "kubernetes": (
        "Kubernetes is how most organisations run containers in production, and it carries some of the strongest salary premiums in infrastructure work.",
        "This one is genuinely steep: months of hands-on cluster work, and it assumes Docker and networking fundamentals first.",
        None),
    "terraform": (
        "Terraform is the standard for infrastructure-as-code across clouds and has become a fixture in platform engineering postings.",
        "First configurations take days; state management and module design across a real environment take a couple of months.",
        None),
    "jenkins": (
        "Jenkins remains the most installed CI server; enterprise demand is stable even as newer hosted CI services grow faster.",
        "Setting up pipelines takes a week or two; its plugin ecosystem is where the depth lives.",
        "stable"),
    "git": (
        "Version control with Git is a universal expectation in software work — its absence is noticed more than its presence.",
        "Daily-driver basics take days; the mental model for branching and history takes a few weeks of team use.",
        None),
    "github": (
        "GitHub is where most collaborative software development happens; fluency with its review and CI workflows is assumed in team roles.",
        "A week of pull-request-driven work covers what employers look for.",
        None),
    "tableau": (
        "Tableau leads the dedicated BI-tool market, and analyst postings frequently name it specifically.",
        "A couple of weeks of building dashboards covers day-to-day use; calculated fields and LOD expressions are the advanced layer.",
        None),
    "power bi": (
        "Power BI ships with the Microsoft stack, which has made it the fastest-growing BI requirement in analyst roles.",
        "The visual layer takes days if you know Excel; DAX, its formula language, is the part worth deliberate study.",
        None),
    "microsoft excel": (
        "Excel remains the most used analysis tool in business and a baseline requirement well beyond finance.",
        "The basics are quick; pivot tables, lookups, and Power Query are what analyst roles actually test.",
        None),
    "apache kafka": (
        "Kafka is the standard for high-volume event streaming and appears in most data-platform and backend-at-scale listings.",
        "Producing and consuming messages takes a week or two; partitioning, delivery guarantees, and operations are the real learning.",
        None),
    "apache spark": (
        "Spark is the dominant engine for large-scale data processing and a fixture in data engineering requirements.",
        "With Python or SQL, the DataFrame API takes a few weeks; debugging slow jobs — partitioning, shuffles — takes months.",
        None),
    "apache hadoop": (
        "Hadoop demand has been declining as workloads move to cloud warehouses, but it persists in large legacy data platforms.",
        "Worth learning only if you are targeting those environments; the distributed-computing concepts transfer either way.",
        "declining"),
    "apache airflow": (
        "Airflow is the most common orchestrator for scheduled data pipelines and shows up in a majority of data engineering postings.",
        "Writing first DAGs takes days with Python; operating it reliably — retries, backfills, deployment — is the substantive part.",
        None),
    "salesforce": (
        "Salesforce is the largest CRM platform, with an entire job category of administrators and developers built around it.",
        "The admin path is a few months of guided study; platform development adds Apex and its ecosystem on top.",
        None),
    "sap": (
        "SAP runs core operations at a large share of the world's biggest companies, and SAP-adjacent skills stay durable and well paid.",
        "It is a deep ecosystem — count on months, usually tied to a specific module rather than the platform in general.",
        None),
    "linux": (
        "Linux underlies nearly all server infrastructure; shell fluency is a quiet prerequisite for backend, data, and DevOps work.",
        "Daily comfort takes a few weeks in a terminal; system administration is a longer, mostly on-the-job accumulation.",
        None),
    "bash": (
        "Shell scripting is the connective tissue of server automation — a small skill with outsized daily leverage.",
        "A week or two covers variables, pipes, and control flow; the habit of reaching for it is the real acquisition.",
        None),
    "powershell": (
        "PowerShell is the automation language of Windows environments, with steady demand in corporate IT and DevOps.",
        "With any scripting background, a few weeks; its object pipeline is the concept that differs from Unix shells.",
        "stable"),
    "c++": (
        "C++ dominates performance-critical domains — trading systems, games, embedded — which pay well and hire steadily.",
        "One of the longest learning curves in mainstream software: expect a year to genuine competence, less coming from C or Rust.",
        None),
    "c#": (
        "C# anchors the Microsoft development ecosystem, with steady demand in enterprise software and in games via Unity.",
        "Coming from Java the transition takes weeks; from scratch, a few months for the language plus the .NET runtime.",
        None),
    "php": (
        "PHP still runs a large fraction of the web — WordPress alone guarantees demand — mostly in maintenance and agency work.",
        "Quick to start, but employers want modern practice: Composer, a framework, and testing habits.",
        "stable"),
    "django": (
        "Django is Python's dominant full-featured web framework, common in startups and data-heavy products.",
        "With Python, a few weeks to ship a first app; its ORM and conventions reward a couple of months of practice.",
        None),
    "flask": (
        "Flask is the standard lightweight Python web framework — often the fastest route to putting a model or API into production.",
        "Days to a first service if you know Python; the skill scales with whatever you attach to it.",
        None),
    "matlab": (
        "MATLAB holds steady demand in engineering, signal processing, and research roles, though open-source tools are eroding it.",
        "With any programming background the language is quick; the value is in its domain toolboxes.",
        "stable"),
    "sas": (
        "SAS remains entrenched in pharma, banking, and government analytics, though the wider market keeps shifting to Python and R.",
        "The language basics take a few weeks; most postings care about specific procedures and industry conventions.",
        "stable"),
    "scala": (
        "Scala's demand concentrates around Spark and a handful of large backends — a narrowing niche that still pays well.",
        "Plan several months: functional programming concepts plus the JVM ecosystem make it one of the harder mainstream languages.",
        "declining"),
    "tensorflow": (
        "TensorFlow demand has shifted toward model serving and mobile deployment as PyTorch took over research and much of training.",
        "With ML fundamentals in place, a couple of months; without them, the framework is not the hard part.",
        "stable"),
    "pytorch": (
        "PyTorch is now the default framework for deep learning in both research and industry, and postings reflect that shift.",
        "The API takes weeks if you know Python and NumPy; the months go into the modelling craft around it.",
        None),
    "redis": (
        "Redis is the standard in-memory store for caching and queues, present in most production web architectures.",
        "Days to use as a cache; the interesting learning is data-structure modelling and eviction behaviour.",
        None),
    "snowflake": (
        "Snowflake has become the leading cloud data warehouse and appears in a growing share of data engineering postings.",
        "SQL carries you most of the way within weeks; cost management and warehouse sizing are the platform-specific craft.",
        None),
    "atlassian jira": (
        "Jira is the default work-tracking system in software organisations; fluency is assumed rather than taught.",
        "Days as a user; administering workflows is a separate, shallower skill.",
        "stable"),
    "r ": (
        "R keeps a strong position in statistics-heavy roles — biostatistics, research, parts of finance — where its modelling libraries lead.",
        "A month or two via the tidyverse covers data work; the statistical depth is what you build over time.",
        None),
    # Emerging AI/ML skills (not in O*NET's technology data)
    "prompt engineering": (
        "Structuring inputs that get reliable output from language models has become a baseline expectation in AI-adjacent engineering roles.",
        "There is little theory to absorb — competence comes from a week or two of deliberate experimentation against real tasks.",
        "growing"),
    "openai api": (
        "Most production LLM features are still built directly against the OpenAI API, so application-engineer postings increasingly list it alongside core web skills.",
        "If you can call a REST API you can use it the same day; the learning is in handling streaming, token limits, and failure modes.",
        "growing"),
    "anthropic claude api": (
        "Teams increasingly build against multiple LLM providers, and Claude is the second-most adopted hosted model in production systems.",
        "The interface mirrors other LLM APIs — a few days of building, with most effort going into prompt structure and tool use.",
        "growing"),
    "langchain": (
        "LangChain is the most widely used orchestration layer for chaining LLM calls, tools, and data sources in application code.",
        "Expect a few weeks: the API surface is large and changes quickly, so the work is learning its abstractions and when not to use them.",
        "growing"),
    "llamaindex": (
        "LlamaIndex specialises in connecting LLMs to private data — indexing, chunking, retrieval — a narrower, deeper alternative to general orchestration frameworks.",
        "A few weeks of building against your own documents teaches most of it; the depth is in tuning retrieval quality.",
        "growing"),
    "pinecone": (
        "Pinecone is a managed vector database for storing embeddings behind semantic search, and a common line item in RAG job descriptions.",
        "The API takes days; understanding embedding models, distance metrics, and index configuration is the real content.",
        "growing"),
    "chroma": (
        "Chroma is the most common open-source vector store for local and small-scale retrieval work — a low-friction entry into embedding-based search.",
        "It runs locally with one install, so a weekend project covers the essentials; production concerns come later.",
        "growing"),
    "weaviate": (
        "Weaviate is an open-source vector database chosen where teams want hybrid keyword-plus-semantic search or self-hosted deployment.",
        "Plan a few weeks — schema and hybrid-search configuration are more involved than lighter vector stores.",
        "growing"),
    "retrieval-augmented generation": (
        "Retrieval-augmented generation is the standard architecture for grounding LLM output in private data, and it anchors a large share of current AI engineering work.",
        "A first pipeline takes a couple of weeks; making retrieval genuinely good — chunking, evaluation, reranking — is a multi-month practice.",
        "growing"),
    "hugging face transformers": (
        "The transformers library is the de facto interface to open-weight models and appears in most ML postings that go beyond hosted APIs.",
        "Loading and running models takes days; fine-tuning, quantisation, and efficient serving are where the months go.",
        "growing"),
    "hugging face hub": (
        "The Hugging Face Hub is where open models and datasets are published; fluency with it signals you can work outside closed APIs.",
        "Mostly conventions and tooling — a week of publishing and pulling artifacts covers it.",
        "growing"),
    "llm fine-tuning": (
        "Adapting models to domain data separates teams that ship differentiated AI features from those wrapping an API, and it commands a premium in ML hiring.",
        "A genuine multi-month skill: dataset construction, training infrastructure, and evaluation each have their own curve.",
        "growing"),
    "lora": (
        "Low-rank adaptation makes fine-tuning feasible on modest hardware and is the technique most teams actually use to customise open models.",
        "With Python and some ML background, a few months of hands-on training runs builds real competence.",
        "growing"),
    "ai agent development": (
        "Systems where a model plans, calls tools, and iterates are the current frontier of applied AI work, and mentions in postings have grown steadily through 2025.",
        "A couple of months of building: the concepts are simple, but reliability engineering — guardrails, evaluation, recovery — is the actual skill.",
        "growing"),
    "mlflow": (
        "MLflow is the most common open-source tool for tracking experiments and managing model lifecycle — a standing requirement in MLOps-flavoured roles.",
        "Instrumenting a first project takes days; the value compounds as you adopt its registry and deployment conventions.",
        "growing"),
    "weights & biases": (
        "Weights & Biases is the dominant hosted experiment-tracking platform in research-adjacent ML teams and appears by name in many postings.",
        "A few days to instrument training runs; dashboarding and hyperparameter sweeps are incremental from there.",
        "growing"),
}

# Category-level fallbacks keyed by substrings of the O*NET element name.
_CATEGORY_NOTES: List[Tuple[str, str, str]] = [
    ("database management",
     "{s} is a database platform; roles that manage stored data at scale list it among their core systems.",
     "With existing SQL experience the syntax transfers quickly; administration and performance work are the longer investment."),
    ("data base user interface",
     "{s} is a data-access and query tool used daily in roles that work directly with stored records.",
     "Query tools reward consistent use — a few weeks of real work covers what postings expect."),
    ("object or component oriented",
     "{s} is a general-purpose programming language; demand concentrates in teams whose existing systems are built on it.",
     "Budget a few months — professional use of any language means its tooling and ecosystem, not just syntax."),
    ("web platform development",
     "{s} is part of the web development stack, where job volume is broad and continuous.",
     "A few weeks applied to an existing web project is the fastest way in."),
    ("business intelligence",
     "{s} is a reporting and BI tool; analyst listings name tools specifically, so hands-on familiarity matters more than theory.",
     "Building a few real dashboards over two or three weeks covers what interviews test."),
    ("analytical or scientific",
     "{s} is an analytical package used in research and engineering settings; demand is steady within its domain rather than broad.",
     "The software is learnable in weeks; the domain methods it implements are the real skill."),
    ("operating system",
     "{s} is systems-level infrastructure; comfort with it signals you can work below the application layer.",
     "Mostly hands-on learning — weeks to be functional, longer to administer."),
    ("enterprise resource planning",
     "{s} is an ERP platform; enterprises pay a persistent premium for people who can work inside these systems.",
     "Typically learned module by module over months, usually on the job or through vendor training."),
    ("customer relationship management",
     "{s} is a CRM platform; demand comes from sales-adjacent and operations roles across most industries.",
     "User-level fluency takes weeks; administration is a recognised specialisation of its own."),
    ("cloud",
     "{s} is cloud infrastructure tooling, and cloud fluency is now assumed across engineering and data roles.",
     "Hands-on labs beat reading here — a few weeks of building something real."),
    ("project management",
     "{s} is project-tracking software; familiarity is expected in most collaborative technical work.",
     "A few days of use covers it; it rarely needs deliberate study."),
    ("file versioning",
     "{s} is version-control tooling, a universal expectation in software work.",
     "Days to adopt; a few weeks to internalise team workflows."),
    ("network",
     "{s} relates to network infrastructure, where demand is stable and certification-driven.",
     "Structured study helps here — certification tracks map the territory well."),
    ("spreadsheet",
     "{s} is spreadsheet software, still the most common analysis surface in business roles.",
     "Quick to pick up; pivot tables and lookup functions are what employers actually check."),
    ("presentation",
     "{s} is presentation software — a baseline expectation rather than a differentiator.",
     "A few days of use covers what roles require."),
    ("word processing",
     "{s} is document software; listing it matters less than the work products it produces.",
     "No meaningful learning curve for anyone in office work."),
    ("graphic",
     "{s} is a design and graphics tool; demand sits in creative and marketing-adjacent roles.",
     "A few weeks to basic production work; a portfolio matters more than time served."),
    ("development environment",
     "{s} is developer tooling; it appears in postings as an environment expectation rather than a skill in itself.",
     "Picked up in days alongside real work."),
    ("web page creation",
     "{s} is a site-building platform; a large share of small-business and agency web work runs through tools like it.",
     "Productive within days; the depth is in theming, plugins, and deployment rather than the tool itself."),
    ("content workflow",
     "{s} is content-management tooling, common in publishing and marketing-adjacent technical roles.",
     "A few days of hands-on use covers what postings expect."),
]


def _lookup_skill_notes(skill: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Find curated notes for a skill by longest-substring match."""
    raw = skill.lower()
    norm_text = _norm(skill)
    norm_words = set(norm_text.split())
    best, best_len = None, 0
    for key, notes in _SKILL_NOTES.items():
        k = key.strip().lower()
        if any(ch in k for ch in "+#"):
            hit = k in raw          # symbols are lost by normalisation
        elif len(k) <= 3:
            hit = k in norm_words   # short names must match a whole word
        else:
            hit = _norm(k) in norm_text
        if hit and len(k) > best_len:
            best, best_len = notes, len(k)
    return best


def _closest_user_skill(gap_skill: str, user_skills: List[str]) -> Tuple[Optional[str], float]:
    """Return the user's semantically closest skill to a gap skill, with similarity."""
    if not user_skills:
        return None, 0.0
    embs = _embed([gap_skill] + user_skills)
    if embs is not None:
        gap_emb, user_embs = embs[0], embs[1:]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.nan_to_num(user_embs @ gap_emb, nan=0.0, posinf=0.0, neginf=0.0)
        idx = int(np.argmax(sims))
        return user_skills[idx], float(np.clip(sims[idx], 0.0, 1.0))
    # Keyword fallback
    gap_words = set(_norm(gap_skill).split())
    best_skill, best = None, 0.0
    for s in user_skills:
        s_words = set(_norm(s).split())
        union = gap_words | s_words
        if union:
            score = len(gap_words & s_words) / len(union)
            if score > best:
                best_skill, best = s, score
    return best_skill, best


def _fallback_learning_note(months: float) -> str:
    if months <= 0.5:
        return "A few focused days of hands-on use are enough to list it credibly."
    if months <= 1.5:
        return "One to a few weeks of practical use covers working proficiency."
    if months <= 3.0:
        return "Plan one to three months of regular practice to be genuinely productive."
    if months <= 6.0:
        return "A three-to-six-month build with consistent hands-on work."
    return "A long-term investment — six months or more of sustained practice."


def _connection_note(skill: str, closest: Optional[str], sim: float) -> str:
    v = zlib.crc32(skill.encode()) % 3
    if closest and sim >= 0.60:
        return [
            f"Your {closest} experience covers much of the ground it builds on.",
            f"It sits close to {closest}, which you already use, so the on-ramp is short.",
            f"Coming from {closest}, most of the underlying concepts will already be familiar.",
        ][v]
    if closest and sim >= 0.42:
        return [
            f"There is partial overlap with your {closest} background, which speeds up the early stages.",
            f"Your work with {closest} gives you a foothold, though much of this will still be new.",
            f"It connects loosely to {closest} on your resume — enough to orient you, not enough to skip the fundamentals.",
        ][v]
    return [
        "It does not build directly on anything currently on your resume, so budget real study time.",
        "This would be new territory relative to your current skills.",
        "Nothing on your resume shortcuts this one — plan for the full learning curve.",
    ][v]


def _demand_note(skill: str, trend: str, n_occupations: int) -> str:
    v = zlib.crc32(skill.encode()) % 2
    n = n_occupations
    if trend == "growing" and n == 0:
        return "It is too new for O*NET's technology data, but its presence in postings has grown rapidly since 2023."
    if trend == "growing":
        return [
            f"O*NET flags it as a hot technology, listed by {n} occupations — demand is growing.",
            f"Demand is trending up: it carries O*NET's hot-technology flag and appears in {n} occupations' requirements.",
        ][v]
    if trend == "stable":
        return [
            f"Employers list it steadily — O*NET marks it as in-demand across {n} occupations.",
            f"O*NET shows stable demand: it is flagged in-demand and required by {n} occupations.",
        ][v]
    if trend == "declining":
        return f"O*NET still lists it across {n} occupations, but the market for it is contracting."
    return f"It carries no growth flag in the current O*NET cycle — steady rather than expanding demand, listed by {n} occupations."


def build_skill_description(
    skill: str,
    *,
    category: str = "",
    is_hot: bool = False,
    in_demand: bool = False,
    closest: Optional[str] = None,
    closest_sim: float = 0.0,
    months: float = 4.0,
    n_occupations: int = 0,
) -> Tuple[str, str]:
    """
    Compose a unique description for a recommended skill.
    Returns (description, demand_trend) where demand_trend is
    "growing" | "stable" | "steady" | "declining".
    """
    notes = _lookup_skill_notes(skill)

    trend = "growing" if is_hot else ("stable" if in_demand else "steady")
    if notes and notes[2]:
        trend = notes[2]

    parts: List[str] = []
    if notes:
        what, learn = notes[0], notes[1]
        parts = [what, _connection_note(skill, closest, closest_sim), learn]
    else:
        cat = category.lower()
        cat_note = next(
            ((w, l) for key, w, l in _CATEGORY_NOTES if key in cat), None
        )
        if cat_note:
            what = cat_note[0].format(s=skill)
            learn = cat_note[1]
        else:
            what = f"{skill} appears in the listed toolset for this occupation."
            learn = _fallback_learning_note(months)
        parts = [
            what,
            _demand_note(skill, trend, n_occupations),
            _connection_note(skill, closest, closest_sim),
            learn,
        ]

    return " ".join(parts), trend


# ---------------------------------------------------------------------------
# Skill recommendations
# ---------------------------------------------------------------------------


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
    from src.skill_matcher import count_occupations_for_tool  # avoid circular import

    scored: List[Dict] = []

    for gap in gaps:
        skill_name = gap["skill"]
        fs = score_feasibility(skill_name, user_skills, max_months=max_feasibility_months)

        # Skip skills far beyond the feasibility window (> 1.5× threshold)
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
# LLM refinement of recommendations
#
# The dataset-driven gap list is a blunt instrument: it will happily tell a
# senior engineer to "learn Git" just because they didn't spell it out. This
# layer hands the candidate's real context to the LLM to (a) drop gaps they
# obviously already have, (b) keep the genuinely useful ones, (c) add current,
# high-value skills the standard datasets don't track, and (d) write a concrete
# next-steps summary. Falls back to the dataset list if the LLM is unavailable.
# ---------------------------------------------------------------------------

# Near-duplicate skill families — collapsed so we never recommend two members
# of the same family (e.g. both "Git" and "GitHub").
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
        from groq import Groq  # type: ignore
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
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content)
        summary = str(data.get("summary", "")).strip() or _fallback_summary(role, deduped, years)
        automation_note = str(data.get("automation_note", "")).strip()

        # Re-attach dataset metadata (effort_label, demand_trend) by skill-name
        # match; compute fresh values for LLM-added skills.
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
                demand_trend = "growing"  # LLM only adds current, high-value skills
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
