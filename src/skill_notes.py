"""
Human-readable prose for a recommended skill.

Each recommendation gets a description built from three parts: what the skill
is and where it sits in the labour market (curated per skill where possible,
category-derived otherwise), how it relates to what is already on the resume,
and what learning it actually involves. Demand statements are grounded in the
O*NET hot / in-demand flags and the number of occupations listing the tool.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.embeddings import _embed, _norm

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
