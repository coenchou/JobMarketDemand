# JobMarketDemand

Scores a resume against what a job actually requires: hirability, skill gaps
ranked by what closing them is worth, live market demand, and automation
exposure. Grounded in O*NET occupational data, BLS Employment Projections and
skill mentions harvested from live job postings.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add a GROQ_API_KEY for LLM parsing

# report in the terminal
python -m src.pipeline resumes/resume_1_data_analyst.txt

# raw JSON
python -m src.pipeline resume.txt --json --out report.json

# score against a specific occupation instead of the best match
python -m src.pipeline resume.txt --target-soc 15-2051

# score against a job posting you paste into a file
python -m src.pipeline resume.txt --job-description posting.txt

# web UI (serves index.html and the API)
python -m uvicorn src.main:app --reload    # → http://localhost:8000
```

`GROQ_API_KEY` in `.env` enables LLM resume extraction and recommendation
refinement. Without it everything still runs on the regex parser and the
dataset-derived recommendations. Responses are cached under `data/cache/llm/`,
so re-running the same resume costs no requests (`LLM_CACHE=0` disables).

## What the target is

Every number is relative to one target, chosen three ways:

| source | how |
|--------|-----|
| `auto` | the shortlisted role the candidate is *most hirable* for, not the closest tool match — a role only qualifies if the résumé genuinely fits it, or the pick drifts to whichever occupation has the lowest entry bar |
| `override` | the user picked a different occupation from the ranked list |
| `posting` | a pasted job description, scored against the skills it names |

## Scoring

The weights move with career stage. A new graduate should not be scored mostly
on experience they have not had time to accumulate, and a senior engineer's
degree should not still carry a quarter of the verdict — a credential is a
signal that decays as an employer can observe actual output.

| stage | years | skill | experience | education | track record |
|-------|-------|-------|-----------|-----------|--------------|
| entry | 0–1 | 0.45 | 0.15 | **0.35** | 0.05 |
| early | 2–4 | 0.45 | 0.28 | 0.17 | 0.10 |
| mid | 5–9 | 0.42 | 0.33 | 0.10 | 0.15 |
| senior | 10+ | 0.40 | **0.35** | 0.05 | **0.20** |

Track record is what the work shows rather than how long it lasted — scope,
leadership, quantified impact, outside recognition — read off the résumé's
achievements. See `src/career_stage.py`.

Skill strength is not coverage of a tool list. Every tool the target requires is
classified as **held**, **implied** (entailed by something the candidate does
have — a PyTorch user has Python), **commodity** (cheap and ubiquitous, so its
absence carries no signal and it leaves the denominator) or a real **gap**. Only
gaps can lower a score. The result is then shrunk toward an average candidate in
proportion to how much the resume documents, because a short skills section is
weak evidence rather than evidence of weakness — which is why the report shows a
range, not just a number. Track record is shrunk the same way, so a PDF that
parses into bare job headers is not scored as though it claimed no achievements.

See `src/skill_implication.py` and `src/skill_score.py` for the details.

## Refreshing market data

```bash
python -m scripts.harvest_posting_skills --pages 12
```

Walks the public job catalog and writes per-category skill demand to
`data/processed/posting_skills.csv`. Everything degrades to O*NET-only scoring
when that file is absent.

## Deploying

The API serves the frontend, so a single process is the whole app:

```bash
uvicorn src.main:app --host 0.0.0.0 --port $PORT
```

`Procfile` declares exactly that for buildpack-style hosts. The page calls its
own origin, so no API URL needs configuring; set `ALLOWED_ORIGINS` only if the
frontend is served from a different host than the API.

Nothing resume-derived is written to disk: uploads live in a per-request temp
file that is deleted in a `finally`, any stragglers from a hard kill are purged
at startup, and the LLM response cache is disabled server-side because cached
responses contain extracted resume content. The only state that persists is a
count of analyses run, served at `/stats`.

The O*NET and BLS datasets are not in the repository — they are large, licensed
and regenerable. Fetch them before first run:

```bash
python scripts/download_onet.py            # O*NET workbooks
python -m scripts.build_bls_market         # BLS employment projections
python -m scripts.harvest_posting_skills   # optional: live posting demand
```

`GET /health` reports which datasets are present, whether an LLM key is
configured, and which model is in use — point your platform's health check at
it. `GET /stats` returns the analysis count.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s testing
```

`testing/test_scoring.py` pins the scoring behaviour — implication rules, the
strict matcher, and baseline scores for three fixed profiles. It is offline and
deterministic; the calibration constants were fitted by eye, so these baselines
are the only thing standing between a refactor and a fifteen-point drift.
