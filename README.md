# JobMarketDemand

Scores a resume against what a job actually requires: hirability, skill gaps
ranked by what closing them is worth, live market demand, and automation
exposure. Grounded in O*NET occupational data, BLS Employment Projections and
skill mentions harvested from live job postings.

## Running it

```bash
pip install -r requirements.txt

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

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s testing
```

`testing/test_scoring.py` pins the scoring behaviour — implication rules, the
strict matcher, and baseline scores for three fixed profiles. It is offline and
deterministic; the calibration constants were fitted by eye, so these baselines
are the only thing standing between a refactor and a fifteen-point drift.
