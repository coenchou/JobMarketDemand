"""
Semantic occupation matching using sentence-transformers.

On first run, embeds all ~1,000 ONET occupation descriptions and caches them
to data/processed/occ_embeddings.npz so subsequent calls are instant.

The semantic score is blended with the tool-based match_score from
skill_matcher.py to produce a final `blended_score` per candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "raw" / "onet"
PROCESSED_DIR = ROOT / "data" / "processed"
EMB_FILE = PROCESSED_DIR / "occ_embeddings.npz"
META_FILE = PROCESSED_DIR / "occ_embeddings_meta.json"


def _load_occ_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "Occupation Data.xlsx")
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code_full",
        "Title": "title",
        "Description": "description",
    })
    df["soc_code"] = df["soc_code_full"].str[:7]
    return df[["soc_code", "title", "description"]].drop_duplicates(subset=["soc_code"])


def _build_and_cache_embeddings() -> Tuple[List[str], np.ndarray]:
    """
    Embed all occupation descriptions and cache to disk.
    Called automatically on first run; takes ~30–60 s on CPU.
    """
    from src.nlp_skills import _load_model

    model = _load_model()
    if model is None:
        return [], np.empty((0, 0), dtype=np.float32)

    occ = _load_occ_data()
    texts = (occ["title"] + ". " + occ["description"]).tolist()
    soc_codes = occ["soc_code"].tolist()

    print(
        f"[semantic_matcher] First-time setup: embedding {len(texts)} occupations "
        f"(~30–60 s on CPU, cached after this)…",
        flush=True,
    )

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        embs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
    embs = np.array(embs, dtype=np.float32)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(EMB_FILE), embeddings=embs)
    with open(META_FILE, "w") as f:
        json.dump(soc_codes, f)

    print("[semantic_matcher] Embeddings cached.", flush=True)
    return soc_codes, embs


def load_occ_embeddings() -> Tuple[List[str], np.ndarray]:
    """Return (soc_codes, embedding_matrix), computing cache on first call."""
    if EMB_FILE.exists() and META_FILE.exists():
        data = np.load(str(EMB_FILE))
        embs = data["embeddings"].astype(np.float32)
        with open(META_FILE) as f:
            soc_codes = json.load(f)
        return soc_codes, embs
    return _build_and_cache_embeddings()


def build_resume_query(
    experience_lines: List[str],
    skills: List[str],
    max_exp_lines: int = 20,
) -> str:
    """
    Construct a single text string that represents the resume's experience
    and skill profile for embedding.

    Using experience bullets gives the model context about *how* skills
    were applied, not just a list of tool names.
    """
    skills_text = ", ".join(skills[:30])
    exp_text = " ".join(experience_lines[:max_exp_lines])
    return f"Skills and technologies: {skills_text}. Work experience: {exp_text}"


def semantic_score_candidates(
    experience_lines: List[str],
    skills: List[str],
    candidate_soc_codes: List[str],
) -> Dict[str, float]:
    """
    Compute cosine similarity between the resume and each candidate occupation.

    Returns {soc_code: raw_cosine_similarity}.
    Only scores occupations in `candidate_soc_codes` (avoids embedding 1000+
    descriptions per call — those are pre-cached and indexed).
    """
    from src.nlp_skills import _embed

    soc_codes, occ_embs = load_occ_embeddings()
    if not soc_codes:
        return {}

    query = build_resume_query(experience_lines, skills)
    query_embs = _embed([query])
    if query_embs is None:
        return {}

    query_emb = query_embs[0]
    soc_to_idx = {soc: i for i, soc in enumerate(soc_codes)}

    scores: Dict[str, float] = {}
    for soc in candidate_soc_codes:
        idx = soc_to_idx.get(soc)
        if idx is None:
            continue
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sim = float(np.nan_to_num(np.dot(occ_embs[idx], query_emb)))
        scores[soc] = round(sim, 4)

    return scores


def blend_scores(
    candidates: List[Dict],
    semantic_scores: Dict[str, float],
    semantic_weight: float = 0.40,
) -> List[Dict]:
    """
    Produce a final ranked list by blending tool-based and semantic scores.

    Both are min-max normalised to [0, 1] before blending so neither
    dominates due to scale differences.

    semantic_weight=0.40 means:  60% tool match  +  40% semantic similarity.
    """
    if not semantic_scores or not candidates:
        for c in candidates:
            c["semantic_score"] = None
            c["blended_score"] = c["match_score"]
        return candidates

    tool_vals = [c["match_score"] for c in candidates]
    t_min, t_max = min(tool_vals), max(tool_vals)
    t_range = max(t_max - t_min, 1e-6)

    sem_vals = [v for v in semantic_scores.values() if v is not None]
    s_min = min(sem_vals) if sem_vals else 0.0
    s_max = max(sem_vals) if sem_vals else 1.0
    s_range = max(s_max - s_min, 1e-6)

    blended: List[Dict] = []
    for cand in candidates:
        t_norm = (cand["match_score"] - t_min) / t_range
        sem_raw = semantic_scores.get(cand["soc_code"])
        if sem_raw is not None:
            s_norm = (sem_raw - s_min) / s_range
        else:
            s_norm = 0.0

        score = round(
            (1.0 - semantic_weight) * t_norm + semantic_weight * s_norm, 4
        )
        blended.append({
            **cand,
            "semantic_score": sem_raw,
            "blended_score": score,
        })

    blended.sort(key=lambda x: x["blended_score"], reverse=True)
    return blended
