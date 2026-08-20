"""
Shared sentence-transformers access.

One model, loaded lazily and cached for the process, plus the text
normalisation every module matches on. Everything here degrades to None rather
than raising, so the rest of the system can fall back to keyword heuristics
when the model (or its download) is unavailable.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np


@lru_cache(maxsize=1)
def _load_model():
    """Load sentence-transformers model once per process; return None on failure."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# Per-string embedding cache. Scoring a resume against several occupations
# re-embeds the same skill list and a heavily overlapping tool vocabulary each
# time; encoding is the dominant cost, so remembering vectors turns repeat
# passes nearly free. Bounded because tool names are drawn from a fixed corpus.
_CACHE: Dict[str, np.ndarray] = {}
_CACHE_MAX = 20000


def _embed(texts: List[str]) -> Optional[np.ndarray]:
    """Return (N, D) float32 embedding matrix, or None if model unavailable."""
    model = _load_model()
    if model is None or not texts:
        return None

    missing = [t for t in dict.fromkeys(texts) if t not in _CACHE]
    if missing:
        try:
            fresh = model.encode(
                missing, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            return None
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        for text, vec in zip(missing, np.array(fresh, dtype=np.float32)):
            _CACHE[text] = vec

    try:
        return np.stack([_CACHE[t] for t in texts]).astype(np.float32)
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Skill difficulty
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text).lower().strip()


# Public aliases — the underscored names are kept because most of the codebase
# already imports them.
embed = _embed
normalise = _norm
