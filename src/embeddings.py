"""
Shared sentence-transformers access.

One model, loaded lazily and cached for the process, plus the text
normalisation every module matches on. Everything here degrades to None rather
than raising, so the rest of the system can fall back to keyword heuristics
when the model (or its download) is unavailable.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np


def embeddings_enabled() -> bool:
    """
    Whether semantic matching is switched on.

    The model costs roughly half a gigabyte of resident memory and downloads
    ~90 MB on first use. On a small container that is the difference between
    an analysis finishing and the process being killed mid-request, so hosts
    that cannot afford it set ENABLE_EMBEDDINGS=0 and every embedding call
    falls back to keyword matching.
    """
    return os.getenv("ENABLE_EMBEDDINGS", "1").lower() not in {"0", "false", "off", "no"}


@lru_cache(maxsize=1)
def _load_model():
    """Load sentence-transformers model once per process; return None on failure."""
    if not embeddings_enabled():
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


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


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text).lower().strip()


embed = _embed
normalise = _norm
