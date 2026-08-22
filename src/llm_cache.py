"""
On-disk cache for LLM calls.

Every stage that talks to Groq — resume extraction, recommendation refinement,
job-fit analysis — is deterministic enough at low temperature that re-running
the same resume should not cost another request. The free tier has a daily
token cap, and hitting it degrades the whole report silently (regex parser,
templated summary), so caching is a quality feature, not just a speed one.

Keyed on model + prompt, so editing a prompt invalidates its entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ROOT / "data" / "cache" / "llm"))

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

MAX_AGE_DAYS = 30

# The SDK defaults to a 60s read timeout and retries twice, so a stalled
# provider can hold a request open for minutes with no way for the caller to
# tell. Two calls run per analysis, so this is bounded deliberately.
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "40"))


def _key(model: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()
    return digest[:32]


def _path(model: str, prompt: str) -> Path:
    return CACHE_DIR / f"{_key(model, prompt)}.json"


def enabled() -> bool:
    return os.getenv("LLM_CACHE", "1") not in {"0", "false", "off"}


def get(model: str, prompt: str) -> Optional[str]:
    """Cached completion text, or None on a miss / stale / unreadable entry."""
    if not enabled():
        return None
    path = _path(model, prompt)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - entry.get("ts", 0) > MAX_AGE_DAYS * 86400:
        return None
    return entry.get("response")


def put(model: str, prompt: str, response: str) -> None:
    if not enabled() or not response:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _path(model, prompt).write_text(
            json.dumps({"ts": time.time(), "model": model, "response": response}),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"[llm_cache] write failed: {e}", file=sys.stderr)


def complete(
    client: Any,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    json_mode: bool = True,
) -> str:
    """
    Cached chat completion. Returns the raw response text; callers parse it.
    A cache hit costs no request, so a rate-limited key still produces a full
    report for any resume seen before.
    """
    hit = get(model, prompt)
    if hit is not None:
        return hit

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs, timeout=TIMEOUT)
    text = resp.choices[0].message.content
    put(model, prompt, text)
    return text


def clear() -> int:
    """Delete every cached entry; returns how many files were removed."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n
