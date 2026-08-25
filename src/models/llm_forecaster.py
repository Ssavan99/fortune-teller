"""The LLM forecast arm: Gemini free tier, strictly parsed, disk-cached, live-only.

Live-only by design, not by accident: the free RSS news source (:mod:`src.data.news`) has no
deep history, so there is no honest way to backfill this arm the way the LSTM and persistence
arms are backfilled. Presenting a "backtested" LLM score built from headlines it could never
have actually seen at the time would be exactly the kind of retroactive flattery this repo
exists to avoid. So this module is only ever called from the live monthly run, never from
:mod:`scripts.backfill_scoreboard`.

One call per ``(as_of, ticker)``, ever — every response is cached to
``data/llm_cache/{as_of}_{ticker}.json`` and committed, and a cached key is never re-called even
across separate process runs. A malformed reply, a refusal, an unparseable body, or an inverted
``low >= high`` range all become an **abstention** (``point``/``lo``/``hi`` all ``None``) rather
than a fabricated number — an LLM that won't commit to a number is more honest than one that is
coerced into producing a nonsense one just to fill a cell in a table.

The API key is read from the environment only (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``) and is
never written anywhere, including into the cache files this module writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "llm_cache"

MODEL = "gemini-flash-lite-latest"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
TIMEOUT_S = 30
MAX_CLOSES = 60
MAX_HEADLINES = 10


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available() -> bool:
    """Whether a free-tier key is present. Callers use this to skip the arm entirely rather
    than call in and get an abstention for every ticker."""
    return api_key() is not None


def _cache_path(as_of: str, ticker: str) -> Path:
    return CACHE_DIR / f"{as_of}_{ticker}.json"


def _build_prompt(ticker: str, closes: list[float], headlines: list[str]) -> str:
    recent_closes = closes[-MAX_CLOSES:]
    closes_str = ", ".join(f"{c:.2f}" for c in recent_closes)
    recent_headlines = headlines[:MAX_HEADLINES]
    headlines_str = "\n".join(f"- {h}" for h in recent_headlines) or "(no recent headlines)"

    return (
        f"You are forecasting the closing price of {ticker} 21 trading days from now.\n\n"
        f"Last {len(recent_closes)} daily closes, oldest to newest:\n{closes_str}\n\n"
        f"Recent headlines:\n{headlines_str}\n\n"
        "Respond with ONLY strict JSON and nothing else — no markdown fence, no commentary "
        'before or after it: {"low": <number>, "high": <number>, "reason": "<one sentence>"}\n'
        "low and high are a plausible closing-price range 21 trading days from now. "
        "low must be strictly less than high."
    )


def _call_gemini(prompt: str, key: str) -> str:
    """Raw call to the Gemini API. Raises on any transport or HTTP error; the caller catches."""
    resp = requests.post(
        f"{API_URL}?key={key}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse(raw_text: str) -> dict | None:
    """Strict parse of a reply into ``{"low", "high", "reason"}``, or ``None`` on anything
    malformed: not JSON, not an object, missing/non-numeric low or high, or low >= high."""
    text = _strip_markdown_fence(raw_text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "low" not in obj or "high" not in obj:
        return None
    try:
        low = float(obj["low"])
        high = float(obj["high"])
    except (TypeError, ValueError):
        return None
    if not (low < high):
        return None
    reason = obj.get("reason")
    return {"low": low, "high": high, "reason": str(reason) if reason is not None else ""}


def _abstention(reason: str) -> dict:
    return {"point": None, "lo": None, "hi": None, "reason": reason, "abstained": True}


def _forecast_uncached(ticker: str, closes: list[float], headlines: list[str], key: str) -> dict:
    prompt = _build_prompt(ticker, closes, headlines)
    try:
        raw_text = _call_gemini(prompt, key)
    except Exception as exc:  # noqa: BLE001 - any transport/HTTP failure becomes an abstention
        return _abstention(f"API call failed: {exc}")

    parsed = _parse(raw_text)
    if parsed is None:
        return _abstention("malformed, non-JSON, or inverted-range reply")

    low, high = parsed["low"], parsed["high"]
    return {
        "point": (low + high) / 2.0,
        "lo": low,
        "hi": high,
        "reason": parsed["reason"],
        "abstained": False,
    }


def forecast(ticker: str, as_of: str, closes: list[float], headlines: list[str]) -> dict:
    """One cached, strictly-parsed forecast for ``(as_of, ticker)``.

    Returns ``{"point", "lo", "hi", "reason", "abstained"}``. Without an API key in the
    environment, returns an abstention immediately and never attempts a network call or writes
    a cache entry — a cache entry means "we tried", and "no key" means we never did.
    """
    key = api_key()
    if key is None:
        return _abstention("no GEMINI_API_KEY or GOOGLE_API_KEY in environment")

    cache_path = _cache_path(as_of, ticker)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    result = _forecast_uncached(ticker, closes, headlines, key)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
