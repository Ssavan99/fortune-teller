"""Confidence-weighted FinBERT sentiment for headline text.

The pipeline is loaded lazily, inside :func:`score_headlines`, rather than at import time.
``transformers`` pulls real model weights over the network on first use — importing this module
(e.g. from a test that mocks the pipeline) must never trigger that download, so the import and
construction live behind the function call, not at module scope.

A label's confidence score is folded into the sign rather than discarded: a headline FinBERT is
only 55% sure is "positive" should move the mean less than one it is 95% sure about. Neutral is
the one label with no sign to weight, so it always contributes exactly 0.0 regardless of the
pipeline's confidence in it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data import news

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_CACHE = news.CACHE_DIR / "scores.json"

_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        _pipeline = pipeline(
            "sentiment-analysis",
            model="yiyanghkust/finbert-tone",
            tokenizer="yiyanghkust/finbert-tone",
        )
    return _pipeline


def score_headlines(titles: list[str]) -> list[float]:
    """Run FinBERT over ``titles`` and map each result to a signed, confidence-weighted score.

    ``label`` is matched case-insensitively since different FinBERT builds have returned both
    ``"positive"`` and ``"Positive"``. An unrecognized label contributes ``0.0`` rather than
    raising, since a batch of many headlines should not fail entirely over one odd label.
    """
    if not titles:
        return []

    pipe = _get_pipeline()
    results = pipe(titles)

    scores = []
    for result in results:
        sign = _SIGN.get(str(result["label"]).lower(), 0.0)
        scores.append(sign * float(result["score"]))
    return scores


def _load_scores_cache() -> dict[str, float]:
    if SCORES_CACHE.exists():
        try:
            with open(SCORES_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_scores_cache(cache: dict[str, float]) -> None:
    try:
        SCORES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCORES_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass  # caching is best-effort


def daily_score(ticker: str, date: str) -> float | None:
    """Mean FinBERT score over ``ticker``'s headlines published on ``date``.

    Returns ``None``, never ``0.0``, when there are no headlines for that ticker/date pair.
    ``0.0`` would assert "the news was neutral", a claim the absence of any news does not
    support — downstream code must be able to tell the two apart.
    """
    key = f"{ticker}_{date}"
    cache = _load_scores_cache()
    if key in cache:
        return cache[key]

    headlines = [h for h in news.fetch_headlines(ticker) if h["published"] == date]
    if not headlines:
        return None

    scores = score_headlines([h["title"] for h in headlines])
    value = float(sum(scores) / len(scores))

    cache[key] = value
    _save_scores_cache(cache)
    return value
