"""Free, keyless headline fetching for the live scoreboard's news arm.

Yahoo's RSS feed has no SLA and no key: it can time out, return malformed XML, or simply have
nothing for a thinly-covered ticker on a quiet day. A batch run over the whole ticker universe
must survive all of that per-ticker — one bad feed degrading to zero headlines for that symbol,
never an exception that aborts the run for every other ticker. Results are cached by calendar
day so a re-run (e.g. retrying after a partial failure) doesn't re-hit the network for tickers
that already succeeded today.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "news_cache"

FEED_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _cache_path(ticker: str, day: str) -> Path:
    return CACHE_DIR / f"{ticker}_{day}.json"


def _parse_entry(entry, ticker: str) -> dict | None:
    """Turn one feedparser entry into our headline dict, or ``None`` if it isn't usable."""
    title = getattr(entry, "title", None)
    link = getattr(entry, "link", None)
    parsed = getattr(entry, "published_parsed", None)
    if not title or not parsed:
        return None
    published = datetime(*parsed[:6], tzinfo=timezone.utc).date().isoformat()
    return {"title": title, "published": published, "link": link or "", "ticker": ticker}


def fetch_headlines(ticker: str) -> list[dict]:
    """Fetch today's headlines for ``ticker``, using (and writing) the on-disk cache.

    Never raises. A network error, an empty feed, or entries with no usable publish date all
    fall through to ``[]`` — the caller (a batch run over many tickers) must not be aborted by
    one bad feed.
    """
    day = _today()
    cache_path = _cache_path(ticker, day)

    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass  # fall through and re-fetch

    try:
        import feedparser

        feed = feedparser.parse(FEED_URL.format(ticker=ticker))
        entries = getattr(feed, "entries", None) or []
        headlines = [h for e in entries if (h := _parse_entry(e, ticker)) is not None]
    except Exception:
        headlines = []

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(headlines, f, indent=2)
    except OSError:
        pass  # caching is best-effort; a write failure must not lose the fetched result

    return headlines


def main() -> None:
    from src.data.prices import TICKERS

    for ticker in TICKERS:
        headlines = fetch_headlines(ticker)
        if headlines:
            print(f"{ticker}: {len(headlines)} headlines")
        else:
            print(f"{ticker}: 0 headlines (unavailable)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch headlines for all tickers")
    parser.parse_args()
    main()
