"""News fetching and FinBERT scoring: everything here runs fully offline.

``feedparser.parse`` and the HuggingFace pipeline are both mocked out — no test may hit the
network or download real model weights, so a broken feed or a slow model download can never
turn into a hanging or flaky test run.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.data import finbert, news


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point both caches at a throwaway directory so tests never touch the real cache."""
    cache_dir = tmp_path / "news_cache"
    monkeypatch.setattr(news, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(finbert, "SCORES_CACHE", cache_dir / "scores.json")
    return cache_dir


class TestFetchHeadlines:
    def test_malformed_feed_returns_empty_not_exception(self, monkeypatch):
        """A bozo/parse-error feed (no usable entries) must not raise."""

        def fake_parse(url):
            return SimpleNamespace(bozo=True, entries=[])

        monkeypatch.setattr("feedparser.parse", fake_parse)
        assert news.fetch_headlines("AAPL") == []

    def test_network_error_returns_empty_not_exception(self, monkeypatch):
        def raising_parse(url):
            raise OSError("connection refused")

        monkeypatch.setattr("feedparser.parse", raising_parse)
        assert news.fetch_headlines("AAPL") == []

    def test_empty_feed_is_recorded_as_unavailable(self, monkeypatch):
        """An empty entries list is distinguishable from real headlines: it's just ``[]``,
        never silently padded or treated as a headline."""

        def fake_parse(url):
            return SimpleNamespace(bozo=False, entries=[])

        monkeypatch.setattr("feedparser.parse", fake_parse)
        result = news.fetch_headlines("ZZZZ")
        assert result == []
        assert isinstance(result, list)

    def test_entries_missing_publish_date_are_dropped(self, monkeypatch):
        entry = SimpleNamespace(title="Some headline", link="http://x", published_parsed=None)

        def fake_parse(url):
            return SimpleNamespace(bozo=False, entries=[entry])

        monkeypatch.setattr("feedparser.parse", fake_parse)
        assert news.fetch_headlines("AAPL") == []

    def test_good_feed_returns_parsed_headlines(self, monkeypatch):
        entry = SimpleNamespace(
            title="AAPL beats expectations",
            link="http://example.com/a",
            published_parsed=(2026, 8, 24, 12, 0, 0, 0, 0, 0),
        )

        def fake_parse(url):
            return SimpleNamespace(bozo=False, entries=[entry])

        monkeypatch.setattr("feedparser.parse", fake_parse)
        result = news.fetch_headlines("AAPL")
        assert result == [
            {
                "title": "AAPL beats expectations",
                "published": "2026-08-24",
                "link": "http://example.com/a",
                "ticker": "AAPL",
            }
        ]

    def test_second_call_same_day_uses_cache_not_network(self, monkeypatch):
        calls = {"n": 0}
        entry = SimpleNamespace(
            title="Cached headline",
            link="http://example.com/b",
            published_parsed=(2026, 8, 24, 9, 0, 0, 0, 0, 0),
        )

        def fake_parse(url):
            calls["n"] += 1
            return SimpleNamespace(bozo=False, entries=[entry])

        monkeypatch.setattr("feedparser.parse", fake_parse)
        first = news.fetch_headlines("MSFT")
        second = news.fetch_headlines("MSFT")
        assert first == second
        assert calls["n"] == 1


class TestScoresCacheAndDailyScore:
    def test_no_headlines_gives_none_not_zero(self, monkeypatch):
        monkeypatch.setattr(news, "fetch_headlines", lambda ticker: [])
        result = finbert.daily_score("AAPL", "2026-08-24")
        assert result is None
        assert result != 0.0

    def test_cache_roundtrip_preserves_shape(self, isolated_cache):
        isolated_cache.mkdir(parents=True, exist_ok=True)
        payload = {"AAPL_2026-08-24": 0.42, "MSFT_2026-08-24": -0.13}
        with open(finbert.SCORES_CACHE, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        loaded = finbert._load_scores_cache()
        assert loaded == payload
        assert isinstance(loaded["AAPL_2026-08-24"], float)
        assert isinstance(loaded["MSFT_2026-08-24"], float)

    def test_headline_cache_roundtrip_preserves_shape(self, isolated_cache, monkeypatch):
        entry = SimpleNamespace(
            title="Roundtrip headline",
            link="http://example.com/c",
            published_parsed=(2026, 8, 24, 9, 0, 0, 0, 0, 0),
        )

        def fake_parse(url):
            return SimpleNamespace(bozo=False, entries=[entry])

        monkeypatch.setattr("feedparser.parse", fake_parse)
        written = news.fetch_headlines("GOOG")

        cache_file = isolated_cache / f"GOOG_{news._today()}.json"
        assert cache_file.exists()
        with open(cache_file, encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded == written
        assert isinstance(reloaded, list)
        assert isinstance(reloaded[0]["title"], str)

    def test_daily_score_averages_and_caches(self, monkeypatch):
        headlines = [
            {"title": "Great news", "published": "2026-08-24", "link": "", "ticker": "AAPL"},
            {"title": "Bad news", "published": "2026-08-24", "link": "", "ticker": "AAPL"},
            {"title": "Other day", "published": "2026-08-23", "link": "", "ticker": "AAPL"},
        ]
        monkeypatch.setattr(news, "fetch_headlines", lambda ticker: headlines)
        monkeypatch.setattr(finbert, "score_headlines", lambda titles: [0.8, -0.4])

        score = finbert.daily_score("AAPL", "2026-08-24")
        assert score == pytest.approx(0.2)

        # Second call must hit the cache, not recompute (score_headlines would blow up if
        # called again since we only stub two return values above).
        monkeypatch.setattr(
            finbert, "score_headlines", lambda titles: (_ for _ in ()).throw(AssertionError)
        )
        assert finbert.daily_score("AAPL", "2026-08-24") == pytest.approx(0.2)


class TestFinbertMapping:
    def test_finbert_mapping_signs_are_correct(self, monkeypatch):
        fake_outputs = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.7},
            {"label": "neutral", "score": 0.6},
        ]

        class FakePipeline:
            def __call__(self, titles):
                return fake_outputs

        monkeypatch.setattr(finbert, "_get_pipeline", lambda: FakePipeline())

        result = finbert.score_headlines(["a", "b", "c"])
        assert result == pytest.approx([0.9, -0.7, 0.0])

    def test_finbert_mapping_is_case_insensitive(self, monkeypatch):
        fake_outputs = [
            {"label": "Positive", "score": 0.55},
            {"label": "Negative", "score": 0.65},
            {"label": "Neutral", "score": 0.99},
        ]

        class FakePipeline:
            def __call__(self, titles):
                return fake_outputs

        monkeypatch.setattr(finbert, "_get_pipeline", lambda: FakePipeline())

        result = finbert.score_headlines(["a", "b", "c"])
        assert result == pytest.approx([0.55, -0.65, 0.0])

    def test_empty_titles_short_circuits_without_loading_pipeline(self, monkeypatch):
        def explode():
            raise AssertionError("pipeline should not be loaded for empty input")

        monkeypatch.setattr(finbert, "_get_pipeline", explode)
        assert finbert.score_headlines([]) == []
