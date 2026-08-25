"""The LLM forecast arm: fully offline. No test may hit the real Gemini API."""

from __future__ import annotations

import json

import pytest

from src.models import llm_forecaster as llm


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "llm_cache")


@pytest.fixture(autouse=True)
def has_key(monkeypatch):
    """Most tests need a key present; the one that doesn't overrides this itself."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


class TestParsing:
    def test_parses_wellformed_json(self, monkeypatch):
        reply = '{"low": 90.0, "high": 110.0, "reason": "steady growth expected"}'
        monkeypatch.setattr(llm, "_call_gemini", lambda prompt, key: reply)

        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])

        assert result["abstained"] is False
        assert result["lo"] == pytest.approx(90.0)
        assert result["hi"] == pytest.approx(110.0)
        assert result["point"] == pytest.approx(100.0)
        assert result["reason"] == "steady growth expected"

    def test_parses_json_wrapped_in_a_markdown_fence(self, monkeypatch):
        reply = '```json\n{"low": 50.0, "high": 60.0, "reason": "ok"}\n```'
        monkeypatch.setattr(llm, "_call_gemini", lambda prompt, key: reply)

        result = llm.forecast("MSFT", "2026-08-01", closes=[55.0] * 60, headlines=[])
        assert result["abstained"] is False
        assert result["lo"] == pytest.approx(50.0)
        assert result["hi"] == pytest.approx(60.0)

    def test_malformed_reply_is_an_abstention_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(llm, "_call_gemini", lambda prompt, key: "not json at all")

        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])

        assert result["abstained"] is True
        assert result["point"] is None
        assert result["lo"] is None
        assert result["hi"] is None

    def test_missing_fields_are_an_abstention(self, monkeypatch):
        monkeypatch.setattr(llm, "_call_gemini", lambda prompt, key: '{"reason": "no numbers"}')
        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        assert result["abstained"] is True

    def test_a_refusal_is_an_abstention(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_call_gemini", lambda prompt, key: "I can't predict stock prices."
        )
        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        assert result["abstained"] is True

    def test_inverted_range_is_rejected(self, monkeypatch):
        reply = '{"low": 120.0, "high": 80.0, "reason": "inverted"}'
        monkeypatch.setattr(llm, "_call_gemini", lambda prompt, key: reply)

        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])

        assert result["abstained"] is True
        assert result["point"] is None

    def test_equal_low_and_high_is_rejected(self, monkeypatch):
        reply = '{"low": 100.0, "high": 100.0, "reason": "flat"}'
        monkeypatch.setattr(llm, "_call_gemini", lambda prompt, key: reply)
        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        assert result["abstained"] is True

    def test_a_transport_error_is_an_abstention(self, monkeypatch):
        def raising(prompt, key):
            raise ConnectionError("network unreachable")

        monkeypatch.setattr(llm, "_call_gemini", raising)
        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        assert result["abstained"] is True


class TestCaching:
    def test_cache_prevents_second_call(self, monkeypatch):
        calls = {"n": 0}

        def fake_call(prompt, key):
            calls["n"] += 1
            return '{"low": 90.0, "high": 110.0, "reason": "ok"}'

        monkeypatch.setattr(llm, "_call_gemini", fake_call)

        first = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        second = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])

        assert calls["n"] == 1
        assert first == second

    def test_cache_is_written_to_disk(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            llm, "_call_gemini", lambda prompt, key: '{"low": 1.0, "high": 2.0, "reason": "x"}'
        )
        llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])

        cache_file = llm.CACHE_DIR / "2026-08-01_AAPL.json"
        assert cache_file.exists()
        with open(cache_file, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["lo"] == pytest.approx(1.0)

    def test_different_tickers_get_separate_cache_entries(self, monkeypatch):
        calls = {"n": 0}

        def fake_call(prompt, key):
            calls["n"] += 1
            return '{"low": 90.0, "high": 110.0, "reason": "ok"}'

        monkeypatch.setattr(llm, "_call_gemini", fake_call)
        llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        llm.forecast("MSFT", "2026-08-01", closes=[100.0] * 60, headlines=[])
        assert calls["n"] == 2


class TestSkippable:
    def test_module_is_skippable_without_a_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        def explode(prompt, key):
            raise AssertionError("must not call the API without a key")

        monkeypatch.setattr(llm, "_call_gemini", explode)

        assert llm.is_available() is False
        result = llm.forecast("AAPL", "2026-08-01", closes=[100.0] * 60, headlines=[])
        assert result["abstained"] is True
        assert result["point"] is None

    def test_is_available_true_when_key_present(self):
        assert llm.is_available() is True
