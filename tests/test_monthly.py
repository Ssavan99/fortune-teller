"""The live monthly run's append-only guarantee: a prediction, once written, is never revised.

The LLM arm is monkeypatched off (``llm_forecaster.is_available`` -> ``False``) in most tests
here regardless of whether a real ``GEMINI_API_KEY`` happens to be set in the environment
running this suite — these tests are about the ledger's append-only contract, not about the
LLM arm, and must never make a real network call either way.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import run_monthly
from src.data import prices

AS_OF = pd.Timestamp("2023-06-01")
HORIZON = 21
FAST = {"hidden": 8, "layers": 1, "max_epochs": 2, "patience": 2}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return prices.load()


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setattr(run_monthly.llm_forecaster, "is_available", lambda: False)


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestPredictIsAppendOnly:
    def test_predict_twice_is_a_noop(self, df, tmp_path, capsys):
        ledger_path = tmp_path / "scoreboard_live.json"

        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        first = _read(ledger_path)

        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        second = _read(ledger_path)
        out = capsys.readouterr().out

        assert first == second
        assert "already predicted" in out

    def test_live_rows_are_marked_live(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        rows = _read(ledger_path)
        assert rows
        assert all(r["mode"] == "live" for r in rows)

    def test_new_rows_start_unscored(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        rows = _read(ledger_path)
        assert all(r["actual"] is None for r in rows)
        assert all(r["covered"] is None for r in rows)
        assert all(r["abs_error"] is None for r in rows)

    def test_predicting_a_second_as_of_appends_rather_than_replaces(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        n_first = len(_read(ledger_path))

        later = pd.Timestamp("2023-07-03")
        run_monthly.predict(df, later, ledger_path=ledger_path, train_config_overrides=FAST)
        rows = _read(ledger_path)

        assert len(rows) > n_first
        as_ofs = {r["as_of"] for r in rows}
        assert as_ofs == {str(AS_OF.date()), str(later.date())}


class TestScoreNeverRevises:
    def test_scoring_does_not_alter_the_prediction(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        before = _read(ledger_path)

        run_monthly.score(df, ledger_path=ledger_path)
        after = _read(ledger_path)

        by_key_before = {(r["symbol"], r["model"]): r for r in before}
        by_key_after = {(r["symbol"], r["model"]): r for r in after}
        for key, b in by_key_before.items():
            a = by_key_after[key]
            assert a["point"] == b["point"]
            assert a["lo"] == b["lo"]
            assert a["hi"] == b["hi"]
            assert a["as_of"] == b["as_of"]
            assert a["target_date"] == b["target_date"]

    def test_unscored_rows_have_null_actual(self, df, tmp_path):
        """as_of is the snapshot's most recent date -> nothing has matured yet."""
        ledger_path = tmp_path / "scoreboard_live.json"
        latest = df["date"].max()
        run_monthly.predict(df, latest, ledger_path=ledger_path, train_config_overrides=FAST)

        run_monthly.score(df, ledger_path=ledger_path)
        rows = _read(ledger_path)
        assert all(r["actual"] is None for r in rows)

    def test_scoring_fills_matured_rows(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)

        run_monthly.score(df, ledger_path=ledger_path)
        rows = _read(ledger_path)
        # AS_OF=2023-06-01 with horizon=21 matured long before the snapshot's latest date.
        assert all(r["actual"] is not None for r in rows)

    def test_rescoring_an_already_scored_ledger_is_stable(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        run_monthly.score(df, ledger_path=ledger_path)
        once = _read(ledger_path)

        run_monthly.score(df, ledger_path=ledger_path)
        twice = _read(ledger_path)
        assert once == twice


class TestLlmArm:
    def test_llm_rows_are_included_and_marked_live_when_available(self, df, tmp_path, monkeypatch):
        monkeypatch.setattr(run_monthly.llm_forecaster, "is_available", lambda: True)
        monkeypatch.setattr(
            run_monthly.llm_forecaster,
            "forecast",
            lambda ticker, as_of, closes, headlines: {
                "point": 100.0, "lo": 90.0, "hi": 110.0, "reason": "test", "abstained": False,
            },
        )
        monkeypatch.setattr(run_monthly.news, "fetch_headlines", lambda ticker: [])

        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        rows = _read(ledger_path)

        llm_rows = [r for r in rows if r["model"] == "llm"]
        assert len(llm_rows) == len(prices.TICKERS)
        assert all(r["mode"] == "live" for r in llm_rows)
        assert all(r["level"] is None for r in llm_rows)

    def test_one_tickers_llm_failure_does_not_lose_the_others(self, df, tmp_path, monkeypatch):
        """A real exception (not one llm_forecaster.forecast already catches) from one ticker
        must not discard the LSTM/persistence rows for all 15 tickers, which already cost real
        training time by the point the LLM loop runs."""
        monkeypatch.setattr(run_monthly.llm_forecaster, "is_available", lambda: True)

        def flaky_forecast(ticker, as_of, closes, headlines):
            if ticker == "AAPL":
                raise RuntimeError("simulated corrupted cache")
            return {"point": 100.0, "lo": 90.0, "hi": 110.0, "reason": "ok", "abstained": False}

        monkeypatch.setattr(run_monthly.llm_forecaster, "forecast", flaky_forecast)
        monkeypatch.setattr(run_monthly.news, "fetch_headlines", lambda ticker: [])

        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        rows = _read(ledger_path)

        # The whole run must not have aborted: LSTM/persistence rows for every ticker are
        # present, and every ticker (including AAPL) still has an llm row, just abstained.
        lstm_symbols = {r["symbol"] for r in rows if r["model"] == "lstm"}
        assert lstm_symbols == set(prices.TICKERS)
        llm_rows = {r["symbol"]: r for r in rows if r["model"] == "llm"}
        assert set(llm_rows) == set(prices.TICKERS)
        assert llm_rows["AAPL"]["point"] is None
        assert llm_rows["MSFT"]["point"] == pytest.approx(100.0)


class TestAtomicWrite:
    def test_ledger_write_leaves_no_tmp_file_behind(self, df, tmp_path):
        ledger_path = tmp_path / "scoreboard_live.json"
        run_monthly.predict(df, AS_OF, ledger_path=ledger_path, train_config_overrides=FAST)
        assert ledger_path.exists()
        assert not ledger_path.with_suffix(ledger_path.suffix + ".tmp").exists()
