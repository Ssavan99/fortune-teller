"""The rolling engine's one non-negotiable property: nothing at or after as_of can leak in.

Real price data is used throughout (not synthetic) because the leakage test needs a realistic
train/val split with enough history either side of the boundary — a few hundred bars of
synthetic noise would not exercise the same code paths as a real six-month validation window.
Networks are shrunk via ``train_config_overrides`` to keep the suite fast, mirroring the
``fast_config`` pattern in ``tests/test_train.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import rolling
from src.data import prices

AS_OF = pd.Timestamp("2023-06-01")
HORIZON = 21

FAST = {"hidden": 8, "layers": 1, "max_epochs": 2, "patience": 2}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return prices.load()


@pytest.fixture(scope="module")
def records(df) -> list[dict]:
    return rolling.run_cycle(
        df, AS_OF, horizon=HORIZON, seed=1, train_config_overrides=FAST
    )


class TestAsOfBoundary:
    def test_as_of_boundary_holds(self, df):
        """The critical test. Poison every close/open/high/low on or after as_of by 100x and
        rerun with the same seed — every prediction must be byte-identical, because nothing
        past the boundary should have been read."""
        poisoned = df.copy()
        future = poisoned["date"] >= AS_OF
        for col in ("open", "high", "low", "close", "volume"):
            poisoned.loc[future, col] *= 100.0

        clean_records = rolling.run_cycle(
            df, AS_OF, horizon=HORIZON, seed=1, train_config_overrides=FAST
        )
        poisoned_records = rolling.run_cycle(
            poisoned, AS_OF, horizon=HORIZON, seed=1, train_config_overrides=FAST
        )

        clean_by_key = {(r["symbol"], r["model"]): r for r in clean_records}
        poisoned_by_key = {(r["symbol"], r["model"]): r for r in poisoned_records}
        assert clean_by_key.keys() == poisoned_by_key.keys()
        for key, clean_row in clean_by_key.items():
            poisoned_row = poisoned_by_key[key]
            assert clean_row["point"] == pytest.approx(poisoned_row["point"], abs=1e-9)
            assert clean_row["lo"] == pytest.approx(poisoned_row["lo"], abs=1e-9)
            assert clean_row["hi"] == pytest.approx(poisoned_row["hi"], abs=1e-9)
            assert clean_row["target_date"] == poisoned_row["target_date"]

    def test_no_record_targets_a_date_before_its_as_of(self, records):
        for r in records:
            assert pd.Timestamp(r["target_date"]) >= pd.Timestamp(r["as_of"])

    def test_target_date_is_exactly_horizon_sessions_after_as_of(self, df, records):
        expected = rolling.target_date_for(df, AS_OF, HORIZON)
        for r in records:
            assert pd.Timestamp(r["target_date"]) == expected


class TestRecordShape:
    def test_both_models_predict_every_symbol(self, records):
        by_symbol: dict[str, set[str]] = {}
        for r in records:
            by_symbol.setdefault(r["symbol"], set()).add(r["model"])
        assert set(by_symbol.keys()) == set(prices.TICKERS)
        for symbol, models in by_symbol.items():
            assert models == {"lstm", "persistence"}, symbol

    def test_intervals_are_ordered(self, records):
        """``lo < hi`` is a true invariant (the quantile levels are strictly ordered, and
        real relative-residual distributions are effectively never point masses).

        ``lo < point < hi`` is NOT a true invariant of the empirical-residual method and is
        deliberately not asserted here: during a strongly trending validation window (e.g.
        META roughly doubled between Dec 2022 and May 2023, the validation window for this
        as-of), even the 10th percentile of the 21-day return can be positive, which pushes
        the whole interval above the point forecast. That is a correct, honest reflection of
        the historical data, not a bug — see Blockers in the plan file for the reasoning.
        Instead, check that the point falls inside its own interval for the large majority of
        records, which would catch a real bug (e.g. swapped lo/hi quantiles) while tolerating
        this legitimate edge case.
        """
        for r in records:
            assert r["lo"] < r["hi"], r
        straddled = sum(1 for r in records if r["lo"] < r["point"] < r["hi"])
        assert straddled / len(records) >= 0.8

    def test_schema_is_exactly_as_specified(self, records):
        expected_fields = {
            "as_of", "target_date", "symbol", "model", "point", "lo", "hi",
            "level", "actual", "covered", "abs_error", "created_utc",
        }
        for r in records:
            assert set(r.keys()) == expected_fields
            assert r["actual"] is None
            assert r["covered"] is None
            assert r["abs_error"] is None
            assert r["level"] == pytest.approx(0.80)


class TestDeterminism:
    def test_same_seed_same_predictions(self, df):
        r1 = rolling.run_cycle(df, AS_OF, horizon=HORIZON, seed=42, train_config_overrides=FAST)
        r2 = rolling.run_cycle(df, AS_OF, horizon=HORIZON, seed=42, train_config_overrides=FAST)
        p1 = {(r["symbol"], r["model"]): r["point"] for r in r1}
        p2 = {(r["symbol"], r["model"]): r["point"] for r in r2}
        assert p1.keys() == p2.keys()
        for key in p1:
            assert p1[key] == pytest.approx(p2[key], abs=1e-9)


class TestNoHistoryGuard:
    def test_raises_when_as_of_predates_all_data(self, df):
        with pytest.raises(ValueError, match="no history"):
            rolling.run_cycle(df, pd.Timestamp("2000-01-01"), horizon=HORIZON, seed=1)

    def test_raises_on_insufficient_history_for_validation_split(self, df):
        """The snapshot starts 2020-12-10; a few weeks in, there isn't 6 months of history
        yet for the train/val split — this must fail loudly, not KeyError on a missing
        'train'/'val' bucket."""
        with pytest.raises(ValueError, match="insufficient history"):
            rolling.run_cycle(
                df, pd.Timestamp("2021-01-04"), horizon=HORIZON, seed=1,
                train_config_overrides=FAST,
            )


class TestStampMode:
    def test_stamps_every_record(self):
        records = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
        rolling.stamp_mode(records, "backtest")
        assert all(r["mode"] == "backtest" for r in records)

    def test_rejects_an_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            rolling.stamp_mode([{"symbol": "AAPL"}], "future")


class TestAsOfNormalization:
    def test_time_of_day_on_as_of_does_not_leak_that_days_bar(self, df):
        """A caller-supplied as_of with a nonzero time-of-day (e.g. pd.Timestamp.now()) must
        behave identically to the midnight-normalized date — otherwise that calendar day's own
        bar leaks into history under a naive `<` comparison."""
        midnight = rolling.run_cycle(
            df, AS_OF, horizon=HORIZON, seed=1, train_config_overrides=FAST
        )
        with_time = rolling.run_cycle(
            df, AS_OF + pd.Timedelta(hours=9, minutes=30), horizon=HORIZON, seed=1,
            train_config_overrides=FAST,
        )
        m = {(r["symbol"], r["model"]): r for r in midnight}
        t = {(r["symbol"], r["model"]): r for r in with_time}
        assert m.keys() == t.keys()
        for key in m:
            assert m[key]["point"] == pytest.approx(t[key]["point"], abs=1e-9)
            assert m[key]["target_date"] == t[key]["target_date"]


class TestScoreRecord:
    def test_fills_matured_predictions(self, df, records):
        # target_date=2023-06-30 has long since passed in a snapshot running to 2026.
        scored = [rolling.score_record(r, df) for r in records]
        assert all(r["actual"] is not None for r in scored)
        assert all(isinstance(r["covered"], bool) for r in scored)
        assert all(r["abs_error"] >= 0 for r in scored)

    def test_leaves_unmatured_predictions_null(self, df):
        latest = df["date"].max()
        recs = rolling.run_cycle(
            df, latest, horizon=HORIZON, seed=1, train_config_overrides=FAST
        )
        scored = [rolling.score_record(r, df) for r in recs]
        assert all(r["actual"] is None for r in scored)
        assert all(r["covered"] is None for r in scored)
        assert all(r["abs_error"] is None for r in scored)

    def test_never_revises_an_existing_score(self, df, records):
        once = [rolling.score_record(r, df) for r in records]
        poisoned_df = df.copy()
        poisoned_df.loc[poisoned_df["date"] > AS_OF, "close"] *= 1000.0
        twice = [rolling.score_record(r, poisoned_df) for r in once]
        for a, b in zip(once, twice, strict=True):
            assert a["actual"] == b["actual"]
            assert a["covered"] == b["covered"]
            assert a["abs_error"] == b["abs_error"]

    def test_never_modifies_the_prediction_fields(self, df, records):
        scored = [rolling.score_record(r, df) for r in records]
        for original, result in zip(records, scored, strict=True):
            assert result["point"] == original["point"]
            assert result["lo"] == original["lo"]
            assert result["hi"] == original["hi"]
            assert result["as_of"] == original["as_of"]
            assert result["target_date"] == original["target_date"]

    def test_covered_matches_the_interval(self, df, records):
        scored = [rolling.score_record(r, df) for r in records]
        for r in scored:
            expected = r["lo"] <= r["actual"] <= r["hi"]
            assert r["covered"] == expected
