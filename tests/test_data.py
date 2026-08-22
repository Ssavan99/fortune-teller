"""Contracts on the committed data snapshots and the chronological splits."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import prices, splits


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return prices.load()


@pytest.fixture(scope="module")
def sent() -> pd.DataFrame:
    return pd.read_csv("data/sentiment_daily.csv", parse_dates=["date"])


class TestPriceSnapshot:
    def test_schema(self, df):
        assert list(df.columns) == prices.COLUMNS

    def test_universe_is_complete(self, df):
        assert set(df["symbol"]) == set(prices.TICKERS)

    def test_no_missing_prices(self, df):
        assert df[["open", "high", "low", "close", "volume"]].notna().all().all()

    def test_prices_are_positive(self, df):
        assert (df[["open", "high", "low", "close"]] > 0).all().all()

    def test_high_low_bracket_close(self, df):
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_dates_are_strictly_increasing_per_symbol(self, df):
        for symbol, part in df.groupby("symbol"):
            dates = part["date"].to_numpy()
            assert (dates[1:] > dates[:-1]).all(), f"{symbol} dates are not strictly increasing"

    def test_no_duplicate_bars(self, df):
        assert not df.duplicated(subset=["date", "symbol"]).any()

    def test_every_symbol_covers_the_same_calendar(self, df):
        counts = df.groupby("symbol").size()
        assert counts.nunique() == 1, f"uneven history: {counts.to_dict()}"

    def test_no_split_sized_discontinuities(self, df):
        """Adjusted prices should have no overnight moves of the size of a stock split.

        Guards the ``auto_adjust=True`` decision: NVDA's 10:1 in June 2024 would show up
        here as a −90% bar if the snapshot were ever regenerated unadjusted.
        """
        for symbol, part in df.groupby("symbol"):
            ratio = part["close"].to_numpy()[1:] / part["close"].to_numpy()[:-1]
            assert ratio.min() > 0.45, f"{symbol} has a split-sized drop"
            assert ratio.max() < 2.2, f"{symbol} has a split-sized jump"


class TestSentimentSnapshot:
    def test_coverage_window(self, sent):
        assert sent["date"].min() == pd.Timestamp("2020-12-21")
        assert sent["date"].max() == pd.Timestamp("2024-03-25")

    def test_universe_is_price_universe_minus_dell(self, sent):
        tickers = {c for c in sent.columns if c != "date"}
        assert tickers == set(prices.TICKERS) - {"DELL"}

    def test_scores_are_bounded(self, sent):
        values = sent.drop(columns=["date"])
        assert values.min().min() >= -1.0
        assert values.max().max() <= 1.0

    def test_ends_before_the_held_out_period(self, sent):
        """Experiment B cannot borrow the main test window; this is why."""
        assert sent["date"].max() < pd.Timestamp("2026-01-01")


class TestChronologicalSplit:
    def test_partitions_are_ordered_and_non_empty(self, df):
        split = splits.chronological_split(df)
        assert split.train["date"].max() < split.val["date"].min()
        assert split.val["date"].max() < split.test["date"].min()

    def test_cutoffs_are_respected(self, df):
        split = splits.chronological_split(df)
        assert split.train["date"].max() <= splits.TRAIN_END
        assert split.val["date"].max() <= splits.VAL_END
        assert split.test["date"].min() > splits.VAL_END

    def test_partitions_are_a_partition(self, df):
        split = splits.chronological_split(df)
        assert len(split.train) + len(split.val) + len(split.test) == len(df)

    def test_held_out_period_is_substantial(self, df):
        """The headline claim depends on the test period being long. Assert it."""
        split = splits.chronological_split(df)
        span = split.test["date"].max() - split.test["date"].min()
        assert span > pd.Timedelta(days=700), f"held-out period is only {span.days} days"

    def test_overlapping_partitions_are_rejected(self, df):
        with pytest.raises(ValueError, match="overlaps"):
            splits.Split(train=df, val=df, test=df)

    def test_empty_partition_is_rejected(self, df):
        with pytest.raises(ValueError, match="empty"):
            splits.Split(train=df.head(0), val=df, test=df)

    def test_sentiment_window_split_stays_inside_coverage(self, df):
        split = splits.sentiment_window_split(df)
        assert split.test["date"].max() <= splits.SENTIMENT_END
