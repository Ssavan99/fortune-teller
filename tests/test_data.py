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
        """Identical *date sets*, not merely identical row counts.

        ``fetch`` drops bars per symbol, so a vendor hole in one ticker yields a ragged
        panel that downstream pooling code would assume is aligned. Equal counts on
        different calendars would pass a count check and still be ragged.
        """
        calendars = {sym: frozenset(part["date"]) for sym, part in df.groupby("symbol")}
        assert len(set(calendars.values())) == 1, "symbols are on different trading calendars"

    def test_no_large_discontinuities(self, df):
        """No overnight move larger than any real move in this universe.

        The largest genuine single-day moves here are META −26.4% and BABA +36.8%, so a
        bound at −40%/+60% sits clear of real market behaviour while catching an
        unadjusted 2:1 split (−50%) or anything larger.
        """
        for symbol, part in df.groupby("symbol"):
            ratio = part["close"].to_numpy()[1:] / part["close"].to_numpy()[:-1]
            assert ratio.min() > 0.60, f"{symbol} has a drop larger than any real move"
            assert ratio.max() < 1.60, f"{symbol} has a jump larger than any real move"

    def test_no_exact_split_ratios(self, df):
        """Sharper guard: an unadjusted split lands near an exact simple fraction.

        Catches ratios the bound above cannot — notably a 3:2 at 0.667, which sits inside
        the −40% floor. The tolerance is 3%, not something tighter, because a split boundary
        lands at the exact fraction *times that day's real move*: an injected 2:1 in this
        universe measures 0.5037, not 0.5000.

        The widest band this opens is 2:3 at [0.647, 0.687]. The largest genuine drop in the
        universe is META at 0.736, so the bands stay clear of real market behaviour.
        """
        split_ratios = [1 / n for n in (2, 3, 4, 5, 10, 20)] + [2 / 3, 3 / 2]
        for symbol, part in df.groupby("symbol"):
            ratio = part["close"].to_numpy()[1:] / part["close"].to_numpy()[:-1]
            for target in split_ratios:
                hits = abs(ratio - target) < target * 0.03
                assert not hits.any(), f"{symbol} has a bar within 3% of a {target:.4f} split"


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

    def test_coverage_reaches_into_the_held_out_period(self, sent):
        """The snapshot does overlap the held-out period — which is why B is clamped.

        Stated as a fact about the data rather than a bound, so it cannot rot into a
        vacuously loose assertion. The protection lives in the next test.
        """
        assert sent["date"].max() > splits.VAL_END

    def test_experiment_b_is_clamped_out_of_the_held_out_period(self):
        """The guard that matters: B's window must stop at or before the main val cutoff."""
        assert splits.SENTIMENT_END <= splits.VAL_END


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

    def test_train_val_overlap_is_rejected(self, df):
        with pytest.raises(ValueError, match="train overlaps val"):
            splits.Split(train=df, val=df, test=df.tail(1))

    def test_val_test_overlap_is_rejected(self, df):
        """Exercised separately: the train/val check fires first and would mask this one."""
        split = splits.chronological_split(df)
        # val is widened to swallow the test period; train stays legitimately earlier, so
        # the train/val check passes and the val/test check is the one that must fire.
        widened_val = pd.concat([split.val, split.test])
        with pytest.raises(ValueError, match="val overlaps test"):
            splits.Split(train=split.train, val=widened_val, test=split.test)

    def test_empty_partition_is_rejected(self, df):
        with pytest.raises(ValueError, match="empty"):
            splits.Split(train=df.head(0), val=df, test=df)

    def test_sentiment_window_split_stays_inside_coverage(self, df):
        split = splits.sentiment_window_split(df)
        assert split.test["date"].max() <= splits.SENTIMENT_END
