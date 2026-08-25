"""Phase D candidate features: no lookahead, and a few hand-computable cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features


def synthetic(n: int = 200, symbol: str = "TEST", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        }
    )


class TestNoLookahead:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda part: features.rsi(part["close"]),
            lambda part: features.macd_histogram(part["close"]),
            lambda part: features.realized_vol_21d(part["close"]),
            lambda part: features.volume_zscore(part["volume"]),
            lambda part: features.high_low_range(part["high"], part["low"], part["close"]),
        ],
    )
    def test_value_at_i_is_unchanged_by_poisoning_everything_after_i(self, fn):
        part = synthetic()
        i = 100
        before = fn(part).iloc[i]

        poisoned = part.copy()
        poisoned.loc[i + 1 :, ["open", "high", "low", "close", "volume"]] *= 1000.0
        after = fn(poisoned).iloc[i]

        if pd.isna(before):
            assert pd.isna(after)
        else:
            assert after == pytest.approx(before)

    def test_technical_features_bundle_matches_individual_functions(self):
        part = synthetic()
        bundle = features.technical_features(part)
        assert list(bundle.columns) == list(features.TECHNICAL_COLUMNS)
        assert np.allclose(
            bundle["rsi14"].to_numpy(), features.rsi(part["close"]).to_numpy(), equal_nan=True
        )


class TestCrossSectional:
    def test_index_return_is_equal_weight_mean_of_symbol_returns(self):
        dates = pd.bdate_range("2021-01-04", periods=5)
        df = pd.concat(
            [
                pd.DataFrame({"date": dates, "symbol": "A", "close": [100, 101, 102, 101, 103]}),
                pd.DataFrame({"date": dates, "symbol": "B", "close": [50, 50, 51, 52, 52]}),
            ],
            ignore_index=True,
        )
        result = features.cross_sectional_features(df)
        a_ret = df[df["symbol"] == "A"]["close"].pct_change().to_numpy()
        b_ret = df[df["symbol"] == "B"]["close"].pct_change().to_numpy()
        index_ret = (a_ret + b_ret) / 2
        assert np.allclose(result["A"].to_numpy(), a_ret - index_ret, equal_nan=True)

    def test_a_relatively_strong_symbol_has_positive_relative_return(self):
        dates = pd.bdate_range("2021-01-04", periods=3)
        df = pd.concat(
            [
                pd.DataFrame({"date": dates, "symbol": "STRONG", "close": [100, 110, 121]}),
                pd.DataFrame({"date": dates, "symbol": "WEAK", "close": [100, 99, 98]}),
            ],
            ignore_index=True,
        )
        result = features.cross_sectional_features(df)
        assert result["STRONG"].iloc[1] > 0
        assert result["WEAK"].iloc[1] < 0

    def test_no_lookahead_in_the_index_return(self):
        """A future date's close must not change an earlier date's index return."""
        dates = pd.bdate_range("2021-01-04", periods=10)
        rng = np.random.default_rng(0)
        df = pd.concat(
            [
                pd.DataFrame({
                    "date": dates, "symbol": s,
                    "close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 10)),
                })
                for s in ("A", "B", "C")
            ],
            ignore_index=True,
        )
        before = features.cross_sectional_features(df)["A"].iloc[5]

        poisoned = df.copy()
        poisoned.loc[poisoned["date"] > dates[5], "close"] *= 1000.0
        after = features.cross_sectional_features(poisoned)["A"].iloc[5]
        assert after == pytest.approx(before)


class TestCalendarFeatures:
    def test_day_of_week_matches_pandas(self):
        dates = pd.Series(pd.bdate_range("2021-01-04", periods=10))  # starts on a Monday
        result = features.calendar_features(dates)
        assert result["day_of_week"].iloc[0] == 0  # Monday

    def test_month_end_flags_the_last_row_before_the_month_changes(self):
        dates = pd.Series(pd.to_datetime(["2021-01-29", "2021-02-01", "2021-02-02"]))
        result = features.calendar_features(dates)
        assert result["month_end"].tolist() == [1.0, 0.0, 1.0]


class TestMissingIndicator:
    def test_nan_becomes_zero_plus_indicator_one(self):
        series = pd.Series([1.0, np.nan, 3.0])
        result = features.with_missing_indicator(series)
        assert result.shape == (3, 2)
        assert result[1, 0] == 0.0
        assert result[1, 1] == 1.0
        assert result[0, 1] == 0.0

    def test_real_zero_is_not_confused_with_missing(self):
        series = pd.Series([0.0, 1.0])
        result = features.with_missing_indicator(series)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0  # not missing, genuinely zero
