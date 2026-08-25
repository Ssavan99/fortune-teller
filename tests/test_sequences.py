"""Sequence construction: no lookahead, correct alignment, correct dollar inversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import sequences
from src.data import prices, splits


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return prices.load()


@pytest.fixture(scope="module")
def split(df):
    return splits.chronological_split(df)


def synthetic(n: int = 400, symbol: str = "TEST") -> pd.DataFrame:
    """A frame whose close on day i is exactly i, so alignment errors are unmissable."""
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": np.arange(n, dtype=float),
            "high": np.arange(n, dtype=float) + 0.5,
            "low": np.arange(n, dtype=float) - 0.5,
            "close": np.arange(n, dtype=float),
            "volume": np.full(n, 1000.0),
        }
    )


class TestNoLookahead:
    def test_target_is_one_step_after_the_window(self):
        """close[i] == i, so prev_close must be exactly one less than y_close, everywhere."""
        df = synthetic()
        sp = splits.Split(train=df.iloc[:200], val=df.iloc[200:300], test=df.iloc[300:])
        built = sequences.build(df, sp, lookback=10, target="level")
        for part in built.values():
            assert np.all(part.y_close - part.prev_close == 1.0)

    def test_window_ends_strictly_before_the_target(self):
        """The scaled close in the last window slot must equal the previous day's close."""
        df = synthetic()
        sp = splits.Split(train=df.iloc[:200], val=df.iloc[200:300], test=df.iloc[300:])
        built = sequences.build(df, sp, lookback=10, target="level")
        test = built["test"]

        lo = float(df.iloc[:200]["close"].min())
        hi = float(df.iloc[:200]["close"].max())
        last_slot = test.x[:, -1, sequences.CLOSE_INDEX] * (hi - lo) + lo
        assert np.allclose(last_slot, test.prev_close)

    def test_no_training_target_falls_after_the_train_cutoff(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        assert pd.Series(built["train"].dates).max() <= splits.TRAIN_END

    def test_no_validation_target_falls_in_the_test_period(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        assert pd.Series(built["val"].dates).max() <= splits.VAL_END

    def test_test_targets_all_fall_after_the_val_cutoff(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        assert pd.Series(built["test"].dates).min() > splits.VAL_END

    def test_partitions_share_no_dates_within_a_symbol(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        keys = {}
        for name, part in built.items():
            keys[name] = set(
                zip(part.symbols.tolist(), pd.Series(part.dates).astype(str), strict=True)
            )
        assert not keys["train"] & keys["val"]
        assert not keys["val"] & keys["test"]
        assert not keys["train"] & keys["test"]


class TestTargets:
    def test_return_target_matches_the_actual_return(self):
        df = synthetic()
        sp = splits.Split(train=df.iloc[:200], val=df.iloc[200:300], test=df.iloc[300:])
        built = sequences.build(df, sp, lookback=10, target="return")
        part = built["test"]
        assert np.allclose(part.y, part.y_close / part.prev_close - 1.0)

    def test_level_target_inverts_to_the_true_close(self, df, split):
        built = sequences.build(df, split, lookback=20, target="level")
        inverter = sequences.ScaledInverter(df, split)
        recovered = sequences.to_dollars(built["test"].y, built["test"], "level", inverter)
        assert np.allclose(recovered, built["test"].y_close, atol=1e-6)

    def test_return_target_inverts_to_the_true_close(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        recovered = sequences.to_dollars(built["test"].y, built["test"], "return")
        assert np.allclose(recovered, built["test"].y_close, atol=1e-6)

    def test_unknown_target_is_rejected(self):
        df = synthetic()
        sp = splits.Split(train=df.iloc[:200], val=df.iloc[200:300], test=df.iloc[300:])
        with pytest.raises(ValueError, match="unknown target"):
            sequences.build(df, sp, lookback=10, target="nonsense")

    def test_level_inversion_requires_an_inverter(self, df, split):
        built = sequences.build(df, split, lookback=20, target="level")
        with pytest.raises(ValueError, match="require a ScaledInverter"):
            sequences.to_dollars(built["test"].y, built["test"], "level")

    def test_zero_horizon_is_rejected(self):
        """horizon=0 would let a window's target coincide with its own last slot."""
        df = synthetic()
        sp = splits.Split(train=df.iloc[:200], val=df.iloc[200:300], test=df.iloc[300:])
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            sequences.build(df, sp, lookback=10, target="return", horizon=0)


class TestShapes:
    def test_shapes_are_consistent(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        for part in built.values():
            assert part.x.shape[1] == 20
            assert part.x.shape[2] == len(sequences.FEATURES)
            assert len(part.x) == len(part.y) == len(part.y_close) == len(part.prev_close)

    def test_every_symbol_appears_in_every_partition(self, df, split):
        built = sequences.build(df, split, lookback=20, target="return")
        for part in built.values():
            assert set(part.symbols) == set(prices.TICKERS)

    def test_lookback_changes_the_window_not_the_count_much(self, df, split):
        short = sequences.build(df, split, lookback=5, target="return")
        long_ = sequences.build(df, split, lookback=40, target="return")
        assert short["test"].x.shape[1] == 5
        assert long_["test"].x.shape[1] == 40
        # Only the earliest windows are lost, and only from the training partition.
        assert len(short["test"]) == len(long_["test"])
