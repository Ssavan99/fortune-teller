"""Prediction intervals: no lookahead in the quantiles, and the scoring rule can't be gamed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import intervals, metrics, sequences
from src.data import splits


def synthetic(n: int = 400, symbol: str = "TEST", scale: float = 1.0) -> pd.DataFrame:
    """A frame whose close on day i is exactly ``scale * i``, so alignment errors are unmissable."""
    dates = pd.bdate_range("2021-01-04", periods=n)
    closes = np.arange(n, dtype=float) * scale + scale
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(n, 1000.0),
        }
    )


class TestWindowNeverTouchesItsTarget:
    def test_window_never_touches_its_target_at_any_horizon(self):
        df = synthetic()
        for horizon in (1, 5, 21):
            sp = splits.Split(train=df.iloc[:250], val=df.iloc[250:350], test=df.iloc[350:])
            built = sequences.build(df, sp, lookback=10, target="level", horizon=horizon)
            for part in built.values():
                # The last window slot's dollar close is exactly one session before the
                # target, i.e. prev_close.
                lo = float(sp.train["close"].min())
                hi = float(sp.train["close"].max())
                last_slot = part.x[:, -1, sequences.CLOSE_INDEX] * (hi - lo) + lo
                assert np.allclose(last_slot, part.prev_close)

                # Target date is exactly `horizon` trading sessions after the window's
                # last date. The synthetic frame is built from a business-day range with
                # no gaps, so this is a plain index difference.
                date_to_idx = {d: i for i, d in enumerate(df["date"])}
                window_last_idx = np.array([date_to_idx[d] for d in pd.Series(part.dates)])
                # part.prev_close encodes the window's last close; recover its index via
                # the close-to-index map since closes are unique and monotone here.
                close_to_idx = {c: i for i, c in enumerate(df["close"])}
                prev_idx = np.array([close_to_idx[c] for c in part.prev_close])
                assert np.all(window_last_idx - prev_idx == horizon)


class TestCoverage:
    def test_coverage_is_one_when_interval_spans_everything(self):
        y = np.array([1.0, 5.0, -3.0, 100.0])
        lo = np.full(4, -1e9)
        hi = np.full(4, 1e9)
        assert metrics.coverage(y, lo, hi) == 1.0

    def test_coverage_is_zero_when_interval_excludes_everything(self):
        y = np.array([1.0, 5.0, -3.0, 100.0])
        lo = y + 10.0
        hi = y + 20.0
        assert metrics.coverage(y, lo, hi) == 0.0

    def test_coverage_matches_hand_computed_case(self):
        y = np.arange(10, dtype=float)
        lo = np.zeros(10)
        hi = np.full(10, 6.5)  # covers 0..6, i.e. 7 of 10 points
        assert metrics.coverage(y, lo, hi) == pytest.approx(0.7)


class TestIntervalScore:
    def test_interval_score_penalises_width(self):
        y = np.array([10.0, 10.0, 10.0])
        tight_lo, tight_hi = np.full(3, 9.0), np.full(3, 11.0)
        wide_lo, wide_hi = np.full(3, -0.0), np.full(3, 20.0)
        tight = metrics.interval_score(y, tight_lo, tight_hi, level=0.80)
        wide = metrics.interval_score(y, wide_lo, wide_hi, level=0.80)
        assert tight < wide

    def test_interval_score_penalises_misses(self):
        y = np.array([10.0])
        lo, hi = np.array([8.0]), np.array([9.0])  # same width, y is outside on both sides
        hit_lo, hit_hi = np.array([9.0]), np.array([11.0])
        miss = metrics.interval_score(y, lo, hi, level=0.80)
        hit = metrics.interval_score(y, hit_lo, hit_hi, level=0.80)
        assert miss > hit

    def test_interval_score_matches_hand_computed_case(self):
        # level=0.80 -> alpha=0.20 -> 2/alpha = 10
        # y=12, lo=10, hi=11 -> width=1, y above hi by 1 -> IS = 1 + 10*1 = 11
        y = np.array([12.0])
        lo, hi = np.array([10.0]), np.array([11.0])
        assert metrics.interval_score(y, lo, hi, level=0.80) == pytest.approx(11.0)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="no observations"):
            metrics.interval_score([], [], [], level=0.80)


class TestResidualQuantiles:
    def test_quantiles_come_from_validation_only(self):
        df = synthetic(n=500)
        sp = splits.Split(train=df.iloc[:250], val=df.iloc[250:400], test=df.iloc[400:])
        built = sequences.build(df, sp, lookback=10, target="return", horizon=1)
        val = built["val"]
        y_pred = val.prev_close  # a stand-in "prediction" (persistence)
        q_before = intervals.residual_quantiles(
            val.y_close, y_pred, val.prev_close, val.symbols, level=0.80
        )

        # Plant an extreme close only in the test-period rows and rebuild.
        poisoned = df.copy()
        poisoned.loc[poisoned["date"] > sp.val["date"].max(), "close"] *= 1000.0
        poisoned.loc[poisoned["date"] > sp.val["date"].max(), "open"] *= 1000.0
        poisoned.loc[poisoned["date"] > sp.val["date"].max(), "high"] *= 1000.0
        poisoned.loc[poisoned["date"] > sp.val["date"].max(), "low"] *= 1000.0
        sp2 = splits.Split(
            train=poisoned.iloc[:250], val=poisoned.iloc[250:400], test=poisoned.iloc[400:]
        )
        built2 = sequences.build(poisoned, sp2, lookback=10, target="return", horizon=1)
        val2 = built2["val"]
        q_after = intervals.residual_quantiles(
            val2.y_close, val2.prev_close, val2.prev_close, val2.symbols, level=0.80
        )

        assert val.y_close.tolist() == val2.y_close.tolist()
        assert q_before["TEST"] == pytest.approx(q_after["TEST"])

    def test_unknown_symbol_falls_back_to_pooled_quantiles(self):
        y_true = np.array([100.0, 102.0, 98.0, 105.0])
        y_pred = np.array([100.0, 100.0, 100.0, 100.0])
        prev_close = np.array([100.0, 100.0, 100.0, 100.0])
        symbols = np.array(["A", "A", "B", "B"])
        q = intervals.residual_quantiles(y_true, y_pred, prev_close, symbols, level=0.80)

        applied = intervals.apply(
            y_pred=np.array([50.0]),
            prev_close=np.array([50.0]),
            symbols=np.array(["UNSEEN"]),
            quantiles=q,
            level=0.80,
        )
        expected_lo = 50.0 + q["__pooled__"][0] * 50.0
        expected_hi = 50.0 + q["__pooled__"][1] * 50.0
        assert applied.lo[0] == pytest.approx(expected_lo)
        assert applied.hi[0] == pytest.approx(expected_hi)

    def test_relative_residuals_are_scale_free(self):
        rng = np.random.default_rng(0)
        n = 200
        pct_errors = rng.normal(0, 0.02, size=n)

        prev_close_cheap = np.full(n, 20.0)
        prev_close_pricey = np.full(n, 200.0)
        y_pred_cheap = prev_close_cheap.copy()
        y_pred_pricey = prev_close_pricey.copy()
        y_true_cheap = prev_close_cheap * (1 + pct_errors)
        y_true_pricey = prev_close_pricey * (1 + pct_errors)

        symbols = np.full(n, "X")
        q_cheap = intervals.residual_quantiles(
            y_true_cheap, y_pred_cheap, prev_close_cheap, symbols, level=0.80
        )
        q_pricey = intervals.residual_quantiles(
            y_true_pricey, y_pred_pricey, prev_close_pricey, symbols, level=0.80
        )
        # Identical percentage errors -> identical relative quantiles regardless of price scale.
        assert q_cheap["X"] == pytest.approx(q_pricey["X"], rel=1e-9)

        interval_cheap = intervals.apply(
            y_pred_cheap[:1], prev_close_cheap[:1], symbols[:1], q_cheap, level=0.80
        )
        interval_pricey = intervals.apply(
            y_pred_pricey[:1], prev_close_pricey[:1], symbols[:1], q_pricey, level=0.80
        )
        width_cheap = interval_cheap.hi[0] - interval_cheap.lo[0]
        width_pricey = interval_pricey.hi[0] - interval_pricey.lo[0]
        assert width_pricey == pytest.approx(width_cheap * 10.0, rel=1e-9)

    def test_apply_rejects_a_mismatched_level(self):
        y_true = np.array([100.0, 102.0, 98.0, 105.0])
        y_pred = np.full(4, 100.0)
        prev_close = np.full(4, 100.0)
        symbols = np.array(["A", "A", "A", "A"])
        q = intervals.residual_quantiles(y_true, y_pred, prev_close, symbols, level=0.80)
        with pytest.raises(ValueError, match="must match"):
            intervals.apply(y_pred[:1], prev_close[:1], symbols[:1], q, level=0.95)

    def test_apply_requires_pooled_fallback(self):
        with pytest.raises(ValueError, match="__pooled__"):
            intervals.apply(
                y_pred=np.array([1.0]),
                prev_close=np.array([1.0]),
                symbols=np.array(["A"]),
                quantiles={"A": (-0.1, 0.1)},
            )
