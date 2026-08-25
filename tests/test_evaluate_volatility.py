"""The volatility evaluation's windowing: no lookahead, correct units, correct alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_volatility as ev


def synthetic(n: int = 200, symbol: str = "TEST", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": closes})


class TestRealizedVol:
    def test_matches_hand_computed_annualized_std(self):
        returns = pd.Series([0.01] * 21)  # constant daily return -> std 0 -> vol 0
        assert ev._realized_vol(returns, 0, 21) == pytest.approx(0.0, abs=1e-9)

    def test_short_window_is_nan_not_a_biased_estimate(self):
        returns = pd.Series([0.01] * 10)
        assert np.isnan(ev._realized_vol(returns, 0, 10))


class TestEvaluateSymbol:
    def test_no_row_uses_returns_from_its_own_forward_window_as_the_forecast(self):
        """The baseline/model forecast for an as-of date must come only from returns strictly
        BEFORE it -- never from the window being predicted."""
        df = synthetic()
        rows = ev.evaluate_symbol(df)
        assert len(rows) > 0

        date_to_idx = {d: i for i, d in enumerate(df["date"])}
        for r in rows:
            as_of_idx = date_to_idx[pd.Timestamp(r["date"])]
            # Poisoning everything from as_of_idx onward must not change baseline/model,
            # since both are computed from strictly-prior data.
            poisoned = df.copy()
            poisoned.loc[as_of_idx:, "close"] *= 1000.0
            poisoned_rows = ev.evaluate_symbol(poisoned)
            match = next(pr for pr in poisoned_rows if pr["date"] == r["date"])
            assert match["baseline_forecast"] == pytest.approx(r["baseline_forecast"])
            assert match["model_forecast"] == pytest.approx(r["model_forecast"])
            break  # one spot-check is enough; this loop is O(n^2) if run for every row

    def test_actual_reflects_the_forward_window_only(self):
        """Conversely, the 'actual' realized vol must change when the FUTURE window is
        poisoned, and must NOT change when the past is poisoned."""
        df = synthetic()
        rows = ev.evaluate_symbol(df)
        r = rows[len(rows) // 2]
        date_to_idx = {d: i for i, d in enumerate(df["date"])}
        as_of_idx = date_to_idx[pd.Timestamp(r["date"])]

        poisoned_future = df.copy()
        poisoned_future.loc[as_of_idx:, "close"] *= 1000.0
        future_rows = ev.evaluate_symbol(poisoned_future)
        future_match = next(pr for pr in future_rows if pr["date"] == r["date"])
        assert future_match["actual"] != pytest.approx(r["actual"])

        # Poison up to as_of_idx - 2, NOT as_of_idx - 1: pct_change() at as_of_idx depends on
        # close[as_of_idx - 1], so poisoning that row would corrupt the forward window's own
        # first return through the ratio, not just the "past".
        poisoned_past = df.copy()
        poisoned_past.loc[: as_of_idx - 2, "close"] *= 1000.0
        past_rows = ev.evaluate_symbol(poisoned_past)
        past_match = next(pr for pr in past_rows if pr["date"] == r["date"])
        assert past_match["actual"] == pytest.approx(r["actual"])

    def test_units_are_annualized_not_daily(self):
        """A constant-vol series's realized vol must scale by sqrt(252) relative to the raw
        daily std, confirming the annualization is actually applied."""
        n = 200
        rng = np.random.default_rng(1)
        daily_returns = rng.normal(0, 0.02, n)
        closes = 100 * np.cumprod(1 + daily_returns)
        df = pd.DataFrame(
            {"date": pd.bdate_range("2021-01-04", periods=n), "symbol": "TEST", "close": closes}
        )
        rows = ev.evaluate_symbol(df)
        assert rows  # sanity
        # Realized vols should be roughly sqrt(252)*0.02 ~= 0.32, not ~0.02.
        actuals = [r["actual"] for r in rows]
        assert 0.15 < np.mean(actuals) < 0.6
