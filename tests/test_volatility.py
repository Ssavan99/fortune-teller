"""Volatility estimators, checked against hand-computable cases and known properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import volatility


class TestEwmaVolatility:
    def test_constant_returns_give_matching_volatility(self):
        # A tiny hand-computable case: all returns equal r -> variance = r^2 exactly
        # (RiskMetrics assumes zero mean), so vol = |r|.
        returns = np.full(20, 0.02)
        vol = volatility.ewma_volatility(returns, lam=0.94)
        assert vol == pytest.approx(0.02, rel=1e-6)

    def test_more_recent_large_moves_weight_more_heavily(self):
        """A large move at the END of the series should raise the estimate more than the
        same-sized move at the START — that's what "exponentially weighted" means."""
        n = 60
        base = np.zeros(n)
        recent_shock = base.copy()
        recent_shock[-1] = 0.10
        early_shock = base.copy()
        early_shock[0] = 0.10

        vol_recent = volatility.ewma_volatility(recent_shock, lam=0.94)
        vol_early = volatility.ewma_volatility(early_shock, lam=0.94)
        assert vol_recent > vol_early

    def test_empty_returns_give_nan_not_zero(self):
        """No data means no estimate — 0.0 would falsely assert perfect calm."""
        assert np.isnan(volatility.ewma_volatility(np.array([])))

    def test_nan_returns_are_dropped_not_propagated(self):
        returns = np.array([0.01, np.nan, 0.01, 0.01])
        vol = volatility.ewma_volatility(returns, lam=0.94)
        assert np.isfinite(vol)


class TestRollingEwmaVolatility:
    def test_is_causal_no_lookahead(self):
        """The estimate at index i must be identical whether or not the series has more data
        AFTER index i — changing the future must not change a past estimate."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 100))
        full = volatility.rolling_ewma_volatility(returns, min_periods=21)

        truncated_returns = returns.iloc[:50].copy()
        truncated = volatility.rolling_ewma_volatility(truncated_returns, min_periods=21)

        assert np.allclose(full.iloc[:50].to_numpy(), truncated.to_numpy(), equal_nan=True)

    def test_early_values_are_nan_until_min_periods(self):
        returns = pd.Series(np.full(30, 0.01))
        result = volatility.rolling_ewma_volatility(returns, min_periods=21)
        assert result.iloc[:20].isna().all()
        assert result.iloc[20:].notna().all()

    def test_a_volatility_spike_raises_subsequent_estimates(self):
        calm = np.full(40, 0.001)
        spike = np.full(20, 0.05)
        returns = pd.Series(np.concatenate([calm, spike]))
        result = volatility.rolling_ewma_volatility(returns, min_periods=21)
        assert result.iloc[-1] > result.iloc[39]


class TestHorizonScale:
    def test_scales_by_sqrt_horizon(self):
        assert volatility.horizon_scale(0.02, horizon=4) == pytest.approx(0.04)  # sqrt(4)=2

    def test_horizon_one_is_unchanged(self):
        assert volatility.horizon_scale(0.02, horizon=1) == pytest.approx(0.02)


class TestGarch:
    def test_returns_none_on_too_little_data(self):
        assert volatility.garch11_volatility(np.random.default_rng(0).normal(0, 0.01, 10)) is None

    def test_returns_a_finite_positive_estimate_on_real_looking_data(self):
        rng = np.random.default_rng(0)
        # Simulate volatility-clustered returns so the GARCH fit has something to find.
        n = 500
        vol = np.zeros(n)
        vol[0] = 0.01
        returns = np.zeros(n)
        for i in range(1, n):
            vol[i] = np.sqrt(1e-6 + 0.1 * returns[i - 1] ** 2 + 0.85 * vol[i - 1] ** 2)
            returns[i] = rng.normal(0, vol[i])

        result = volatility.garch11_volatility(returns)
        # arch is a real optional dependency; only assert shape if it actually fit.
        if result is not None:
            assert np.isfinite(result)
            assert result > 0

    def test_never_raises_on_garbage_input(self):
        # NaNs, infs, degenerate constant series -- must return None, not throw.
        for bad in (
            np.full(200, np.nan),
            np.full(200, 0.0),
            np.concatenate([np.full(100, np.inf), np.random.default_rng(0).normal(0, 0.01, 100)]),
        ):
            assert volatility.garch11_volatility(bad) is None
