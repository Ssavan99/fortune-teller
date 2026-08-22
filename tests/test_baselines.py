"""Baseline correctness, on synthetic series where the right answer is known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import baselines, metrics


def frame(closes, symbol="TEST", start="2021-01-04") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "symbol": symbol, "close": np.asarray(closes, float)})


class TestPersistence:
    def test_prediction_is_the_previous_close(self):
        df = frame([10, 20, 30, 40])
        fc = baselines.persistence(df, "TEST", df.iloc[1:])
        assert list(fc.y_pred) == [10, 20, 30]
        assert list(fc.y_true) == [20, 30, 40]

    def test_is_exact_on_a_flat_series(self):
        df = frame([50.0] * 10)
        fc = baselines.persistence(df, "TEST", df.iloc[1:])
        assert metrics.rmse(fc.y_true, fc.y_pred) == 0.0

    def test_error_equals_the_daily_change_on_a_ramp(self):
        """A series rising by exactly 2 a day gives persistence an error of exactly 2."""
        df = frame(np.arange(100, 120, 2.0))
        fc = baselines.persistence(df, "TEST", df.iloc[1:])
        assert metrics.mae(fc.y_true, fc.y_pred) == pytest.approx(2.0)
        assert metrics.rmse(fc.y_true, fc.y_pred) == pytest.approx(2.0)

    def test_first_row_of_the_period_is_kept_using_history_before_it(self):
        """The previous close comes from outside the period, so no row is wasted."""
        df = frame([10, 20, 30, 40])
        period = df.iloc[2:]
        fc = baselines.persistence(df, "TEST", period)
        assert len(fc.y_true) == 2
        assert fc.y_pred[0] == 20

    def test_makes_no_directional_call(self):
        df = frame([10, 20, 30, 40])
        fc = baselines.persistence(df, "TEST", df.iloc[1:])
        assert metrics.directional_accuracy(fc.y_true, fc.y_pred, fc.last_close) is None


class TestDrift:
    def test_is_exact_on_a_constant_ramp(self):
        """Trained on a series rising by 2 a day, drift should predict the ramp perfectly."""
        closes = np.arange(100, 160, 2.0)
        df = frame(closes)
        train = df.iloc[:20]
        fc = baselines.drift(df, "TEST", df.iloc[20:], train)
        assert metrics.rmse(fc.y_true, fc.y_pred) == pytest.approx(0.0, abs=1e-9)

    def test_beats_persistence_when_there_is_a_trend(self):
        closes = np.arange(100, 160, 2.0)
        df = frame(closes)
        train, period = df.iloc[:20], df.iloc[20:]
        d = baselines.drift(df, "TEST", period, train)
        p = baselines.persistence(df, "TEST", period)
        assert metrics.rmse(d.y_true, d.y_pred) < metrics.rmse(p.y_true, p.y_pred)

    def test_drift_is_estimated_on_training_only(self):
        """Training is flat, the later period ramps. A leak-free drift term must be zero."""
        closes = np.concatenate([np.full(30, 100.0), np.arange(100, 130, 1.0)])
        df = frame(closes)
        train, period = df.iloc[:30], df.iloc[30:]
        fc = baselines.drift(df, "TEST", period, train)
        p = baselines.persistence(df, "TEST", period)
        assert np.allclose(fc.y_pred, p.y_pred), "drift term picked up the held-out trend"


class TestAutoregressive:
    def test_recovers_a_known_return_process(self):
        """Returns that alternate +1%/−1% are perfectly predictable from one lag."""
        rng = np.random.default_rng(0)
        rets = np.where(np.arange(400) % 2 == 0, 0.01, -0.01)
        closes = 100 * np.cumprod(1 + rets)
        df = frame(closes)
        train, period = df.iloc[:300], df.iloc[300:]

        ar = baselines.autoregressive(df, "TEST", period, train, order=2)
        p = baselines.persistence(df, "TEST", period)
        assert metrics.rmse(ar.y_true, ar.y_pred) < metrics.rmse(p.y_true, p.y_pred)
        assert rng is not None  # seed fixture kept for future noise variants

    def test_does_not_beat_persistence_on_a_random_walk(self):
        """On unpredictable returns AR should have no real edge over persistence."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 900)
        closes = 100 * np.cumprod(1 + rets)
        df = frame(closes)
        train, period = df.iloc[:700], df.iloc[700:]

        ar = baselines.autoregressive(df, "TEST", period, train, order=5)
        p = baselines.persistence(df, "TEST", period)
        skill = metrics.skill_score(
            metrics.rmse(ar.y_true, ar.y_pred), metrics.rmse(p.y_true, p.y_pred)
        )
        assert abs(skill) < 0.10, f"AR found {skill:.3f} skill in a random walk"

    def test_predictions_are_finite(self):
        rng = np.random.default_rng(7)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 500))
        df = frame(closes)
        fc = baselines.autoregressive(df, "TEST", df.iloc[400:], df.iloc[:400])
        assert np.all(np.isfinite(fc.y_pred))

    def test_alignment_matches_persistence(self):
        """AR and persistence must score the same rows or the comparison is invalid."""
        rng = np.random.default_rng(3)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        df = frame(closes)
        train, period = df.iloc[:300], df.iloc[300:]
        ar = baselines.autoregressive(df, "TEST", period, train)
        p = baselines.persistence(df, "TEST", period)
        assert list(ar.dates) == list(p.dates)
        assert np.allclose(ar.y_true, p.y_true)
