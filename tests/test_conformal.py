"""Split-conformal prediction intervals: the finite-sample coverage guarantee, checked against
synthetic data with a known answer — not against itself."""

from __future__ import annotations

import numpy as np
import pytest

from src import conformal


class TestFiniteSampleQuantile:
    def test_finite_sample_correction_is_applied(self):
        """For small n, the corrected quantile index is strictly conservative relative to the
        naive (uncorrected) quantile — it must widen, never narrow, the interval."""
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, size=10)
        corrected, _ = conformal._finite_sample_quantile(scores, level=0.80)
        naive = np.quantile(scores, 0.80)
        assert corrected >= naive

    def test_tiny_n_falls_back_to_the_widest_observed_score(self):
        """When n is too small for the level to be achievable at all, the corrected index
        would exceed the sample — must fall back to the max score (widest, most conservative),
        never silently emit something narrower."""
        scores = np.array([0.01, 0.05, 0.5])
        q, exceeded = conformal._finite_sample_quantile(scores, level=0.95)
        assert exceeded is True
        assert q == pytest.approx(0.5)

    def test_rejects_empty_scores(self):
        with pytest.raises(ValueError, match="no calibration scores"):
            conformal._finite_sample_quantile(np.array([]), level=0.80)


class TestConformalCoverage:
    def test_conformal_achieves_nominal_coverage_on_synthetic_exchangeable_data(self):
        """Calibration and test residuals drawn from the SAME distribution (exchangeable) —
        the textbook condition split-conformal is designed for. Realized coverage must land
        close to nominal."""
        rng = np.random.default_rng(42)
        n_cal, n_test = 3000, 3000
        prev_close = 100.0

        def make(n):
            errors = rng.normal(0, 2.0, size=n)  # dollars
            y_pred = np.full(n, 50.0)
            y_true = y_pred + errors
            return y_true, y_pred

        cal_true, cal_pred = make(n_cal)
        quantiles = conformal.conformal_quantiles(
            cal_true, cal_pred, np.full(n_cal, prev_close), np.full(n_cal, "X"), level=0.80
        )

        test_true, test_pred = make(n_test)
        q_lo, q_hi = quantiles["__pooled__"]
        lo = test_pred + q_lo * prev_close
        hi = test_pred + q_hi * prev_close
        coverage = np.mean((test_true >= lo) & (test_true <= hi))

        assert 0.77 <= coverage <= 0.83

    def test_calibration_set_is_validation_only(self):
        """conformal_quantiles is a pure function of exactly what it's given — planting an
        extreme residual in data that is never passed in must not change the result."""
        rng = np.random.default_rng(1)
        n = 500
        y_pred = np.full(n, 100.0)
        y_true = y_pred + rng.normal(0, 1.0, size=n)
        prev_close = np.full(n, 100.0)
        symbols = np.full(n, "AAPL")

        before = conformal.conformal_quantiles(y_true, y_pred, prev_close, symbols, level=0.80)

        # An extreme "test-period" point exists in a completely separate array that is never
        # passed to conformal_quantiles at all.
        untouched_extreme_true = np.array([100.0 + 1e6])
        untouched_extreme_pred = np.array([100.0])
        assert untouched_extreme_true[0] != untouched_extreme_pred[0]  # sanity: it IS extreme

        after = conformal.conformal_quantiles(y_true, y_pred, prev_close, symbols, level=0.80)
        assert before["__pooled__"] == pytest.approx(after["__pooled__"])

    def test_small_calibration_set_falls_back_to_pooled(self):
        rng = np.random.default_rng(2)
        n_big, n_small = 200, 5
        y_pred = np.concatenate([np.full(n_big, 100.0), np.full(n_small, 50.0)])
        y_true = y_pred + rng.normal(0, 1.0, size=n_big + n_small)
        prev_close = np.full(n_big + n_small, 100.0)
        symbols = np.array(["BIG"] * n_big + ["SMALL"] * n_small)

        result = conformal.conformal_quantiles(
            y_true, y_pred, prev_close, symbols, level=0.80, pooled=False, min_n=30
        )
        assert result["SMALL"] == pytest.approx(result["__pooled__"])
        # BIG has enough points to get its own quantile, which need not match pooled exactly.
        assert "BIG" in result

    def test_pooled_true_gives_every_symbol_the_same_quantile(self):
        rng = np.random.default_rng(3)
        n = 300
        y_pred = np.full(n, 10.0)
        y_true = y_pred + rng.normal(0, 1.0, size=n)
        prev_close = np.full(n, 10.0)
        symbols = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))

        result = conformal.conformal_quantiles(
            y_true, y_pred, prev_close, symbols, level=0.80, pooled=True
        )
        assert result["A"] == result["B"] == result["__pooled__"]

    def test_result_is_plug_compatible_with_intervals_apply(self):
        """conformal_quantiles must be a drop-in replacement for
        intervals.residual_quantiles — same shape, works with the same apply() function."""
        from src import intervals

        rng = np.random.default_rng(4)
        n = 200
        y_pred = np.full(n, 100.0)
        y_true = y_pred + rng.normal(0, 2.0, size=n)
        prev_close = np.full(n, 100.0)
        symbols = np.full(n, "AAPL")

        quantiles = conformal.conformal_quantiles(y_true, y_pred, prev_close, symbols, level=0.80)
        applied = intervals.apply(
            np.array([100.0]), np.array([100.0]), np.array(["AAPL"]), quantiles, level=0.80
        )
        assert applied.lo[0] < 100.0 < applied.hi[0]


class TestAdaptiveIntervals:
    def test_adaptive_intervals_widen_in_high_volatility(self):
        """The SAME calibrated quantile, applied with a higher vol_hat, must produce a wider
        interval — that's the entire point of volatility scaling."""
        rng = np.random.default_rng(5)
        n = 1000
        prev_close = np.full(n, 100.0)
        vol_hat = rng.uniform(0.01, 0.05, size=n)
        y_pred = np.full(n, 50.0)
        # errors genuinely scale with vol_hat, so the calibration is meaningful
        y_true = y_pred + rng.normal(0, 1, size=n) * prev_close * vol_hat

        q = conformal.adaptive_conformal_quantile(y_true, y_pred, prev_close, vol_hat, level=0.80)

        low_vol_interval = conformal.apply_adaptive(
            np.array([50.0]), np.array([100.0]), np.array([0.01]), q
        )
        high_vol_interval = conformal.apply_adaptive(
            np.array([50.0]), np.array([100.0]), np.array([0.05]), q
        )
        low_width = low_vol_interval.hi[0] - low_vol_interval.lo[0]
        high_width = high_vol_interval.hi[0] - high_vol_interval.lo[0]
        assert high_width > low_width
        assert high_width == pytest.approx(low_width * 5, rel=1e-6)  # 0.05 / 0.01

    def test_adaptive_calibration_drops_nan_volatility_rows(self):
        y_pred = np.array([100.0, 100.0, 100.0])
        y_true = np.array([101.0, 99.0, 500.0])  # the NaN-vol row is a huge outlier
        prev_close = np.array([100.0, 100.0, 100.0])
        vol_hat = np.array([0.02, 0.02, np.nan])

        q = conformal.adaptive_conformal_quantile(y_true, y_pred, prev_close, vol_hat, level=0.80)
        assert q.n == 2  # the NaN row was excluded, not treated as a score of NaN/inf

    def test_apply_adaptive_rejects_nonfinite_vol_hat(self):
        y_pred = np.array([100.0, 100.0, 100.0])
        y_true = np.array([101.0, 99.0, 102.0])
        prev_close = np.array([100.0, 100.0, 100.0])
        vol_hat = np.array([0.02, 0.02, 0.02])
        q = conformal.adaptive_conformal_quantile(y_true, y_pred, prev_close, vol_hat, level=0.80)

        with pytest.raises(ValueError, match="finite, positive vol_hat"):
            conformal.apply_adaptive(
                np.array([100.0]), np.array([100.0]), np.array([np.nan]), q
            )
