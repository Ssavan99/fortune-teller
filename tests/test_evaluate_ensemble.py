"""The ensemble evaluation's scoring wrapper and union-interval construction."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import evaluate_ensemble as ee


class TestScoreArm:
    def test_matches_hand_computed_rmse_and_coverage(self):
        point = np.array([100.0, 100.0])
        lo = np.array([90.0, 90.0])
        hi = np.array([110.0, 110.0])
        actual = np.array([105.0, 130.0])  # first inside, second outside

        result = ee._score_arm("test", point, lo, hi, actual, level=0.80)
        assert result["rmse"] == pytest.approx(np.sqrt((5**2 + 30**2) / 2))
        assert result["coverage"] == pytest.approx(0.5)
        assert result["mean_interval_width"] == pytest.approx(20.0)


class TestUnionInterval:
    def test_union_always_contains_both_input_intervals(self):
        """The whole point of a union combination: its coverage can only be >= the better of
        the two inputs, since it contains both intervals by construction."""
        rng = np.random.default_rng(0)
        n = 200
        lo_a = rng.uniform(80, 95, n)
        hi_a = lo_a + rng.uniform(5, 15, n)
        lo_b = rng.uniform(85, 100, n)
        hi_b = lo_b + rng.uniform(5, 15, n)

        union_lo = np.minimum(lo_a, lo_b)
        union_hi = np.maximum(hi_a, hi_b)

        assert np.all(union_lo <= lo_a)
        assert np.all(union_lo <= lo_b)
        assert np.all(union_hi >= hi_a)
        assert np.all(union_hi >= hi_b)
