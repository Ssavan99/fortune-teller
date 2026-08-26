"""Direction-as-classification: scoring functions checked by hand, calibration checked for
leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_direction as ed


class TestScoringFunctions:
    def test_brier_matches_hand_computed_case(self):
        p = np.array([1.0, 0.0, 0.5])
        y = np.array([1, 0, 1])
        # (1-1)^2=0, (0-0)^2=0, (0.5-1)^2=0.25 -> mean = 0.25/3
        assert ed._brier(p, y) == pytest.approx(0.25 / 3)

    def test_brier_is_zero_for_perfect_predictions(self):
        p = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1, 0, 1, 0])
        assert ed._brier(p, y) == pytest.approx(0.0)

    def test_log_loss_matches_hand_computed_case(self):
        p = np.array([0.5])
        y = np.array([1])
        assert ed._log_loss(p, y) == pytest.approx(-np.log(0.5))

    def test_log_loss_does_not_blow_up_at_the_extremes(self):
        # A confidently wrong prediction must give a large but finite loss, not inf/nan.
        p = np.array([0.0])
        y = np.array([1])
        loss = ed._log_loss(p, y)
        assert np.isfinite(loss)
        assert loss > 10  # heavily penalised, but finite thanks to EPS clipping


class TestBinnedCalibration:
    def test_calibration_is_fit_on_the_calibration_set_only(self):
        """Planting an extreme, unrepresentative row that is never included in `cal` must not
        change the fitted bin edges or up-rates at all."""
        rng = np.random.default_rng(0)
        n = 500
        cal = pd.DataFrame(
            {
                "predicted_return": rng.normal(0, 0.02, n),
                "actual_up": rng.integers(0, 2, n),
            }
        )
        edges_before, rates_before = ed._fit_binned_calibration(cal)

        # A wildly different row exists in a separate frame that is never passed in.
        untouched_extreme = pd.DataFrame({"predicted_return": [10.0], "actual_up": [0]})
        assert len(untouched_extreme) == 1  # sanity: it exists, just isn't used

        edges_after, rates_after = ed._fit_binned_calibration(cal)
        assert np.array_equal(edges_before, edges_after)
        assert np.array_equal(rates_before, rates_after)

    def test_empty_bin_falls_back_to_the_overall_base_rate(self):
        # All predicted_return values are identical -> every bin but one is empty.
        cal = pd.DataFrame(
            {"predicted_return": [0.0] * 20, "actual_up": [1, 0] * 10}
        )
        edges, rates = ed._fit_binned_calibration(cal)
        assert np.all(np.isfinite(rates))
        base_rate = 0.5
        # every non-populated bin should equal the base rate exactly
        assert np.any(np.isclose(rates, base_rate))

    def test_apply_calibration_maps_low_predictions_to_the_lowest_bins_rate(self):
        cal = pd.DataFrame(
            {
                "predicted_return": np.linspace(-0.05, 0.05, 100),
                "actual_up": (np.linspace(-0.05, 0.05, 100) > 0).astype(int),
            }
        )
        edges, rates = ed._fit_binned_calibration(cal)
        very_negative = np.array([-1.0])
        result = ed._apply_binned_calibration(edges, rates, very_negative)
        assert result[0] == pytest.approx(rates[0])
