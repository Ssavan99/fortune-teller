"""Metric correctness, checked against hand-computed values rather than against itself."""

from __future__ import annotations

import numpy as np
import pytest

from src import metrics


class TestPointErrors:
    def test_rmse_on_a_known_case(self):
        # errors 1, -1, 2, -2 -> mean square 2.5 -> sqrt 1.5811...
        assert metrics.rmse([10, 10, 10, 10], [9, 11, 8, 12]) == pytest.approx(np.sqrt(2.5))

    def test_mae_on_a_known_case(self):
        assert metrics.mae([10, 10, 10, 10], [9, 11, 8, 12]) == pytest.approx(1.5)

    def test_mape_on_a_known_case(self):
        # |1|/100 and |3|/100 -> 1% and 3% -> 2%
        assert metrics.mape([100, 100], [99, 103]) == pytest.approx(2.0)

    def test_perfect_prediction_scores_zero(self):
        y = [1.0, 2.0, 3.0]
        assert metrics.rmse(y, y) == 0.0
        assert metrics.mae(y, y) == 0.0

    def test_rmse_penalises_large_errors_more_than_mae(self):
        y_true = [0, 0, 0, 0]
        concentrated = [0, 0, 0, 4]
        spread = [1, 1, 1, 1]
        assert metrics.mae(y_true, concentrated) == metrics.mae(y_true, spread)
        assert metrics.rmse(y_true, concentrated) > metrics.rmse(y_true, spread)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.rmse([1, 2, 3], [1, 2])

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="no observations"):
            metrics.rmse([], [])

    def test_mape_rejects_zero_prices(self):
        with pytest.raises(ValueError, match="undefined"):
            metrics.mape([0.0, 1.0], [1.0, 1.0])


class TestDirectionalAccuracy:
    def test_all_directions_correct(self):
        last = [100, 100, 100]
        true = [101, 99, 102]
        pred = [105, 95, 101]  # up, down, up — all match
        assert metrics.directional_accuracy(true, pred, last) == 1.0

    def test_all_directions_wrong(self):
        last = [100, 100]
        true = [101, 99]
        pred = [95, 105]
        assert metrics.directional_accuracy(true, pred, last) == 0.0

    def test_half_correct(self):
        last = [100, 100]
        true = [101, 99]
        pred = [105, 105]
        assert metrics.directional_accuracy(true, pred, last) == pytest.approx(0.5)

    def test_persistence_has_undefined_direction(self):
        """A forecast that never predicts a move has made no directional call at all."""
        last = [100, 100, 100]
        true = [101, 99, 102]
        pred = last  # persistence
        assert metrics.directional_accuracy(true, pred, last) is None

    def test_flat_predictions_are_excluded_not_counted_wrong(self):
        last = [100, 100]
        true = [101, 99]
        pred = [105, 100]  # one call (correct), one non-call
        assert metrics.directional_accuracy(true, pred, last) == 1.0


class TestSkillScore:
    def test_zero_when_model_matches_baseline(self):
        assert metrics.skill_score(5.0, 5.0) == 0.0

    def test_positive_when_model_is_better(self):
        assert metrics.skill_score(4.0, 5.0) == pytest.approx(0.2)

    def test_negative_when_model_is_worse(self):
        assert metrics.skill_score(6.0, 5.0) == pytest.approx(-0.2)

    def test_perfect_model_scores_one(self):
        assert metrics.skill_score(0.0, 5.0) == 1.0

    def test_rejects_nonpositive_baseline(self):
        with pytest.raises(ValueError, match="must be positive"):
            metrics.skill_score(1.0, 0.0)


class TestScoreSummary:
    def test_bundles_the_individual_metrics(self):
        true = [100.0, 101.0]
        pred = [99.0, 103.0]
        last = [100.0, 100.0]
        s = metrics.score(true, pred, last)
        assert s.rmse == pytest.approx(metrics.rmse(true, pred))
        assert s.mae == pytest.approx(metrics.mae(true, pred))
        assert s.mape == pytest.approx(metrics.mape(true, pred))
        assert s.n == 2

    def test_direction_is_none_without_an_anchor(self):
        assert metrics.score([1.0, 2.0], [1.0, 2.0]).directional_accuracy is None
