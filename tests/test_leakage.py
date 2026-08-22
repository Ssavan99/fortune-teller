"""The guard tests.

Every other result in this repository is worthless if these fail. They are written to fail
loudly if someone reintroduces the pre-split-scaler pattern, so each one is checked against a
deliberately broken pipeline as well as the correct one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import preprocessing
from src.data import splits
from src.preprocessing import AlreadyFittedError, NotFittedError, TrainOnlyScaler


@pytest.fixture
def series_with_a_spike_in_the_test_period() -> pd.DataFrame:
    """A price series that is flat in training and spikes only after the test cutoff.

    Any scaler that has seen the test period will have a maximum of 1000; a scaler fitted on
    training alone will have a maximum of 100. The two are impossible to confuse.
    """
    dates = pd.bdate_range("2021-01-01", "2026-01-01")
    close = np.full(len(dates), 100.0)
    close[dates > splits.VAL_END] = 1000.0
    return pd.DataFrame({"date": dates, "symbol": "TEST", "close": close})


class TestTrainOnlyScalerBehaviour:
    def test_scales_to_unit_range_on_the_fitted_data(self):
        x = np.array([[0.0], [5.0], [10.0]])
        s = TrainOnlyScaler().fit(x)
        assert s.transform(x).min() == pytest.approx(0.0)
        assert s.transform(x).max() == pytest.approx(1.0)

    def test_roundtrips_through_inverse_transform(self):
        x = np.array([[1.0, 50.0], [2.0, 70.0], [3.0, 90.0]])
        s = TrainOnlyScaler().fit(x)
        assert np.allclose(s.inverse_transform(s.transform(x)), x)

    def test_unseen_high_values_exceed_one_rather_than_clipping(self):
        """Clipping would hide the distribution shift this project is trying to measure."""
        s = TrainOnlyScaler().fit(np.array([[0.0], [100.0]]))
        assert s.transform(np.array([[1000.0]]))[0, 0] == pytest.approx(10.0)

    def test_constant_column_does_not_divide_by_zero(self):
        s = TrainOnlyScaler().fit(np.array([[7.0], [7.0], [7.0]]))
        assert np.all(np.isfinite(s.transform(np.array([[7.0], [8.0]]))))

    def test_transform_before_fit_is_refused(self):
        with pytest.raises(NotFittedError):
            TrainOnlyScaler().transform(np.array([[1.0]]))

    def test_inverse_transform_before_fit_is_refused(self):
        with pytest.raises(NotFittedError):
            TrainOnlyScaler().inverse_transform(np.array([[1.0]]))

    def test_refitting_is_refused(self):
        """The second fit is the leak. Make it an error, not a silent overwrite."""
        s = TrainOnlyScaler().fit(np.array([[1.0], [2.0]]))
        with pytest.raises(AlreadyFittedError):
            s.fit(np.array([[1.0], [1000.0]]))

    def test_empty_fit_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            TrainOnlyScaler().fit(np.empty((0, 3)))

    def test_one_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="2-D"):
            TrainOnlyScaler().fit(np.array([1.0, 2.0, 3.0]))


class TestScalerNeverSeesTheHeldOutPeriod:
    def test_correct_pipeline_ignores_the_test_period_spike(
        self, series_with_a_spike_in_the_test_period
    ):
        df = series_with_a_spike_in_the_test_period
        split = splits.chronological_split(df)

        scaler = preprocessing.fit_on_train(split.train[["close"]].to_numpy())

        assert scaler.max_[0] == pytest.approx(100.0), (
            "scaler maximum is the test-period spike — the scaler saw held-out data"
        )
        assert scaler.n_fit_rows == len(split.train)

    def test_the_broken_pipeline_would_be_caught(self, series_with_a_spike_in_the_test_period):
        """Mutation check: fitting on the full frame is what we are defending against.

        If this assertion ever stops holding, the test above has stopped being meaningful.
        """
        df = series_with_a_spike_in_the_test_period
        leaky = TrainOnlyScaler().fit(df[["close"]].to_numpy())
        assert leaky.max_[0] == pytest.approx(1000.0)

    def test_scaler_row_count_equals_the_training_partition(self):
        from src.data import prices

        df = prices.load()
        split = splits.chronological_split(df)
        scaler = preprocessing.fit_on_train(split.train[["close"]].to_numpy())
        assert scaler.n_fit_rows == len(split.train)
        assert scaler.n_fit_rows < len(df)

    def test_training_maximum_is_below_the_full_series_maximum_on_real_data(self):
        """On this universe the market rose, so the two differ. If they ever coincided the
        guard above would pass vacuously — assert they do not."""
        from src.data import prices

        df = prices.load()
        split = splits.chronological_split(df)
        assert split.train["close"].max() < df["close"].max()
