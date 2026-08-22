"""Model and training-loop contracts, on small synthetic problems so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src import sequences
from src.data import splits
from src.models.lstm import BiLSTMRegressor, count_parameters
from src.train import TargetNormalizer, TrainConfig, predict, set_seed, train


def tiny_frame(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1000.0),
        }
    )


@pytest.fixture(scope="module")
def tiny():
    df = tiny_frame()
    sp = splits.Split(train=df.iloc[:160], val=df.iloc[160:210], test=df.iloc[210:])
    built = sequences.build(df, sp, lookback=10, target="return")
    return df, sp, built


def fast_config(**kw) -> TrainConfig:
    base = dict(target="return", lookback=10, hidden=8, layers=1, max_epochs=2, patience=2)
    base.update(kw)
    return TrainConfig(**base)


class TestModel:
    def test_output_is_one_scalar_per_sequence(self):
        model = BiLSTMRegressor(n_features=5, hidden=8, layers=1)
        out = model(torch.randn(7, 10, 5))
        assert out.shape == (7,)

    def test_parameter_count_grows_with_width(self):
        small = count_parameters(BiLSTMRegressor(5, hidden=8, layers=1))
        large = count_parameters(BiLSTMRegressor(5, hidden=64, layers=1))
        assert large > small

    def test_accepts_the_configured_feature_count(self):
        model = BiLSTMRegressor(n_features=7, hidden=8, layers=1)
        assert model(torch.randn(3, 10, 7)).shape == (3,)

    def test_rejects_the_wrong_feature_count(self):
        model = BiLSTMRegressor(n_features=5, hidden=8, layers=1)
        with pytest.raises(RuntimeError):
            model(torch.randn(3, 10, 9))


class TestTargetNormalizer:
    def test_standardises_to_zero_mean_unit_variance(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        n = TargetNormalizer(y)
        z = n.forward(y)
        assert np.mean(z) == pytest.approx(0.0, abs=1e-12)
        assert np.std(z) == pytest.approx(1.0)

    def test_roundtrips(self):
        y = np.array([0.01, -0.02, 0.005])
        n = TargetNormalizer(y)
        assert np.allclose(n.inverse(n.forward(y)), y)

    def test_constant_target_does_not_divide_by_zero(self):
        n = TargetNormalizer(np.full(5, 0.3))
        assert np.all(np.isfinite(n.forward(np.array([0.3, 0.4]))))

    def test_statistics_come_from_the_array_it_was_given(self):
        """Fitted on train, applied to val — val values must not change the constants."""
        n = TargetNormalizer(np.array([0.0, 1.0]))
        before = (n.mean, n.std)
        n.forward(np.array([100.0, 200.0]))
        assert (n.mean, n.std) == before


class TestTrainingLoop:
    def test_runs_and_returns_a_usable_model(self, tiny):
        _, _, built = tiny
        model, norm, config = train(built["train"], built["val"], fast_config(), verbose=False)
        preds = predict(model, built["test"], norm)
        assert preds.shape == (len(built["test"]),)
        assert np.all(np.isfinite(preds))

    def test_records_history_for_every_epoch(self, tiny):
        _, _, built = tiny
        _, _, config = train(built["train"], built["val"], fast_config(), verbose=False)
        assert len(config.history) == 2
        assert {"epoch", "train_loss", "val_loss"} <= set(config.history[0])

    def test_is_deterministic_under_a_fixed_seed(self, tiny):
        """Two runs of the same config must produce identical predictions."""
        _, _, built = tiny
        m1, n1, _ = train(built["train"], built["val"], fast_config(seed=7), verbose=False)
        p1 = predict(m1, built["test"], n1)
        m2, n2, _ = train(built["train"], built["val"], fast_config(seed=7), verbose=False)
        p2 = predict(m2, built["test"], n2)
        assert np.allclose(p1, p2)

    def test_different_seeds_give_different_models(self, tiny):
        _, _, built = tiny
        m1, n1, _ = train(built["train"], built["val"], fast_config(seed=1), verbose=False)
        m2, n2, _ = train(built["train"], built["val"], fast_config(seed=2), verbose=False)
        assert not np.allclose(
            predict(m1, built["test"], n1), predict(m2, built["test"], n2)
        )

    def test_early_stopping_can_stop_before_max_epochs(self, tiny):
        _, _, built = tiny
        config = fast_config(max_epochs=50, patience=1)
        _, _, config = train(built["train"], built["val"], config, verbose=False)
        assert len(config.history) < 50

    def test_set_seed_makes_torch_reproducible(self):
        set_seed(11)
        a = torch.randn(4)
        set_seed(11)
        assert torch.allclose(a, torch.randn(4))

    def test_config_serialises_without_the_history(self):
        d = fast_config().as_dict()
        assert "history" not in d
        assert d["target"] == "return"
