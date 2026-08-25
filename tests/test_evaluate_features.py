"""The feature-ablation harness's extra-feature builders: correct shape and alignment, since a
misaligned extra-feature column would silently feed the model garbage without ever raising."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import evaluate_features as ef


def synthetic_history(n: int = 60, symbols=("A", "B")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-04", periods=n)
    frames = []
    for symbol in symbols:
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": closes,
                    "high": closes * 1.01,
                    "low": closes * 0.99,
                    "close": closes,
                    "volume": np.full(n, 1_000_000.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestExtraFeatureBuilders:
    def test_technical_extra_has_one_row_per_symbol_row_and_ten_columns(self):
        history = synthetic_history()
        extra = ef._technical_extra(history)
        for symbol, part in history.groupby("symbol"):
            assert extra[symbol].shape == (len(part), 10)  # 5 features x (value, missing)

    def test_cross_sectional_extra_has_one_row_per_symbol_row_and_two_columns(self):
        history = synthetic_history()
        extra = ef._cross_sectional_extra(history)
        for symbol, part in history.groupby("symbol"):
            assert extra[symbol].shape == (len(part), 2)

    def test_calendar_extra_has_one_row_per_symbol_row_and_two_columns_no_missing(self):
        history = synthetic_history()
        extra = ef._calendar_extra(history)
        for symbol, part in history.groupby("symbol"):
            assert extra[symbol].shape == (len(part), 2)
            assert np.all(np.isfinite(extra[symbol]))  # calendar features are never missing

    def test_extra_features_are_row_aligned_with_the_symbols_own_date_order(self):
        """The first technical-feature row for a symbol must correspond to that symbol's
        earliest date, not an arbitrary row -- a silent misalignment here would feed each
        window the wrong day's indicators without any shape error to catch it."""
        history = synthetic_history()
        extra = ef._technical_extra(history)
        part_a = history[history["symbol"] == "A"].sort_values("date").reset_index(drop=True)
        # hl_range (index 8 in the 10-wide value/missing-interleaved block: 5 features x 2)
        # should exactly match a fresh direct computation for symbol A, row by row.
        from src import features

        direct = features.high_low_range(part_a["high"], part_a["low"], part_a["close"])
        hl_col_index = ef.features.TECHNICAL_COLUMNS.index("hl_range") * 2
        assert np.allclose(extra["A"][:, hl_col_index], direct.to_numpy())
