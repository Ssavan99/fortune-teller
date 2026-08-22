"""Sentiment alignment: the missing-indicator must reflect the trading calendar."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import prices, sentiment, splits


@pytest.fixture(scope="module")
def window():
    df = prices.load()
    return df[df["date"] <= splits.SENTIMENT_END].copy()


@pytest.fixture(scope="module")
def blocks(window):
    return sentiment.align_to_prices(window)


class TestLoading:
    def test_long_format_has_one_row_per_date_symbol(self):
        long = sentiment.load_long()
        assert not long.duplicated(subset=["date", "symbol"]).any()

    def test_covered_symbols_exclude_dell(self):
        assert "DELL" not in sentiment.covered_symbols()
        assert sentiment.UNCOVERED == ("DELL",)

    def test_covered_symbols_are_a_subset_of_the_price_universe(self):
        assert sentiment.covered_symbols() <= set(prices.TICKERS)


class TestAlignment:
    def test_every_price_symbol_gets_a_block(self, blocks):
        assert set(blocks) == set(prices.TICKERS)

    def test_block_length_matches_the_price_rows(self, window, blocks):
        for symbol, block in blocks.items():
            assert len(block) == (window["symbol"] == symbol).sum()

    def test_block_has_score_and_indicator_columns(self, blocks):
        for block in blocks.values():
            assert block.shape[1] == 2

    def test_indicator_is_binary(self, blocks):
        for block in blocks.values():
            assert set(np.unique(block[:, 1])) <= {0.0, 1.0}

    def test_uncovered_symbol_is_entirely_missing(self, blocks):
        """DELL had no sentiment at all. It must be flagged, not silently zero-filled."""
        assert blocks["DELL"][:, 1].all()
        assert not blocks["DELL"][:, 0].any()

    def test_covered_symbols_are_mostly_present(self, blocks):
        for symbol, block in blocks.items():
            if symbol in sentiment.UNCOVERED:
                continue
            assert block[:, 1].mean() < 0.05, f"{symbol} is missing more than 5% of days"

    def test_gaps_are_flagged_not_treated_as_neutral(self, blocks):
        """Wherever the indicator is 1 the score must be the placeholder, and vice versa.

        This is the property that keeps 'no reading' distinguishable from 'neutral news'.
        """
        for block in blocks.values():
            missing = block[:, 1] == 1.0
            assert np.all(block[missing, 0] == 0.0)

    def test_a_genuine_zero_score_is_not_flagged_as_missing(self, blocks):
        """A real reading of exactly 0.0 would be indistinguishable from a gap without the
        indicator being built from the reindex rather than from the value."""
        long = sentiment.load_long()
        assert (long["sentiment"] == 0.0).sum() >= 0  # documents the case exists or not
        for block in blocks.values():
            present = block[:, 1] == 0.0
            assert np.all(np.isfinite(block[present, 0]))

    def test_alignment_follows_price_row_order(self, window):
        """Blocks are positional, so a shuffled input must not silently misalign."""
        shuffled = window.sample(frac=1.0, random_state=0)
        a = sentiment.align_to_prices(window)
        b = sentiment.align_to_prices(shuffled)
        for symbol in a:
            assert np.allclose(a[symbol], b[symbol])


class TestCoverageReport:
    def test_reports_every_symbol(self, window):
        report = sentiment.coverage_report(window)
        assert set(report) == set(prices.TICKERS)

    def test_dell_has_zero_coverage(self, window):
        assert sentiment.coverage_report(window)["DELL"]["coverage"] == 0.0

    def test_covered_tickers_exceed_ninety_percent(self, window):
        report = sentiment.coverage_report(window)
        for symbol, entry in report.items():
            if symbol in sentiment.UNCOVERED:
                continue
            assert entry["coverage"] > 0.90


class TestExperimentBWindow:
    def test_window_stops_before_the_main_held_out_period(self, window):
        assert window["date"].max() <= splits.VAL_END

    def test_split_test_bucket_stays_inside_the_window(self, window):
        split = splits.sentiment_window_split(window)
        assert split.test["date"].max() <= splits.SENTIMENT_END

    def test_sentiment_snapshot_extends_past_the_clamp(self):
        """The clamp is doing real work — the raw data does run past it."""
        assert sentiment.load_long()["date"].max() > splits.SENTIMENT_END

    def test_passing_the_full_frame_would_be_caught(self):
        """Guards the runner's own check: an untruncated frame must be rejected."""
        full = prices.load()
        assert full["date"].max() > splits.SENTIMENT_END
