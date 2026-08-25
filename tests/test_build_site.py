"""scripts/build_site.py's scoreboard aggregation: n_abstained must count every abstention,
not just the ones that happen to have matured yet."""

from __future__ import annotations

from scripts import build_site


def _row(model="llm", point=None, lo=None, hi=None, level=None, actual=None) -> dict:
    scored = point is not None and lo is not None and actual is not None
    return {
        "as_of": "2026-08-25", "target_date": "2026-09-22", "symbol": "AAPL", "model": model,
        "point": point, "lo": lo, "hi": hi, "level": level,
        "actual": actual,
        "covered": (lo <= actual <= hi) if scored else None,
        "abs_error": abs(actual - point) if scored else None,
        "created_utc": "x",
        "mode": "live",
    }


class TestModelSummaryAbstentions:
    def test_open_unscored_abstentions_are_counted(self):
        """A record that has abstained (lo=None) but whose target hasn't matured yet
        (actual=None) must still show up in n_abstained -- the bug this test guards against
        made it invisible until a month later when the row finally matured."""
        records = [
            _row(point=None, lo=None, hi=None, actual=None),  # open abstention
            _row(point=100.0, lo=90.0, hi=110.0, level=0.8, actual=None),  # open, not abstained
        ]
        summary = build_site._model_summary(records)
        assert summary["n_abstained"] == 1
        assert summary["n_scored"] == 0
        assert summary["n"] == 2

    def test_matured_abstentions_are_still_counted(self):
        records = [
            _row(point=None, lo=None, hi=None, actual=150.0),  # matured abstention
        ]
        summary = build_site._model_summary(records)
        assert summary["n_abstained"] == 1
        assert summary["n_scored"] == 0

    def test_mixed_open_and_matured_abstentions_both_count(self):
        records = [
            _row(point=None, lo=None, hi=None, actual=None),  # open abstention
            _row(point=None, lo=None, hi=None, actual=150.0),  # matured abstention
            _row(point=100.0, lo=90.0, hi=110.0, level=0.8, actual=105.0),  # scored, not abstained
        ]
        summary = build_site._model_summary(records)
        assert summary["n_abstained"] == 2
        assert summary["n_scored"] == 1
        assert summary["n"] == 3

    def test_no_abstentions_gives_zero(self):
        records = [
            _row(point=100.0, lo=90.0, hi=110.0, level=0.8, actual=105.0),
        ]
        summary = build_site._model_summary(records)
        assert summary["n_abstained"] == 0
