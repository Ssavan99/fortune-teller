"""scripts/build_site.py's scoreboard aggregation: n_abstained must count every abstention,
not just the ones that happen to have matured yet; and _findings() must re-shape the
already-computed improvement/volatility results without dropping or renaming anything the
site depends on."""

from __future__ import annotations

import json

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


class TestFindings:
    """_findings() is pure re-shaping of results/improvement.json and
    results/volatility_evaluation.json for the "what's actually predictable" section — it must
    not silently drop a key the page reads, and must not compute anything of its own."""

    def _write_fixtures(self, tmp_path, monkeypatch):
        improvement = {
            "calibration_before_after": {
                "lstm": {"before": {"coverage": 0.674}, "after": {"coverage": 0.777}},
                "persistence": {"before": {"coverage": 0.678}, "after": {"coverage": 0.798}},
                "nominal_level": 0.8,
                "pre_registered_success_band": [0.76, 0.84],
            },
            "direction_arm": {
                "directional_accuracy": 0.5095,
                "brier_score": {"model": 0.2545, "base_rate_baseline": 0.2501},
                "log_loss": {"model": 0.7023, "base_rate_baseline": 0.6934},
                "base_rate_up": 0.5284,
            },
            "n_variants_tried": 9,
            "what_didnt_work": [{"variant": "x", "phase": "A", "outcome": "y"}],
        }
        volatility = {
            "skill_vs_persistence": 0.0841,
            "tickers_beating_persistence": 15,
            "n_tickers": 15,
            "units": "annualized daily-return volatility (std * sqrt(252))",
            "window_days": 21,
            "per_ticker": {"AAPL": {"skill_vs_persistence": 0.06}},
            "series_stride_days": 5,
            "series": {"AAPL": [{"date": "2021-01-13", "actual": 0.31}]},
        }
        improvement_path = tmp_path / "improvement.json"
        volatility_path = tmp_path / "volatility_evaluation.json"
        improvement_path.write_text(json.dumps(improvement), encoding="utf-8")
        volatility_path.write_text(json.dumps(volatility), encoding="utf-8")
        monkeypatch.setattr(build_site, "IMPROVEMENT_FILE", improvement_path)
        monkeypatch.setattr(build_site, "VOLATILITY_FILE", volatility_path)
        return improvement, volatility

    def test_findings_carries_every_key_the_page_reads(self, tmp_path, monkeypatch):
        improvement, volatility = self._write_fixtures(tmp_path, monkeypatch)
        findings = build_site._findings()

        assert findings["calibration"] == improvement["calibration_before_after"]
        assert findings["direction"] == improvement["direction_arm"]
        assert findings["n_variants_tried"] == improvement["n_variants_tried"]
        assert findings["what_didnt_work"] == improvement["what_didnt_work"]

        vol = findings["volatility"]
        assert vol["skill_vs_persistence"] == volatility["skill_vs_persistence"]
        assert vol["tickers_beating_persistence"] == volatility["tickers_beating_persistence"]
        assert vol["per_ticker"] == volatility["per_ticker"]
        assert vol["series"] == volatility["series"]

    def test_missing_improvement_file_fails_loudly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_site, "IMPROVEMENT_FILE", tmp_path / "missing.json")
        monkeypatch.setattr(build_site, "VOLATILITY_FILE", tmp_path / "also_missing.json")
        try:
            build_site._findings()
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
