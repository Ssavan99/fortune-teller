"""The scoreboard's honesty guarantees, checked against the actual committed files.

Everything else in this repo can be wrong and only the study suffers. These three are
different: a failure here means the repo's central claim — that live predictions are honest,
in-advance evidence, kept strictly apart from replayed history — is not actually true of what
is on disk right now.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_FILE = REPO_ROOT / "results" / "scoreboard_backtest.json"
LIVE_FILE = REPO_ROOT / "results" / "scoreboard_live.json"
SCOREBOARD_JSON = REPO_ROOT / "docs" / "data" / "scoreboard.json"

LEDGER_FIELDS = {
    "as_of", "target_date", "symbol", "model", "point", "lo", "hi",
    "level", "actual", "covered", "abs_error", "mode", "created_utc",
}


def _records(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path.name} does not exist yet")
    return json.loads(path.read_text(encoding="utf-8"))


class TestModeSeparation:
    """The rule the whole repo exists to protect: backtest and live are never one number."""

    def test_ledger_files_are_mode_pure(self):
        """Every row in the backtest ledger is 'backtest'; every row in the live ledger is
        'live'. A row of the wrong mode in either file is exactly the failure this test
        exists to catch — it would silently let the two kinds of evidence blend."""
        backtest = _records(BACKTEST_FILE)
        live = _records(LIVE_FILE)
        assert backtest, "backtest ledger is empty"
        assert live, "live ledger is empty"
        assert all(r["mode"] == "backtest" for r in backtest)
        assert all(r["mode"] == "live" for r in live)

    def test_backtest_and_live_are_never_aggregated_together(self):
        """Scan docs/data/scoreboard.json: live_summary and backtest_summary must be separate
        objects, and every count inside each must trace back to records of that mode alone —
        never a blended total."""
        backtest = _records(BACKTEST_FILE)
        live = _records(LIVE_FILE)
        if not SCOREBOARD_JSON.exists():
            pytest.skip("scoreboard.json has not been built yet")
        scoreboard = json.loads(SCOREBOARD_JSON.read_text(encoding="utf-8"))

        assert "live_summary" in scoreboard
        assert "backtest_summary" in scoreboard
        assert scoreboard["live_summary"].keys() or scoreboard["backtest_summary"].keys()

        for model, summary in scoreboard["live_summary"].items():
            expected_n = sum(1 for r in live if r["model"] == model)
            assert summary["n"] == expected_n, f"live/{model} count includes non-live rows"

        for model, summary in scoreboard["backtest_summary"].items():
            expected_n = sum(1 for r in backtest if r["model"] == model)
            assert summary["n"] == expected_n, f"backtest/{model} count includes non-backtest rows"

    def test_open_predictions_are_live_only(self):
        """The 'open predictions' section is what makes the page feel alive — it must never
        surface a backtest row as if it were an open live forecast."""
        if not SCOREBOARD_JSON.exists():
            pytest.skip("scoreboard.json has not been built yet")
        scoreboard = json.loads(SCOREBOARD_JSON.read_text(encoding="utf-8"))
        assert all(r["mode"] == "live" for r in scoreboard["open_predictions"])


class TestLedgerSchema:
    def test_ledger_schema_is_stable(self):
        """Exact field set, on every row of both ledgers. A schema drift here (a renamed or
        dropped field) would silently break scoring or the page without raising anywhere else."""
        for path in (BACKTEST_FILE, LIVE_FILE):
            for record in _records(path):
                assert set(record.keys()) == LEDGER_FIELDS, path.name

    def test_mode_is_always_one_of_the_two_valid_values(self):
        for path in (BACKTEST_FILE, LIVE_FILE):
            for record in _records(path):
                assert record["mode"] in ("backtest", "live")

    def test_null_fields_are_null_not_missing_or_stringified(self):
        """actual/covered/abs_error must be exactly None when unscored — never "null" the
        string, never 0, never absent."""
        for path in (BACKTEST_FILE, LIVE_FILE):
            for record in _records(path):
                if record["actual"] is None:
                    assert record["covered"] is None
                    assert record["abs_error"] is None
                else:
                    assert isinstance(record["covered"], bool)
                    assert isinstance(record["abs_error"], (int, float))


class TestNoSecretsInRepo:
    # Gemini API keys observed in this project start "AQ."; Google API keys generally start
    # "AIza". Both patterns are checked; either one appearing in tracked source is a real leak.
    KEY_PATTERNS = (
        re.compile(r"AQ\.[A-Za-z0-9_-]{20,}"),
        re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    )
    # File types where a match is expected to be real content, not noise (binary/lockfiles
    # aren't scanned since git already stores them verbatim and grep on them is meaningless).
    TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".js", ".html", ".css", ".txt", ""}

    def test_no_api_keys_in_repo(self):
        result = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        tracked = [f for f in result.stdout.splitlines() if f]
        assert tracked, "git ls-files returned nothing — not run inside a git checkout?"

        offenders = []
        for rel_path in tracked:
            path = REPO_ROOT / rel_path
            if not path.is_file():
                continue
            if path.suffix not in self.TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern in self.KEY_PATTERNS:
                if pattern.search(text):
                    offenders.append(rel_path)
                    break

        assert not offenders, f"key-shaped strings found in: {offenders}"
