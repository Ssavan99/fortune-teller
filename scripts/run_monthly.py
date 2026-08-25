"""The live monthly forecast run: predict now, score whatever has matured, append-only.

Run: ``python -m scripts.run_monthly --score --predict`` (score before predict, so a maturing
prediction from a prior month is scored before this month's row is added — matches the order
the GitHub Actions workflow runs these in).

Every row this script writes is stamped ``mode="live"``. Predictions are committed to git
*before* the outcome exists — that is the entire evidentiary value of the live table on the
scoreboard page, and it only holds if this script never revises a prediction after the fact.
Two things enforce that:

* ``--predict`` is a no-op (prints "already predicted", exits 0) if a row for this ``as_of``
  already exists in the ledger — re-running it **sequentially** can never silently duplicate or
  replace a row. This guarantee is sequential-only: two *concurrent* invocations against the
  same ledger file (e.g. a manual local run overlapping a scheduled one) can both pass the
  "already predicted" check before either writes, and the later write wins outright rather than
  merging — the GitHub Actions workflow's ``concurrency:`` block prevents that for scheduled/
  dispatched runs, but nothing in this script itself defends against a local concurrent run.
* ``--score`` only ever fills ``actual``/``covered``/``abs_error`` on rows that don't have them
  yet (via :func:`src.rolling.score_record`, which is itself idempotent) — it never touches
  ``point``, ``lo``, ``hi``, ``as_of``, or ``target_date``.

The optional LLM arm (:mod:`src.models.llm_forecaster`) is folded in here, not in
:mod:`src.rolling`, because it is live-only — there is no honest way to backfill it (see that
module's docstring) — so it has no business living in the engine both backtest and live share.
Its rows use ``level: None`` rather than a numeric confidence level: unlike the LSTM and
persistence arms, whose ``lo``/``hi`` come from a residual-quantile procedure calibrated to an
explicit level, the LLM's range is just whatever it decided to output — reporting it as a
calibrated "80% interval" would overstate what it actually is.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src import rolling
from src.data import news, prices
from src.models import llm_forecaster

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "results" / "scoreboard_live.json"

HORIZON = 21
SEED = 20260822


def _load_ledger(path: Path) -> list[dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_ledger(path: Path, records: list[dict]) -> None:
    """Write via a temp file + atomic rename, never a direct truncating write.

    A direct ``path.write_text(...)`` truncates the file immediately, then writes — a crash
    (runner timeout, OOM kill, Ctrl+C) in between leaves a corrupted, unparseable ledger. The
    temp-file-then-``os.replace`` swap is atomic on both POSIX and Windows: the ledger is
    either the old complete file or the new complete file, never a partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _llm_records(
    df: pd.DataFrame, as_of: pd.Timestamp, target_date: pd.Timestamp, created_utc: str
) -> list[dict]:
    """One record per ticker from the LLM arm, or ``[]`` entirely if no free key is available.

    Reads only ``history`` (rows dated before ``as_of``) for the closes it feeds the prompt —
    the live run's ``as_of`` is "today", so this is naturally never a lookahead risk the way it
    would be in a backtest, which is exactly why this arm never runs in backtest at all.

    Every ticker is wrapped in its own ``try/except``: an unexpected failure in one ticker's
    news fetch or cache I/O (``llm_forecaster.forecast`` already catches transport/HTTP errors
    internally, but not e.g. a corrupted cache file) must become an abstention for that ticker
    alone, never an exception that propagates up through ``predict()`` and discards the whole
    month's already-computed LSTM/persistence rows, which cost real training time.
    """
    if not llm_forecaster.is_available():
        return []

    history = df[df["date"] < as_of]
    as_of_str = str(as_of.date())
    records = []
    for symbol in prices.TICKERS:
        part = history[history["symbol"] == symbol].sort_values("date")
        if part.empty:
            continue
        try:
            closes = part["close"].tolist()
            headlines = [h["title"] for h in news.fetch_headlines(symbol)]
            result = llm_forecaster.forecast(symbol, as_of_str, closes, headlines)
        except Exception as exc:  # noqa: BLE001 - isolate one ticker's failure from the rest
            result = {"point": None, "lo": None, "hi": None, "reason": f"error: {exc}"}
        records.append(
            {
                "as_of": as_of_str,
                "target_date": str(target_date.date()),
                "symbol": symbol,
                "model": "llm",
                "point": result["point"],
                "lo": result["lo"],
                "hi": result["hi"],
                "level": None,
                "actual": None,
                "covered": None,
                "abs_error": None,
                "created_utc": created_utc,
            }
        )
    return records


def predict(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    ledger_path: Path = LEDGER,
    train_config_overrides: dict | None = None,
) -> None:
    """Add live rows for ``as_of`` to the ledger, or do nothing if it's already there."""
    ledger = _load_ledger(ledger_path)
    as_of_str = str(pd.Timestamp(as_of).date())
    if any(r["as_of"] == as_of_str for r in ledger):
        print(f"already predicted for as_of={as_of_str}")
        return

    kwargs = {"train_config_overrides": train_config_overrides} if train_config_overrides else {}
    records = rolling.run_cycle(df, as_of, horizon=HORIZON, seed=SEED, **kwargs)
    rolling.stamp_mode(records, "live")

    target_date = rolling.target_date_for(df, as_of, HORIZON)
    created_utc = pd.Timestamp.now(tz="UTC").isoformat()
    llm_records = _llm_records(df, pd.Timestamp(as_of), target_date, created_utc)
    rolling.stamp_mode(llm_records, "live")
    records.extend(llm_records)

    ledger.extend(records)
    _save_ledger(ledger_path, ledger)
    print(
        f"predicted {len(records)} rows for as_of={as_of_str} "
        f"(target_date={target_date.date()}, llm={'on' if llm_records else 'off'})"
    )


def score(df: pd.DataFrame, ledger_path: Path = LEDGER) -> None:
    """Fill in outcomes for any ledger row whose target has matured since it was predicted."""
    ledger = _load_ledger(ledger_path)
    scored = [
        rolling.score_record(r, df) if r["actual"] is None else r for r in ledger
    ]
    newly_scored = sum(
        1
        for old, new in zip(ledger, scored, strict=True)
        if old["actual"] is None and new["actual"] is not None
    )
    _save_ledger(ledger_path, scored)
    print(f"scored {newly_scored} newly matured row(s) out of {len(ledger)} total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="add live rows for today")
    parser.add_argument("--score", action="store_true", help="fill outcomes for matured rows")
    args = parser.parse_args()
    if not args.predict and not args.score:
        parser.error("pass --predict and/or --score")

    df = prices.load()

    if args.score:
        score(df)
    if args.predict:
        as_of = pd.Timestamp(pd.Timestamp.now(tz="UTC").date())
        predict(df, as_of)


if __name__ == "__main__":
    main()
