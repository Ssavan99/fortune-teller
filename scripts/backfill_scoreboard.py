"""Historical backfill of the live scoreboard: replay src.rolling.run_cycle every month from
2022-01 to the latest as-of whose target has already matured, and score each cycle immediately
since — being history — the outcome already exists.

Run: ``python -m scripts.backfill_scoreboard`` (takes roughly 45 minutes on CPU; ~40 as-of
dates x 15 tickers x 2 models, one Bi-LSTM trained per as-of).

Every row this script writes is stamped ``mode="backtest"``. It goes through the identical
:func:`~src.rolling.run_cycle` code path the live monthly run (Phase 5) uses — that is what
makes the backtest/live comparison in the scoreboard meaningful rather than an apples-to-oranges
artifact of two different pipelines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src import rolling
from src.data import prices

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "scoreboard_backtest.json"

HORIZON = 21
SEED = 20260822
START_MONTH = "2022-01"


def monthly_as_of_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    """First trading day of each month from :data:`START_MONTH` through the latest month
    whose target date (``HORIZON`` sessions later) has already occurred in this snapshot —
    i.e. every as-of that can be both predicted AND scored right now.
    """
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    last_available = all_dates.max()

    months = pd.period_range(START_MONTH, last_available.to_period("M"), freq="M")
    as_of_dates: list[pd.Timestamp] = []
    for period in months:
        month_start = period.start_time
        candidates = all_dates[all_dates >= month_start]
        if candidates.empty:
            continue
        as_of = candidates.min()

        target = rolling.target_date_for(df, as_of, HORIZON)
        if target > last_available:
            continue  # the outcome hasn't happened yet in this snapshot
        as_of_dates.append(as_of)

    return as_of_dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="quantile",
        choices=rolling.INTERVAL_METHODS,
        help="interval-construction method to pass through to rolling.run_cycle",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="output path (default: results/scoreboard_backtest.json, the production file)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N as-of dates — for a cheap smoke test before the full run",
    )
    args = parser.parse_args()

    df = prices.load()
    as_of_dates = monthly_as_of_dates(df)
    if not as_of_dates:
        raise RuntimeError("no as-of dates are both predictable and scorable in this snapshot")
    if args.limit is not None:
        as_of_dates = as_of_dates[: args.limit]

    print(
        f"backfilling {len(as_of_dates)} as-of dates: "
        f"{as_of_dates[0].date()} -> {as_of_dates[-1].date()}  (method={args.method})"
    )

    all_records: list[dict] = []
    started = time.time()
    for i, as_of in enumerate(as_of_dates, start=1):
        cycle_start = time.time()
        records = rolling.run_cycle(df, as_of, horizon=HORIZON, seed=SEED, method=args.method)
        scored = [rolling.score_record(r, df) for r in records]
        rolling.stamp_mode(scored, "backtest")
        all_records.extend(scored)

        n_unscored = sum(1 for r in scored if r["actual"] is None)
        elapsed = time.time() - cycle_start
        total_elapsed = time.time() - started
        print(
            f"[{i}/{len(as_of_dates)}] as_of={as_of.date()}  {len(scored)} rows  "
            f"{n_unscored} unscored  ({elapsed:.0f}s, {total_elapsed / 60:.1f}m total)"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({len(all_records)} rows)")


if __name__ == "__main__":
    main()
