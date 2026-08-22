"""Daily OHLCV bars for the ticker universe.

The repository ships a committed snapshot at ``data/prices.csv`` so that every result is
reproducible offline and in CI without a network call. ``--refresh`` re-fetches from Yahoo
Finance and overwrites that snapshot.

Prices are **split- and dividend-adjusted** (``auto_adjust=True``). This is not cosmetic:
NVDA alone split 4:1 in July 2021 and 10:1 in June 2024, both inside the study window. On
unadjusted bars those events appear as an overnight ``close`` collapse of 75% and 90%, which
a next-day model would be scored against as if it were a real price move.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: Large-cap tech universe. Fixed for the whole study.
TICKERS: tuple[str, ...] = (
    "AAPL", "ADBE", "AMD", "AMZN", "BABA", "DELL", "GOOG", "IBM",
    "INTC", "META", "MSFT", "NVDA", "ORCL", "SAP", "TSLA",
)

START = "2020-12-10"

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "data" / "prices.csv"


def fetch(tickers: tuple[str, ...] = TICKERS, start: str = START, end: str | None = None):
    """Download adjusted daily bars and return them in long format.

    Returns one row per (date, symbol) with columns :data:`COLUMNS`, sorted by symbol then
    date. Rows where the close is missing — a ticker not yet listed, or a data gap — are
    dropped rather than forward-filled, so no fabricated bar can reach a model.
    """
    import yfinance as yf

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if not isinstance(raw.columns, pd.MultiIndex):
        # yfinance flattens the column index for a single ticker; restore the shape the rest
        # of this function expects rather than silently producing an empty frame.
        raw.columns = pd.MultiIndex.from_product([[tickers[0]], raw.columns])

    available = set(raw.columns.get_level_values(0))
    missing = [s for s in tickers if s not in available]
    if missing:
        raise ValueError(f"no data returned for: {', '.join(missing)}")

    frames = []
    for symbol in tickers:
        part = raw[symbol].reset_index()
        part.columns = [str(c).lower() for c in part.columns]
        part["symbol"] = symbol
        frames.append(part[["date", "symbol", "open", "high", "low", "close", "volume"]])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["close"])
    df = df.sort_values(["symbol", "date"], ignore_index=True)
    return df[COLUMNS]


def load(path: Path = SNAPSHOT):
    """Load the committed snapshot. Raises if it is missing — never silently refetches."""
    if not path.exists():
        raise FileNotFoundError(
            f"Price snapshot not found at {path}. Run: python -m src.data.prices --refresh"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["symbol", "date"], ignore_index=True)[COLUMNS]


def describe(df) -> str:
    lines = [
        f"rows      {len(df):,}",
        f"symbols   {df['symbol'].nunique()} ({', '.join(sorted(df['symbol'].unique()))})",
        f"dates     {df['date'].min().date()} -> {df['date'].max().date()}",
        "",
        "per symbol:",
    ]
    per = df.groupby("symbol")["date"].agg(["count", "min", "max"])
    for sym, row in per.iterrows():
        lines.append(f"  {sym:5s} {row['count']:5d}  {row['min'].date()} -> {row['max'].date()}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch and overwrite the snapshot"
    )
    parser.add_argument("--end", default=None, help="end date (exclusive), YYYY-MM-DD")
    args = parser.parse_args()

    if args.refresh:
        df = fetch(end=args.end)
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SNAPSHOT, index=False)
        print(f"wrote {SNAPSHOT}")
    else:
        df = load()

    print(describe(df))


if __name__ == "__main__":
    main()
