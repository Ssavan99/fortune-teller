"""Daily sentiment, aligned onto the trading calendar.

The snapshot has **no empty cells**. Every gap is a missing *row*: 14 trading days inside the
coverage window have no sentiment at all, and 33 rows fall on days the market was shut. So the
missing-indicator cannot be derived cell by cell — the series has to be reindexed onto each
symbol's actual trading days first, and the indicator built from what that reindex leaves
empty.

The gap value is `0.0` purely as a placeholder that the network can multiply away; the
indicator column beside it is what carries "there was no reading here". Filling gaps with 0.0
*without* the indicator would mean "neutral news today", which is a claim the data does not
make.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "data" / "sentiment_daily.csv"

#: The price universe has 15 tickers; the sentiment source covered 14 of them.
UNCOVERED = ("DELL",)


def load_wide(path: Path = SNAPSHOT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"sentiment snapshot not found at {path}")
    return pd.read_csv(path, parse_dates=["date"])


def load_long(path: Path = SNAPSHOT) -> pd.DataFrame:
    """One row per (date, symbol) with a ``sentiment`` column."""
    wide = load_wide(path)
    long = wide.melt(id_vars="date", var_name="symbol", value_name="sentiment")
    return long.dropna(subset=["sentiment"]).sort_values(
        ["symbol", "date"], ignore_index=True
    )


def covered_symbols(path: Path = SNAPSHOT) -> set[str]:
    return {c for c in load_wide(path).columns if c != "date"}


def align_to_prices(price_df: pd.DataFrame, path: Path = SNAPSHOT) -> dict[str, np.ndarray]:
    """Build per-symbol ``(n_rows, 2)`` feature blocks aligned to ``price_df``'s row order.

    Column 0 is the sentiment score, column 1 is a missing-indicator that is 1.0 wherever
    that trading day had no reading. Symbols the source never covered get an all-missing
    block rather than being dropped, so both ablation arms score the same universe.
    """
    long = load_long(path)
    lookup = {sym: part.set_index("date")["sentiment"] for sym, part in long.groupby("symbol")}

    blocks: dict[str, np.ndarray] = {}
    for symbol, part in price_df.groupby("symbol"):
        dates = pd.to_datetime(part.sort_values("date")["date"])
        series = lookup.get(symbol)

        if series is None:
            values = pd.Series(np.nan, index=dates.to_numpy())
        else:
            values = series.reindex(dates.to_numpy())

        missing = values.isna().to_numpy(dtype=float)
        filled = values.fillna(0.0).to_numpy(dtype=float)
        blocks[symbol] = np.column_stack([filled, missing])

    return blocks


def coverage_report(price_df: pd.DataFrame, path: Path = SNAPSHOT) -> dict:
    """How much of the price calendar the sentiment snapshot actually covers."""
    blocks = align_to_prices(price_df, path)
    per_symbol = {}
    for symbol, block in blocks.items():
        n = len(block)
        missing = int(block[:, 1].sum())
        per_symbol[symbol] = {
            "rows": n,
            "missing": missing,
            "coverage": round(1.0 - missing / n, 4) if n else 0.0,
        }
    return per_symbol
