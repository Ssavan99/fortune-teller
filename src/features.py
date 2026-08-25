"""Phase D candidate LSTM input features, evaluated one group at a time on validation only.

Every function here computes a value for row ``i`` from rows up to and including ``i`` alone —
never from a later row. That is not an extra precaution on top of ``sequences.py``'s own
leakage guard, it is a *necessary* one: a feature that peeked at a future row would let the
window "see" its own target through the back door, regardless of how carefully the window
boundary itself is enforced elsewhere.

Missing values (the first few rows of any rolling computation, or a symbol lacking a feature)
follow the same convention as the sentiment ablation in the original study: filled with 0.0
plus a separate missing-indicator column, never silently treated as a real zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index. Values in [0, 100]; NaN for the first `period` rows."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_histogram(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line minus its own signal line) — a single scalar per row rather
    than two correlated series, since the histogram is what's usually treated as the signal."""
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def realized_vol_21d(closes: pd.Series) -> pd.Series:
    return closes.pct_change().rolling(21).std()


def volume_zscore(volume: pd.Series, window: int = 21) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)


def high_low_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high - low) / close


TECHNICAL_COLUMNS = ("rsi14", "macd_hist", "vol21d", "volume_z", "hl_range")


def technical_features(part: pd.DataFrame) -> pd.DataFrame:
    """All five technical indicators for one symbol's price rows, in a fixed column order."""
    return pd.DataFrame(
        {
            "rsi14": rsi(part["close"]),
            "macd_hist": macd_histogram(part["close"]),
            "vol21d": realized_vol_21d(part["close"]),
            "volume_z": volume_zscore(part["volume"]),
            "hl_range": high_low_range(part["high"], part["low"], part["close"]),
        }
    )


def cross_sectional_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Equal-weight index return of every symbol, and each symbol's own return minus that —
    the "was this ticker relatively strong or weak today" signal, indexed by that symbol's own
    dates. Cross-sectional structure is more predictable than absolute price moves.

    The index return at date `d` uses only that date's own closes-vs-prior-day, across
    whichever symbols have both — no future date is involved, so this is exactly as leak-free
    as a single symbol's own return.
    """
    wide = df.pivot(index="date", columns="symbol", values="close").sort_index()
    daily_returns = wide.pct_change()
    index_return = daily_returns.mean(axis=1)  # equal-weight across whatever symbols exist

    out: dict[str, pd.Series] = {}
    for symbol in wide.columns:
        relative = daily_returns[symbol] - index_return
        out[symbol] = relative
    return out


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Day-of-week (0=Monday) and a month-end flag (last trading day of the calendar month
    present in `dates` — approximated as "the last row before the month number changes",
    which needs no forward-looking information since it only compares each date to the next
    one already in the series)."""
    dow = dates.dt.dayofweek.astype(float)
    month = dates.dt.month
    is_month_end = (month != month.shift(-1)).astype(float)
    is_month_end.iloc[-1] = 1.0  # the series' own last row is trivially a month boundary
    return pd.DataFrame({"day_of_week": dow, "month_end": is_month_end})


def with_missing_indicator(series: pd.Series) -> np.ndarray:
    """(n, 2) array: the series with NaNs filled to 0.0, plus a missing-indicator column —
    the same convention `src/data/sentiment.py` uses, so a filled 0.0 is never confused with a
    genuine zero reading."""
    missing = series.isna().to_numpy(dtype=float)
    filled = series.fillna(0.0).to_numpy(dtype=float)
    return np.column_stack([filled, missing])
