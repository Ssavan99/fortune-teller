"""Turn long-format bars into supervised sequences without letting the future in.

Two rules make this safe, and both are asserted in the tests:

1. **A sequence's inputs end strictly before its target date.** For a target at day *t*, the
   window covers days *t−lookback* through *t−1*. Nothing in the window is contemporaneous
   with, or later than, the thing being predicted.
2. **A sequence belongs to the partition of its target date.** Windows are allowed to reach
   back across a partition boundary — at prediction time you genuinely do know the preceding
   fortnight — but a training target can never draw on a later partition, because its window
   lies entirely in its own past.

Scaling statistics come from each symbol's training rows only, via
:class:`~src.preprocessing.TrainOnlyScaler`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing import TrainOnlyScaler, fit_on_train

FEATURES = ("open", "high", "low", "close", "volume")
CLOSE_INDEX = FEATURES.index("close")

Target = str  # "level" or "return"


@dataclass
class SequenceSet:
    """Sequences for one partition, with everything needed to score them in dollars."""

    x: np.ndarray  # (n, lookback, n_features), scaled
    y: np.ndarray  # (n,) scaled level, or raw return
    y_close: np.ndarray  # (n,) true next close, in dollars
    prev_close: np.ndarray  # (n,) close on the day before the target
    symbols: np.ndarray  # (n,)
    dates: np.ndarray  # (n,)

    def __len__(self) -> int:
        return len(self.y)


def _build_for_symbol(
    part: pd.DataFrame,
    scaler: TrainOnlyScaler,
    lookback: int,
    target: Target,
    extra: np.ndarray | None,
):
    values = part[list(FEATURES)].to_numpy(dtype=float)
    scaled = scaler.transform(values)
    if extra is not None:
        scaled = np.hstack([scaled, extra])

    closes = part["close"].to_numpy(dtype=float)
    dates = part["date"].to_numpy()

    n = len(part)
    idx = np.arange(lookback, n)
    if idx.size == 0:
        return None

    # Window [i - lookback, i - 1] predicts the close at i.
    x = np.stack([scaled[i - lookback : i] for i in idx])
    y_close = closes[idx]
    prev_close = closes[idx - 1]

    if target == "level":
        lo = scaler.min_[CLOSE_INDEX]
        hi = scaler.max_[CLOSE_INDEX]
        span = hi - lo if hi != lo else 1.0
        y = (y_close - lo) / span
    elif target == "return":
        y = y_close / prev_close - 1.0
    else:
        raise ValueError(f"unknown target {target!r}; expected 'level' or 'return'")

    return x, y, y_close, prev_close, dates[idx]


def build(
    df: pd.DataFrame,
    split,
    lookback: int = 20,
    target: Target = "level",
    extra_features: dict[str, np.ndarray] | None = None,
) -> dict[str, SequenceSet]:
    """Build train/val/test sequence sets from a long-format frame.

    ``extra_features`` optionally supplies per-symbol columns already aligned to ``df``'s rows
    for that symbol — used by the sentiment ablation. They are passed through unscaled, since
    sentiment is already bounded and its missing-indicator is binary.
    """
    train_dates = set(split.train["date"])
    val_dates = set(split.val["date"])

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}

    for symbol in sorted(df["symbol"].unique()):
        part = df[df["symbol"] == symbol].sort_values("date", ignore_index=True)

        train_rows = part[part["date"].isin(train_dates)]
        if train_rows.empty:
            continue
        scaler = fit_on_train(train_rows[list(FEATURES)].to_numpy(dtype=float))

        extra = extra_features.get(symbol) if extra_features else None
        built = _build_for_symbol(part, scaler, lookback, target, extra)
        if built is None:
            continue
        x, y, y_close, prev_close, dates = built

        for name, mask in (
            ("train", np.isin(dates, list(train_dates))),
            ("val", np.isin(dates, list(val_dates))),
            ("test", ~np.isin(dates, list(train_dates | val_dates))),
        ):
            if not mask.any():
                continue
            buckets[name].append(
                SequenceSet(
                    x=x[mask],
                    y=y[mask],
                    y_close=y_close[mask],
                    prev_close=prev_close[mask],
                    symbols=np.full(int(mask.sum()), symbol),
                    dates=dates[mask],
                )
            )

    return {name: _concat(parts) for name, parts in buckets.items()}


def _concat(parts: list[SequenceSet]) -> SequenceSet:
    if not parts:
        raise ValueError("no sequences were built for a partition")
    return SequenceSet(
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        y_close=np.concatenate([p.y_close for p in parts]),
        prev_close=np.concatenate([p.prev_close for p in parts]),
        symbols=np.concatenate([p.symbols for p in parts]),
        dates=np.concatenate([p.dates for p in parts]),
    )


class ScaledInverter:
    """Holds each symbol's close-scaling constants so level predictions can be un-scaled.

    Level predictions live in each symbol's own training-derived scale, so inverting them
    needs the same per-symbol constants the sequences were built with. Return predictions
    need no inverter — see :func:`to_dollars`.
    """

    def __init__(self, df: pd.DataFrame, split) -> None:
        train_dates = set(split.train["date"])
        self.bounds: dict[str, tuple[float, float]] = {}
        for symbol in sorted(df["symbol"].unique()):
            part = df[df["symbol"] == symbol]
            train_rows = part[part["date"].isin(train_dates)]
            if train_rows.empty:
                continue
            lo = float(train_rows["close"].min())
            hi = float(train_rows["close"].max())
            self.bounds[symbol] = (lo, hi if hi != lo else lo + 1.0)

    def __call__(self, pred: np.ndarray, seqs: SequenceSet) -> np.ndarray:
        out = np.empty(len(pred), dtype=float)
        for symbol in np.unique(seqs.symbols):
            mask = seqs.symbols == symbol
            lo, hi = self.bounds[symbol]
            out[mask] = pred[mask] * (hi - lo) + lo
        return out


def to_dollars(
    pred: np.ndarray,
    seqs: SequenceSet,
    target: Target,
    inverter: ScaledInverter | None = None,
) -> np.ndarray:
    """Convert model output to a dollar price prediction.

    ``return`` targets need only the anchor price. ``level`` targets need the per-symbol
    scaling constants, supplied by a :class:`ScaledInverter`.
    """
    if target == "return":
        return seqs.prev_close * (1.0 + np.asarray(pred, dtype=float))
    if target == "level":
        if inverter is None:
            raise ValueError("level predictions require a ScaledInverter")
        return inverter(np.asarray(pred, dtype=float), seqs)
    raise ValueError(f"unknown target {target!r}; expected 'level' or 'return'")
