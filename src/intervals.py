"""Prediction intervals from empirical residual quantiles.

No distributional assumption, no test data touched: the quantiles come from each symbol's
*validation* residuals only, expressed as a fraction of the previous close so a $200 stock and
a $20 stock with the same percentage error get proportionally sized intervals rather than
identical dollar ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A prediction band for a batch of points, all at the same nominal coverage level."""

    lo: np.ndarray
    hi: np.ndarray
    level: float  # e.g. 0.80


def residual_quantiles(
    y_true, y_pred, prev_close, symbols, level: float = 0.80
) -> dict[str, tuple[float, float]]:
    """Per-symbol relative residual quantiles, computed on whatever partition is passed in.

    Callers must pass the **validation** partition only — this function does not know which
    partition it was given, so that guarantee lives in the caller.

    ``r = (y_true - y_pred) / prev_close``. Returns ``{symbol: (q_lo, q_hi)}`` at
    ``(1 - level) / 2`` and ``1 - (1 - level) / 2``, plus a pooled entry under the key
    ``"__pooled__"`` computed across every symbol's residuals together, for
    :func:`apply` to fall back on when a symbol is missing. The ``level`` these quantiles were
    computed at is stamped onto the dict under ``"__level__"`` so :func:`apply` can catch a
    caller applying them at a different, mismatched level.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    symbols = np.asarray(symbols).ravel()

    alpha = 1.0 - level
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0

    residuals = (y_true - y_pred) / prev_close

    out: dict[str, tuple[float, float]] = {}
    for symbol in np.unique(symbols):
        mask = symbols == symbol
        r = residuals[mask]
        out[symbol] = (float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q)))

    out["__pooled__"] = (float(np.quantile(residuals, lo_q)), float(np.quantile(residuals, hi_q)))
    out["__level__"] = level
    return out


def apply(
    y_pred,
    prev_close,
    symbols,
    quantiles: dict[str, tuple[float, float]],
    level: float = 0.80,
) -> Interval:
    """Turn point forecasts into an interval using per-symbol (or pooled) residual quantiles.

    ``lo = y_pred + q_lo * prev_close`` ; ``hi = y_pred + q_hi * prev_close``.

    A symbol absent from ``quantiles`` falls back to the pooled quantile under
    ``"__pooled__"`` rather than raising or emitting a zero-width interval — this is the
    common case in a live run, where a symbol's own validation residuals may not exist yet.
    ``level`` is carried through onto the returned :class:`Interval` for downstream scoring, and
    must match the level :func:`residual_quantiles` computed ``quantiles`` at (checked via the
    ``"__level__"`` entry it stamps into the dict) — a mismatch would silently score an 80%
    interval as if it were a 95% one, corrupting :func:`~src.metrics.interval_score`.
    """
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    symbols = np.asarray(symbols).ravel()

    if "__pooled__" not in quantiles:
        raise ValueError("quantiles must include a '__pooled__' fallback entry")
    baked_level = quantiles.get("__level__")
    if baked_level is not None and baked_level != level:
        raise ValueError(
            f"quantiles were computed at level={baked_level}, but apply() was called with "
            f"level={level} — these must match"
        )

    q_lo = np.empty(len(y_pred), dtype=float)
    q_hi = np.empty(len(y_pred), dtype=float)
    for i, symbol in enumerate(symbols):
        lo, hi = quantiles.get(symbol, quantiles["__pooled__"])
        q_lo[i] = lo
        q_hi[i] = hi

    lo = y_pred + q_lo * prev_close
    hi = y_pred + q_hi * prev_close
    return Interval(lo=lo, hi=hi, level=level)
