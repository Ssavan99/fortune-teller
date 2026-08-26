"""Split-conformal prediction intervals — a finite-sample coverage guarantee under
exchangeability, in place of the empirical per-symbol residual quantiles this replaces.

Why this exists: the shipped scoreboard's intervals undercover (67-68% realized vs 80%
nominal) even though their *width* is close to what an oracle calibrated on the same data
would need (see `scripts/diagnose_calibration.py` and the plan's §7 diagnosis). That points at
an *effective sample size* problem, not a width or centering problem: each cycle's validation
set is ~126 daily rows at horizon 21, so adjacent rows share ~95% of their input window and are
strongly autocorrelated — the effective sample size for estimating a percentile is closer to
~6 independent windows than 126. A noisy quantile estimate from ~6 effective points is exactly
what produces "roughly right average width, wrong precise coverage, no clean regime pattern,"
which is what the diagnosis found.

Two things fix that here:

1. **A symmetric nonconformity score** (``|y_true - y_pred| / prev_close``) instead of separate
   asymmetric lo/hi quantiles of the signed residual — one quantity to estimate instead of two,
   which halves the noise from small effective samples.
2. **Pooling calibration scores across all 15 tickers** by default. Relative (÷prev_close)
   residuals are already scale-free across tickers (established in Phase 1), so pooling is
   valid, and it multiplies the effective calibration sample size roughly 15x — which is what
   actually addresses the diagnosed mechanism.

The finite-sample correction (`ceil((n+1)(1-alpha))/n`, not the naive `(1-alpha)`-th quantile)
is what makes this an honest *finite-sample* coverage guarantee rather than an asymptotic one
that happens to be close on typical data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.intervals import Interval


def _finite_sample_quantile(scores: np.ndarray, level: float) -> tuple[float, bool]:
    """The finite-sample-corrected empirical quantile of a nonconformity score array.

    Uses the standard split-conformal index ``ceil((n+1)(1-alpha))`` (1-indexed) rather than
    the naive ``level``-th quantile — this is what gives conformal prediction its finite-sample
    guarantee instead of just an asymptotic approximation.

    Returns ``(q, exceeded)``. ``exceeded=True`` means ``n`` is too small for the requested
    level to be achievable at all (the corrected index would exceed the sample) — in that case
    ``q`` falls back to the single widest observed score, which is the most conservative
    interval this calibration set can support. Silently returning a narrower value here would
    silently under-cover, which is exactly the failure mode this module exists to prevent.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = scores.size
    if n == 0:
        raise ValueError("no calibration scores to compute a quantile from")
    alpha = 1.0 - level
    idx = math.ceil((n + 1) * (1.0 - alpha))
    if idx >= n:
        return float(scores[-1]), True
    idx = max(idx, 1)
    return float(scores[idx - 1]), False


def conformal_quantiles(
    y_true,
    y_pred,
    prev_close,
    symbols,
    level: float = 0.80,
    pooled: bool = True,
    min_n: int = 30,
) -> dict:
    """Split-conformal quantiles, plug-compatible with :func:`src.intervals.apply`.

    Nonconformity score: ``s = |y_true - y_pred| / prev_close``. The resulting interval is
    symmetric: ``point +/- q * prev_close`` (unlike the asymmetric per-symbol residual
    quantiles this replaces).

    ``pooled=True`` (the default, and what the backfill re-run actually uses — see the plan's
    diagnosis for why) computes ONE calibration quantile across every symbol's scores
    together. ``pooled=False`` computes a separate quantile per symbol, falling back to the
    pooled quantile for any symbol with fewer than ``min_n`` calibration points — never
    raising, never silently emitting a too-narrow per-symbol interval from too little data.

    Returns the same shape :func:`src.intervals.apply` expects:
    ``{symbol: (q_lo, q_hi), "__pooled__": (q_lo, q_hi), "__level__": level}``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    symbols = np.asarray(symbols).ravel()

    scores = np.abs(y_true - y_pred) / prev_close

    q_pooled, _ = _finite_sample_quantile(scores, level)
    out: dict = {"__pooled__": (-q_pooled, q_pooled), "__level__": level}

    for symbol in np.unique(symbols):
        if pooled:
            out[symbol] = (-q_pooled, q_pooled)
            continue
        mask = symbols == symbol
        n = int(mask.sum())
        if n < min_n:
            out[symbol] = (-q_pooled, q_pooled)
            continue
        q_symbol, _ = _finite_sample_quantile(scores[mask], level)
        out[symbol] = (-q_symbol, q_symbol)

    return out


@dataclass(frozen=True)
class AdaptiveQuantile:
    """A single pooled conformal quantile calibrated on volatility-normalized scores."""

    q: float
    level: float
    n: int
    exceeded: bool  # True if n was too small to hit `level` exactly (see _finite_sample_quantile)


def adaptive_conformal_quantile(
    y_true, y_pred, prev_close, vol_hat, level: float = 0.80
) -> AdaptiveQuantile:
    """Calibrate a single, pooled, volatility-normalized conformal quantile.

    Nonconformity score: ``s = |y_true - y_pred| / (prev_close * vol_hat)``. Pooling across
    symbols is not just a sample-size convenience here — it's necessary, since normalizing by
    each row's own trailing volatility already puts every symbol's score on a comparable scale
    regardless of price level or typical volatility, which per-symbol quantiles would not add
    anything over.

    Rows with a non-finite (``NaN``) ``vol_hat`` — insufficient history to estimate volatility
    — are dropped from calibration rather than propagating a NaN quantile.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    vol_hat = np.asarray(vol_hat, dtype=float).ravel()

    valid = np.isfinite(vol_hat) & (vol_hat > 0)
    if not valid.any():
        raise ValueError("no rows with a usable (finite, positive) volatility estimate")

    scores = np.abs(y_true[valid] - y_pred[valid]) / (prev_close[valid] * vol_hat[valid])
    q, exceeded = _finite_sample_quantile(scores, level)
    return AdaptiveQuantile(q=q, level=level, n=int(valid.sum()), exceeded=exceeded)


def apply_adaptive(y_pred, prev_close, vol_hat, quantile: AdaptiveQuantile) -> Interval:
    """Apply a calibrated :class:`AdaptiveQuantile` to new points.

    ``half_width = quantile.q * prev_close * vol_hat`` — this is what makes the interval
    *adaptive*: the same calibrated ``q`` produces a wide interval when ``vol_hat`` (the
    current volatility regime) is high, and a tight one when it's low.
    """
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    prev_close = np.asarray(prev_close, dtype=float).ravel()
    vol_hat = np.asarray(vol_hat, dtype=float).ravel()

    if not np.all(np.isfinite(vol_hat) & (vol_hat > 0)):
        raise ValueError("apply_adaptive requires a finite, positive vol_hat for every row")

    half_width = quantile.q * prev_close * vol_hat
    return Interval(lo=y_pred - half_width, hi=y_pred + half_width, level=quantile.level)
