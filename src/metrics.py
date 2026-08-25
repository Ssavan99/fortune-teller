"""Error metrics, reported in dollars.

Every number here is in the units of the price series. Scaled-unit metrics — a MAE of
``0.0258`` on MinMax-normalised closes — cannot be compared across tickers, cannot be compared
to a baseline, and cannot be sanity-checked by a reader. They are not reported anywhere in
this repository.

The headline number is :func:`skill_score`: the fractional reduction in RMSE relative to the
persistence baseline. Positive means the model beat "tomorrow's close is today's close".
Zero or negative means it did not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Scores:
    """Error summary for one set of predictions, in dollars except where noted."""

    rmse: float
    mae: float
    mape: float  # percent
    directional_accuracy: float | None  # fraction in [0, 1], None when undefined
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _as_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no observations to score")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error. Undefined at zero prices, which cannot occur here."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when a true price is zero")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def directional_accuracy(y_true, y_pred, last_close) -> float | None:
    """Fraction of days where the predicted direction of change matched the actual.

    ``last_close`` is the price the move is measured from — the previous close.

    Returns ``None`` when the model never predicts a move, which is exactly the case for the
    persistence baseline. Scoring persistence's direction as though a flat prediction were a
    call would be meaningless, so it is reported as undefined rather than as 0 or 50%.

    Days where the price did not actually move are also excluded. ``np.sign(0)`` matches no
    nonzero prediction, so counting them would mark every such day wrong regardless of what
    the model said — a small but systematic penalty on a day with no direction to call.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    last_close = np.asarray(last_close, dtype=float).ravel()

    predicted_move = y_pred - last_close
    actual_move = y_true - last_close

    scorable = (predicted_move != 0) & (actual_move != 0)
    if not scorable.any():
        return None

    correct = np.sign(predicted_move[scorable]) == np.sign(actual_move[scorable])
    return float(np.mean(correct))


def score(y_true, y_pred, last_close=None) -> Scores:
    """Full error summary for one set of predictions."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return Scores(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
        directional_accuracy=(
            directional_accuracy(y_true, y_pred, last_close) if last_close is not None else None
        ),
        n=int(y_true.size),
    )


def coverage(y_true, lo, hi) -> float:
    """Fraction of observations whose true value falls inside ``[lo, hi]``, inclusive."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("no observations to score")
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def mean_interval_width(lo, hi) -> float:
    """Mean interval width in dollars. Reported alongside coverage — never alone.

    Coverage in isolation is trivially gamed: predict $0-$10,000 and score 100%. Width is what
    makes a wide, useless interval visibly worse than a tight, well-calibrated one.
    """
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if lo.size == 0:
        raise ValueError("no observations to score")
    return float(np.mean(hi - lo))


def interval_score(y_true, lo, hi, level: float) -> float:
    """Gneiting-Raftery interval score. Lower is better; penalises both width and misses.

    ``alpha = 1 - level``::

        IS = (hi - lo)
           + (2 / alpha) * (lo - y) * [y < lo]
           + (2 / alpha) * (y - hi) * [y > hi]

    Returns the mean over observations.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("no observations to score")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")

    alpha = 1.0 - level
    below = y_true < lo
    above = y_true > hi

    penalty_lo = (2.0 / alpha) * (lo - y_true) * below
    penalty_hi = (2.0 / alpha) * (y_true - hi) * above
    return float(np.mean((hi - lo) + penalty_lo + penalty_hi))


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Fractional RMSE reduction against a baseline.

    ``1 - model_rmse / baseline_rmse``. A value of 0.10 means the model's RMSE is 10% lower
    than the baseline's. Zero means no improvement; negative means the baseline is better.
    """
    if baseline_rmse <= 0:
        raise ValueError("baseline RMSE must be positive")
    return float(1.0 - model_rmse / baseline_rmse)
