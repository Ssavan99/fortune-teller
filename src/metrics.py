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
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    last_close = np.asarray(last_close, dtype=float).ravel()

    predicted_move = y_pred - last_close
    actual_move = y_true - last_close

    called = predicted_move != 0
    if not called.any():
        return None

    correct = np.sign(predicted_move[called]) == np.sign(actual_move[called])
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


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Fractional RMSE reduction against a baseline.

    ``1 - model_rmse / baseline_rmse``. A value of 0.10 means the model's RMSE is 10% lower
    than the baseline's. Zero means no improvement; negative means the baseline is better.
    """
    if baseline_rmse <= 0:
        raise ValueError("baseline RMSE must be positive")
    return float(1.0 - model_rmse / baseline_rmse)
