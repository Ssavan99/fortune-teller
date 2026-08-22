"""Trivial forecasters that any useful model has to beat.

All three are fitted on the training partition only. Persistence has nothing to fit; drift and
AR do, and estimating either of them on the full series would leak the test period into the
baseline — which would flatter the LSTM by inflating the number it is compared against.

Each function returns predictions aligned to ``target`` rows: for a row dated *t*, the
prediction of that row's close, made from information available at *t−1*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Forecast:
    """Predictions for one symbol over one period, with the anchor price for each row."""

    symbol: str
    dates: pd.Series
    y_true: np.ndarray
    y_pred: np.ndarray
    last_close: np.ndarray


def _series(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    part = df[df["symbol"] == symbol].sort_values("date", ignore_index=True)
    if len(part) < 2:
        raise ValueError(f"{symbol}: need at least two bars")
    return part


def persistence(df: pd.DataFrame, symbol: str, period: pd.DataFrame) -> Forecast:
    """Tomorrow's close is today's close.

    The baseline. Nothing is fitted, so ``df`` is used only to supply the previous close for
    the first row of ``period`` — without it that row would have to be dropped.
    """
    full = _series(df, symbol)
    full["prev_close"] = full["close"].shift(1)

    rows = full[full["date"].isin(set(period["date"]))].dropna(subset=["prev_close"])
    return Forecast(
        symbol=symbol,
        dates=rows["date"],
        y_true=rows["close"].to_numpy(),
        y_pred=rows["prev_close"].to_numpy(),
        last_close=rows["prev_close"].to_numpy(),
    )


def drift(df: pd.DataFrame, symbol: str, period: pd.DataFrame, train: pd.DataFrame) -> Forecast:
    """Random walk with drift: tomorrow's close is today's close plus the mean daily change.

    The drift term is the mean first difference **over the training partition only**.
    """
    train_part = _series(train, symbol)
    mu = float(np.mean(np.diff(train_part["close"].to_numpy())))

    base = persistence(df, symbol, period)
    return Forecast(
        symbol=base.symbol,
        dates=base.dates,
        y_true=base.y_true,
        y_pred=base.y_pred + mu,
        last_close=base.last_close,
    )


def autoregressive(
    df: pd.DataFrame,
    symbol: str,
    period: pd.DataFrame,
    train: pd.DataFrame,
    order: int = 5,
) -> Forecast:
    """AR(p) on daily returns, fitted by least squares on the training partition only.

    Predicts the next return from the previous ``order`` returns, then converts back to a
    price: ``close_hat = prev_close * (1 + r_hat)``. Working in returns rather than levels
    keeps the design matrix stationary; fitting AR on raw prices would mostly recover a
    coefficient of one on the last lag, which is persistence with extra steps.
    """
    full = _series(df, symbol)
    full["ret"] = full["close"].pct_change()
    full["prev_close"] = full["close"].shift(1)

    def design(frame: pd.DataFrame):
        lags = [frame["ret"].shift(i) for i in range(1, order + 1)]
        x = pd.concat(lags, axis=1)
        x.columns = [f"lag{i}" for i in range(1, order + 1)]
        return x

    x_all = design(full)
    usable = x_all.notna().all(axis=1) & full["ret"].notna()

    train_dates = set(train[train["symbol"] == symbol]["date"])
    fit_mask = usable & full["date"].isin(train_dates)

    x_fit = np.column_stack([np.ones(int(fit_mask.sum())), x_all[fit_mask].to_numpy()])
    y_fit = full.loc[fit_mask, "ret"].to_numpy()
    coefs, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    period_dates = set(period["date"])
    pred_mask = usable & full["date"].isin(period_dates) & full["prev_close"].notna()

    x_pred = np.column_stack([np.ones(int(pred_mask.sum())), x_all[pred_mask].to_numpy()])
    ret_hat = x_pred @ coefs

    rows = full[pred_mask]
    prev = rows["prev_close"].to_numpy()
    return Forecast(
        symbol=symbol,
        dates=rows["date"],
        y_true=rows["close"].to_numpy(),
        y_pred=prev * (1.0 + ret_hat),
        last_close=prev,
    )
