"""The rolling forecast engine: one monthly cycle, replayable identically for backfill or live.

:func:`run_cycle` is the single code path both the historical backfill (Phase 3) and the live
monthly run (Phase 5) go through. That is deliberate — if backfill and live diverged even
slightly in how they build features, split data, or fit the scaler, a skill difference between
them would prove nothing about the model and everything about the code paths being different.

**The as-of boundary is the whole point of this module.** ``history = prices_df[prices_df.date
< as_of]`` is the only place ``as_of`` is applied, and everything downstream — the train/val
split, the scaler, the model, the residual quantiles, the final prediction window — sees only
``history``. Nothing below that line may reach back into ``prices_df`` again. This is asserted
by ``tests/test_rolling.py::test_as_of_boundary_holds``, which reruns a cycle after multiplying
every future close by 100 and requires byte-identical predictions.

Returned records carry no ``mode`` field. A single cycle has no way of knowing whether it is
being replayed against known history (``"backtest"``) or run against the live edge of the data
before the outcome exists (``"live"``) — that label belongs to the caller, which does know, and
stamping it here would just move the one fact this whole project depends on into a place that
can't see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from src import conformal, intervals, sequences, volatility
from src.preprocessing import fit_on_train
from src.train import TrainConfig, predict, train

LOOKBACK = 20
LEVEL = 0.80
_US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
VAL_MONTHS = 6

INTERVAL_METHODS = ("quantile", "conformal", "conformal_ewma")

VALID_MODES = ("backtest", "live")


def stamp_mode(records: list[dict], mode: str) -> list[dict]:
    """Set ``record["mode"] = mode`` on every record, after validating ``mode`` itself.

    A one-line loop is easy to get right by hand, which is exactly the problem: a copy-pasted
    call site that hardcodes the wrong string, or a typo that silently produces a third mode
    value, would corrupt the one fact ("was this predicted before or after the outcome
    existed?") the whole scoreboard's honesty rests on. Routing every call site through this
    function means that mistake fails loudly instead of shipping.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    for record in records:
        record["mode"] = mode
    return records


@dataclass
class _RollingSplit:
    """The train/val split :func:`sequences.build` needs. No test partition exists yet —
    "test" is whatever hasn't happened, which is exactly what this module is forecasting."""

    train: pd.DataFrame
    val: pd.DataFrame


def _chronological_split(history: pd.DataFrame) -> _RollingSplit:
    """Last six months of history -> validation; everything before -> train."""
    val_end = history["date"].max()
    train_end = val_end - pd.DateOffset(months=VAL_MONTHS)
    return _RollingSplit(
        train=history[history["date"] <= train_end].copy(),
        val=history[history["date"] > train_end].copy(),
    )


def target_date_for(prices_df: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> pd.Timestamp:
    """The trading day ``horizon`` sessions on or after ``as_of``.

    A backfill run already has the future in ``prices_df`` — it is replaying real history — so
    the exact trading calendar is used: the ``horizon``-th actual trading date at or after
    ``as_of``. A live run does not have that, since the future has not happened yet, so this
    falls back to a US-federal-holiday-aware business-day approximation
    (:data:`_US_BUSINESS_DAY`) rather than a plain ``pd.bdate_range``, which would silently
    count bank holidays as trading sessions and land the target a session early — e.g. Labor
    Day 2026 pushes the true 21st session from 2026-09-22 to 2026-09-23. This still is not the
    exact NYSE calendar (it doesn't know about market-only closures like Good Friday), so it
    can still land a session off in rare cases — that costs nothing, since scoring (Phase 5)
    locates the nearest actual close on or after this date rather than requiring an exact match.

    ``as_of`` is normalized to midnight for the same reason :func:`run_cycle` does it: a
    caller-supplied timestamp with a nonzero time-of-day would otherwise pull that same
    calendar day out of the "future" set entirely, shifting the target date by a session.
    """
    as_of = pd.Timestamp(as_of).normalize()
    future = prices_df.loc[prices_df["date"] >= as_of, "date"]
    future_dates = sorted(pd.Series(future.unique()))
    if len(future_dates) >= horizon:
        return pd.Timestamp(future_dates[horizon - 1])
    return pd.bdate_range(start=as_of, periods=horizon, freq=_US_BUSINESS_DAY)[-1]


def _final_window_predictions(
    history: pd.DataFrame,
    split: _RollingSplit,
    model: torch.nn.Module,
    norm,
    lookback: int,
) -> dict[str, tuple[float, float]]:
    """Point forecast (in dollars) for each symbol from the most recent ``lookback`` days.

    Mirrors :func:`sequences.build`'s per-symbol scaler-fitting exactly (same ``train_dates``,
    same feature columns) so the window fed to the model here sits in the identical input
    distribution the model was trained on. Returns ``{symbol: (point, prev_close)}``.
    """
    train_dates = set(split.train["date"])
    out: dict[str, tuple[float, float]] = {}

    model.eval()
    for symbol in sorted(history["symbol"].unique()):
        part = history[history["symbol"] == symbol].sort_values("date", ignore_index=True)
        if len(part) < lookback:
            continue
        train_rows = part[part["date"].isin(train_dates)]
        if train_rows.empty:
            continue

        scaler = fit_on_train(train_rows[list(sequences.FEATURES)].to_numpy(dtype=float))
        window = part[list(sequences.FEATURES)].to_numpy(dtype=float)[-lookback:]
        scaled = scaler.transform(window)

        x = torch.tensor(scaled[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            raw = model(x).numpy()
        ret_pred = float(norm.inverse(raw)[0])

        prev_close = float(part["close"].iloc[-1])
        point = prev_close * (1.0 + ret_pred)
        out[symbol] = (point, prev_close)

    return out


def _anchor_dates(
    part_dates_sorted: np.ndarray, target_dates: np.ndarray, horizon: int
) -> np.ndarray:
    """Exact per-row lookup of the window's last date, given each row's target date — the
    trading day exactly ``horizon`` sessions before it, in that symbol's own calendar. Used
    only to align a volatility estimate to the correct point in time; never used to build a
    prediction itself, so it carries no leakage risk of its own.
    """
    idx_map = {d: i for i, d in enumerate(part_dates_sorted)}
    return np.array([part_dates_sorted[idx_map[td] - horizon] for td in target_dates])


def _validation_vol_hat(history: pd.DataFrame, val_seqs, horizon: int) -> np.ndarray:
    """Per-row, horizon-scaled trailing EWMA volatility for every validation row, using only
    that row's own symbol's returns strictly before its own anchor date (the window's last
    day) — never later, and never another symbol's returns. This is what makes the resulting
    calibration genuinely track the volatility regime at each point in validation, rather than
    a single "current" estimate applied uniformly (which would cancel out of the calibration
    entirely and do nothing — see the module docstring in ``src/conformal.py``).
    """
    vol_hat = np.full(len(val_seqs), np.nan)
    for symbol in np.unique(val_seqs.symbols):
        part = history[history["symbol"] == symbol].sort_values("date")
        part_dates = part["date"].to_numpy()
        returns = part["close"].pct_change()
        ewma_series = volatility.rolling_ewma_volatility(returns).to_numpy()
        date_to_pos = {d: i for i, d in enumerate(part_dates)}

        mask = val_seqs.symbols == symbol
        rows = np.where(mask)[0]
        anchors = _anchor_dates(part_dates, val_seqs.dates[mask], horizon)
        for row, anchor in zip(rows, anchors, strict=True):
            pos = date_to_pos.get(anchor)
            if pos is None or pos < 0:
                continue
            daily_vol = ewma_series[pos]
            if np.isfinite(daily_vol):
                vol_hat[row] = volatility.horizon_scale(daily_vol, horizon)
    return vol_hat


def _live_vol_hat(history: pd.DataFrame, symbols, horizon: int) -> dict[str, float]:
    """Horizon-scaled trailing EWMA volatility for each symbol as of the live prediction
    point (the most recent date in ``history``) — the volatility estimate that governs the
    width of the interval actually being predicted right now.

    A symbol with too little history for a stable estimate (should not happen once the
    backfill is past its first few months, given ``MIN_RETURNS`` in ``src/volatility.py``)
    falls back to the median of the other symbols' estimates in this same cycle, rather than
    raising and losing every symbol's prediction over one thin history.
    """
    out: dict[str, float] = {}
    for symbol in symbols:
        part = history[history["symbol"] == symbol].sort_values("date")
        returns = part["close"].pct_change()
        ewma_series = volatility.rolling_ewma_volatility(returns)
        daily_vol = float(ewma_series.iloc[-1]) if len(ewma_series) else float("nan")
        out[symbol] = (
            volatility.horizon_scale(daily_vol, horizon)
            if np.isfinite(daily_vol)
            else float("nan")
        )

    valid = [v for v in out.values() if np.isfinite(v)]
    if valid:
        fallback = float(np.median(valid))
        for symbol, v in out.items():
            if not np.isfinite(v):
                out[symbol] = fallback
    return out


def _record(
    as_of, target_date, symbol, model_name, point, lo, hi, level, created_utc, interval_method
) -> dict:
    return {
        "as_of": str(pd.Timestamp(as_of).date()),
        "target_date": str(pd.Timestamp(target_date).date()),
        "symbol": symbol,
        "model": model_name,
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "level": level,
        "actual": None,
        "covered": None,
        "abs_error": None,
        "created_utc": created_utc,
        "interval_method": interval_method,
    }


def run_cycle(
    prices_df: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int = 21,
    seed: int = 20260822,
    *,
    train_config_overrides: dict | None = None,
    method: str = "quantile",
) -> list[dict]:
    """One full forecast cycle. MUST NOT read any row dated >= as_of, except the future
    ``date`` column alone (never a price column) to locate ``target_date`` — see module
    docstring for why that one exception is safe.

    Returns one record per (symbol, model) for ``model in ("lstm", "persistence")``. Every
    record's ``actual``/``covered``/``abs_error`` are ``None`` — this function only predicts,
    never scores. Records carry no ``mode``; the caller stamps that.

    ``train_config_overrides`` lets a caller shrink the network for speed (the test suite uses
    this — see ``fast_config`` in ``tests/test_train.py`` for the same pattern elsewhere in
    this repo). Production callers never pass it.

    ``method`` selects how the prediction interval around each point forecast is built —
    recorded on every returned row as ``interval_method`` so historical rows always show which
    method produced them, even as the default changes over time:

    * ``"quantile"`` — the original empirical per-symbol residual quantiles
      (:func:`src.intervals.residual_quantiles`). Undercovers (~67% realized vs 80% nominal);
      kept only as the named baseline arm, per this project's anti-overfitting protocol.
    * ``"conformal"`` — split-conformal, pooled across symbols
      (:func:`src.conformal.conformal_quantiles`). Fixes marginal coverage.
    * ``"conformal_ewma"`` — split-conformal with the nonconformity score normalized by each
      row's own trailing EWMA volatility, so the interval adapts to the volatility regime at
      prediction time (:func:`src.conformal.adaptive_conformal_quantile`).
    """
    if method not in INTERVAL_METHODS:
        raise ValueError(f"method must be one of {INTERVAL_METHODS}, got {method!r}")
    # Normalized to midnight: prices_df["date"] is always midnight (src/data/prices.py), and
    # comparing against a caller-supplied as_of with a nonzero time-of-day (e.g. the very
    # natural pd.Timestamp.now()) would let that day's own bar slip into `history` under `<`.
    as_of = pd.Timestamp(as_of).normalize()
    history = prices_df[prices_df["date"] < as_of].copy()
    if history.empty:
        raise ValueError(f"no history available before as_of={as_of.date()}")

    split = _chronological_split(history)
    if split.train.empty or split.val.empty:
        raise ValueError(
            f"insufficient history before as_of={as_of.date()} for a {VAL_MONTHS}-month "
            "validation split"
        )

    config_kwargs = {"target": "return", "lookback": LOOKBACK, "seed": seed}
    config_kwargs.update(train_config_overrides or {})
    config = TrainConfig(**config_kwargs)

    built = sequences.build(
        history, split, lookback=config.lookback, target="return", horizon=horizon
    )
    if "train" not in built or "val" not in built:
        raise ValueError(
            f"insufficient history before as_of={as_of.date()} to build any train/val "
            f"sequences at lookback={config.lookback}, horizon={horizon}"
        )
    train_seqs, val_seqs = built["train"], built["val"]
    model, norm, config = train(train_seqs, val_seqs, config, verbose=False)

    val_pred_dollars = sequences.to_dollars(predict(model, val_seqs, norm), val_seqs, "return")

    live_points = _final_window_predictions(history, split, model, norm, config.lookback)
    if not live_points:
        raise ValueError(
            f"no symbol has enough history before as_of={as_of.date()} for a "
            f"{config.lookback}-day prediction window"
        )
    target_date = target_date_for(prices_df, as_of, horizon)
    created_utc = pd.Timestamp.now(tz="UTC").isoformat()

    if method == "quantile":
        lstm_quantiles = intervals.residual_quantiles(
            val_seqs.y_close, val_pred_dollars, val_seqs.prev_close, val_seqs.symbols, level=LEVEL
        )
        persistence_quantiles = intervals.residual_quantiles(
            val_seqs.y_close, val_seqs.prev_close, val_seqs.prev_close, val_seqs.symbols,
            level=LEVEL,
        )
    elif method == "conformal":
        lstm_quantiles = conformal.conformal_quantiles(
            val_seqs.y_close, val_pred_dollars, val_seqs.prev_close, val_seqs.symbols, level=LEVEL
        )
        persistence_quantiles = conformal.conformal_quantiles(
            val_seqs.y_close, val_seqs.prev_close, val_seqs.prev_close, val_seqs.symbols,
            level=LEVEL,
        )
    else:  # "conformal_ewma"
        vol_hat_val = _validation_vol_hat(history, val_seqs, horizon)
        lstm_q = conformal.adaptive_conformal_quantile(
            val_seqs.y_close, val_pred_dollars, val_seqs.prev_close, vol_hat_val, level=LEVEL
        )
        persistence_q = conformal.adaptive_conformal_quantile(
            val_seqs.y_close, val_seqs.prev_close, val_seqs.prev_close, vol_hat_val, level=LEVEL
        )
        vol_hat_live = _live_vol_hat(history, sorted(live_points), horizon)

    records: list[dict] = []
    for symbol in sorted(live_points):
        lstm_point, prev_close = live_points[symbol]
        sym_arr = np.array([symbol])
        prev_arr = np.array([prev_close])

        if method in ("quantile", "conformal"):
            lstm_interval = intervals.apply(
                np.array([lstm_point]), prev_arr, sym_arr, lstm_quantiles, level=LEVEL
            )
            persistence_interval = intervals.apply(
                prev_arr, prev_arr, sym_arr, persistence_quantiles, level=LEVEL
            )
        else:  # "conformal_ewma"
            vh = np.array([vol_hat_live[symbol]])
            lstm_interval = conformal.apply_adaptive(np.array([lstm_point]), prev_arr, vh, lstm_q)
            persistence_interval = conformal.apply_adaptive(prev_arr, prev_arr, vh, persistence_q)

        records.append(
            _record(
                as_of, target_date, symbol, "lstm", lstm_point,
                lstm_interval.lo[0], lstm_interval.hi[0], LEVEL, created_utc, method,
            )
        )
        records.append(
            _record(
                as_of, target_date, symbol, "persistence", prev_close,
                persistence_interval.lo[0], persistence_interval.hi[0], LEVEL, created_utc, method,
            )
        )

    return records


def score_record(record: dict, prices_df: pd.DataFrame) -> dict:
    """Fill ``actual``/``covered``/``abs_error`` for one record, if its target has matured.

    Returns a **new** dict; never mutates the input, and never modifies ``point``, ``lo``,
    ``hi``, ``as_of``, or ``target_date``. Two safety properties fall out of that:

    * A record that is already scored (``actual is not None``) is returned unchanged — scoring
      is idempotent, so calling this on the same record twice can never revise a result after
      the fact.
    * A record whose target date has no actual close yet (the future genuinely hasn't
      happened) is returned unchanged with ``actual`` still ``None`` — never a fabricated or
      interpolated value.

    The actual is taken from the nearest real trading day **on or after** ``target_date``
    rather than requiring an exact match, since ``target_date`` itself may be a
    business-day-calendar approximation (see :func:`target_date_for`).
    """
    if record["actual"] is not None:
        return dict(record)

    symbol = record["symbol"]
    target = pd.Timestamp(record["target_date"])
    candidates = prices_df[(prices_df["symbol"] == symbol) & (prices_df["date"] >= target)]
    if candidates.empty:
        return dict(record)

    actual = float(candidates.sort_values("date").iloc[0]["close"])
    out = dict(record)
    out["actual"] = actual
    # An LLM abstention has point/lo/hi = None: no numeric claim was made, so there is nothing
    # to check coverage or error against. Recording the actual price is still useful context —
    # but covered/abs_error must stay None rather than raise on `None <= actual` or silently
    # coerce None into 0.
    if record["lo"] is None or record["hi"] is None or record["point"] is None:
        out["covered"] = None
        out["abs_error"] = None
    else:
        out["covered"] = bool(record["lo"] <= actual <= record["hi"])
        out["abs_error"] = float(abs(actual - record["point"]))
    return out
