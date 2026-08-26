"""Phase B: volatility as its own, first-class forecast — scored with the same discipline as
the price arms, against the same kind of trivial baseline.

Run: ``python -m scripts.evaluate_volatility``

Unlike next-day price direction, volatility clustering is one of the most robust findings in
finance: calm periods and turbulent periods each tend to persist. This script tests that
directly rather than leaving it as an implementation detail buried inside the conformal
interval code — if this project is going to report "price skill is ~zero" honestly, it should
also report the one place a real, positive skill is actually expected, with equal rigor.

Both arms forecast a scalar, in the same units (annualized daily-return volatility, i.e. the
standard deviation of daily returns times sqrt(252) — a familiar, interpretable unit), for the
volatility realized over the 21 trading days AFTER ``as_of``:

* **Baseline — persistence of volatility.** The realized volatility over the 21 trading days
  BEFORE ``as_of``. The trivial "tomorrow looks like today" forecast, exactly as the price
  arms are baselined against "tomorrow's price is today's".
* **Model — EWMA.** :func:`src.volatility.ewma_volatility` (RiskMetrics lambda=0.94) computed
  from returns strictly before ``as_of``.

No LSTM is trained for this — the forecast is deliberately the cheap, well-understood EWMA
estimator, not a neural net, because the point of this arm is to show that a real edge exists
using an honest, simple method, not to show off a bigger model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics, volatility
from src.data import prices

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "volatility_evaluation.json"

WINDOW = 21  # trading days, matches the scoreboard's forecast horizon
ANNUALIZE = np.sqrt(252)
SERIES_STRIDE = 5  # ~one point per trading week — keeps the site payload small and legible


def _realized_vol(returns: pd.Series, start: int, end: int) -> float:
    """Annualized std of daily returns over the half-open window [start, end)."""
    window = returns.iloc[start:end]
    if len(window) < WINDOW:
        return float("nan")
    return float(window.std() * ANNUALIZE)


def evaluate_symbol(part: pd.DataFrame) -> list[dict]:
    """One row per valid as-of date for this symbol: baseline forecast, model forecast, and
    the realized volatility that actually followed — all three in the same annualized units.

    An as-of date at position i needs at least WINDOW days of return history strictly before
    it (for both the baseline and the EWMA estimator) and WINDOW days of returns strictly
    after it (to know what actually happened) — so valid positions run from WINDOW to
    n - WINDOW - 1.
    """
    part = part.sort_values("date").reset_index(drop=True)
    returns = part["close"].pct_change()
    dates = part["date"].to_numpy()

    rows = []
    n = len(part)
    for i in range(WINDOW, n - WINDOW):
        baseline = _realized_vol(returns, i - WINDOW, i)
        actual = _realized_vol(returns, i, i + WINDOW)
        past_returns = returns.iloc[:i].dropna().to_numpy()
        if past_returns.size < WINDOW or not np.isfinite(baseline) or not np.isfinite(actual):
            continue
        daily_ewma = volatility.ewma_volatility(past_returns)
        if not np.isfinite(daily_ewma):
            continue
        model = float(daily_ewma * ANNUALIZE)
        rows.append(
            {
                "date": str(pd.Timestamp(dates[i]).date()),
                "baseline_forecast": baseline,
                "model_forecast": model,
                "actual": actual,
            }
        )
    return rows


def main() -> None:
    df = prices.load()
    per_symbol: dict[str, list[dict]] = {}
    for symbol, part in df.groupby("symbol"):
        per_symbol[symbol] = evaluate_symbol(part)

    all_rows = [{"symbol": s, **r} for s, rows in per_symbol.items() for r in rows]
    if not all_rows:
        raise RuntimeError("no volatility rows evaluated — price history too short")

    actual = np.array([r["actual"] for r in all_rows])
    baseline = np.array([r["baseline_forecast"] for r in all_rows])
    model = np.array([r["model_forecast"] for r in all_rows])

    baseline_rmse = metrics.rmse(actual, baseline)
    model_rmse = metrics.rmse(actual, model)
    skill = metrics.skill_score(model_rmse, baseline_rmse)

    per_ticker = {}
    series = {}
    for symbol in sorted(per_symbol):
        rows = per_symbol[symbol]
        if not rows:
            continue
        a = np.array([r["actual"] for r in rows])
        b = np.array([r["baseline_forecast"] for r in rows])
        m = np.array([r["model_forecast"] for r in rows])
        b_rmse, m_rmse = metrics.rmse(a, b), metrics.rmse(a, m)
        per_ticker[symbol] = {
            "n": len(rows),
            "baseline_rmse": b_rmse,
            "model_rmse": m_rmse,
            "skill_vs_persistence": metrics.skill_score(m_rmse, b_rmse),
        }
        # Downsampled for the site's "predicted vs realized volatility over time" chart — the
        # full series (n=1389/ticker) is unnecessary for a reader to see the pattern and would
        # bloat the page payload for no benefit. Skill/RMSE above are still computed from the
        # full, non-downsampled series.
        series[symbol] = [
            {
                "date": r["date"],
                "baseline_forecast": r["baseline_forecast"],
                "model_forecast": r["model_forecast"],
                "actual": r["actual"],
            }
            for r in rows[::SERIES_STRIDE]
        ]

    summary = {
        "window_days": WINDOW,
        "units": "annualized daily-return volatility (std * sqrt(252))",
        "n": len(all_rows),
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "skill_vs_persistence": skill,
        "tickers_beating_persistence": sum(
            1 for v in per_ticker.values() if v["skill_vs_persistence"] > 0
        ),
        "n_tickers": len(per_ticker),
        "per_ticker": per_ticker,
        "series_stride_days": SERIES_STRIDE,
        "series": series,
    }

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"n={summary['n']}  baseline RMSE={baseline_rmse:.4f}  model RMSE={model_rmse:.4f}")
    print(f"skill vs persistence-of-volatility: {skill:+.4f}")
    print(
        f"beats persistence on {summary['tickers_beating_persistence']}/{summary['n_tickers']} "
        "tickers"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
