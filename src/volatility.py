"""Volatility estimation, for scaling conformal intervals to the current regime.

Volatility clustering — calm periods and turbulent periods each persist for a while — is one
of the most robust findings in finance, unlike price direction. A model that cannot predict
which way a price moves can still predict *how much* it is likely to move, and that is exactly
what an adaptive prediction interval needs.

Two estimators, both free:

- :func:`ewma_volatility` — RiskMetrics-standard exponentially-weighted volatility. Always
  available, no dependency beyond numpy.
- :func:`garch11_volatility` — a GARCH(1,1) one-step-ahead forecast via the free, open-source
  ``arch`` package. Returns ``None`` (never raises) if the fit fails or ``arch`` isn't
  installed — callers must fall back to EWMA rather than block the pipeline on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LAMBDA = 0.94  # RiskMetrics standard decay factor
MIN_RETURNS = 21


def ewma_volatility(returns, lam: float = DEFAULT_LAMBDA) -> float:
    """RiskMetrics EWMA daily volatility (standard deviation of returns), using every return
    supplied and weighting the most recent one most heavily.

    Assumes zero mean, per the RiskMetrics convention for daily equity returns — the mean
    return is small enough relative to its variance that estimating it separately just adds
    noise.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return float("nan")
    n = returns.size
    weights = (1 - lam) * lam ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()
    variance = float(np.sum(weights * returns**2))
    return float(np.sqrt(variance))


def rolling_ewma_volatility(
    returns: pd.Series, lam: float = DEFAULT_LAMBDA, min_periods: int = MIN_RETURNS
) -> pd.Series:
    """Causal EWMA volatility at every point in time: the value at index ``i`` uses only
    ``returns`` up to and including index ``i`` — never a later one. The first
    ``min_periods - 1`` entries are ``NaN`` rather than an unstable early estimate.
    """
    alpha = 1.0 - lam
    ewm_var = returns.pow(2).ewm(alpha=alpha, adjust=True, min_periods=min_periods).mean()
    return np.sqrt(ewm_var)


def garch11_volatility(returns) -> float | None:
    """One-step-ahead conditional volatility forecast from a GARCH(1,1) fit.

    Returns ``None`` — never raises — if ``arch`` isn't installed, there isn't enough data to
    fit stably, or the optimizer fails to converge. Callers must treat ``None`` as "fall back
    to :func:`ewma_volatility`", not as an error.
    """
    try:
        from arch import arch_model
    except ImportError:
        return None

    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size < 50:
        return None

    try:
        # arch's optimizer is numerically happier with returns scaled to roughly unit
        # variance; percent returns (x100) are the documented convention for daily equity data.
        model = arch_model(returns * 100.0, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        fit = model.fit(disp="off", show_warning=False)
        forecast = fit.forecast(horizon=1, reindex=False)
        variance_pct2 = float(forecast.variance.to_numpy()[-1, 0])
        if not np.isfinite(variance_pct2) or variance_pct2 <= 0:
            return None
        return float(np.sqrt(variance_pct2)) / 100.0
    except Exception:  # noqa: BLE001 - any fit failure falls back to EWMA, never propagates
        return None


def horizon_scale(daily_vol: float, horizon: int) -> float:
    """Scale a daily volatility estimate to a ``horizon``-day estimate under the standard
    iid-returns approximation: volatility grows with the square root of time."""
    return float(daily_vol) * float(np.sqrt(horizon))
