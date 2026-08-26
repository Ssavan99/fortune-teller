"""Diagnose why the backtest intervals undercover, before touching the fix.

Run: ``python -m scripts.diagnose_calibration``

Reads the already-committed ``results/scoreboard_backtest.json`` (produced by
:mod:`scripts.backfill_scoreboard`) and breaks coverage down three ways — by ticker, by year,
and by trailing-21-day realized-volatility regime — plus the ratio of realized interval width
to what 80% coverage would actually have required. The point is to find the *mechanism* before
picking a fix; guessing the mechanism and jumping straight to a method is how "improvements"
turn out to fix nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics
from src.data import prices

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_FILE = REPO_ROOT / "results" / "scoreboard_backtest.json"

VOL_WINDOW = 21


def _load_backtest() -> pd.DataFrame:
    records = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def _trailing_realized_vol(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Trailing VOL_WINDOW-day realized volatility of daily returns, per symbol per date —
    using only returns strictly before that date, so joining this onto an as_of row never
    leaks the cycle's own future into its own volatility-regime label."""
    prices_df = prices_df.sort_values(["symbol", "date"])
    out = []
    for symbol, part in prices_df.groupby("symbol"):
        part = part.sort_values("date")
        ret = part["close"].pct_change()
        vol = ret.rolling(VOL_WINDOW).std().shift(1)  # shift(1): strictly before `date`
        out.append(pd.DataFrame({"symbol": symbol, "date": part["date"], "vol": vol}))
    return pd.concat(out, ignore_index=True)


def _required_quantile_width(y_true: np.ndarray, point: np.ndarray, level: float) -> float:
    """The symmetric half-width that WOULD have achieved `level` coverage on this exact
    sample, for comparison against what was actually used — "how much narrower were the
    realized intervals than an oracle calibrated on this same data would have been."""
    abs_err = np.abs(y_true - point)
    return float(np.quantile(abs_err, level)) * 2


def _coverage_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        cov = metrics.coverage(part["actual"], part["lo"], part["hi"])
        width = metrics.mean_interval_width(part["lo"], part["hi"])
        required = _required_quantile_width(
            part["actual"].to_numpy(), part["point"].to_numpy(), part["level"].iloc[0]
        )
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                "n": len(part),
                "coverage": cov,
                "mean_width": width,
                "oracle_width_for_level": required,
                "width_ratio": width / required if required > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = _load_backtest()
    price_df = prices.load()
    vol_df = _trailing_realized_vol(price_df)

    merged = df.merge(
        vol_df, left_on=["symbol", "as_of"], right_on=["symbol", "date"], how="left"
    )
    merged["year"] = merged["as_of"].dt.year

    print("=" * 70)
    print("Coverage by model")
    print("=" * 70)
    by_model = _coverage_table(merged, ["model"])
    print(by_model.to_string(index=False))

    print()
    print("=" * 70)
    print("Coverage by model x ticker")
    print("=" * 70)
    by_ticker = _coverage_table(merged, ["model", "symbol"])
    print(by_ticker.to_string(index=False))

    print()
    print("=" * 70)
    print("Coverage by model x year")
    print("=" * 70)
    by_year = _coverage_table(merged, ["model", "year"])
    print(by_year.to_string(index=False))

    print()
    print("=" * 70)
    print("Coverage by model x volatility regime (tercile of trailing 21d realized vol)")
    print("=" * 70)
    scored = merged.dropna(subset=["vol"]).copy()
    scored["vol_regime"] = pd.qcut(scored["vol"], 3, labels=["low", "mid", "high"])
    by_regime = _coverage_table(scored, ["model", "vol_regime"])
    print(by_regime.to_string(index=False))

    print()
    print("=" * 70)
    print("Headline")
    print("=" * 70)
    for model in sorted(df["model"].unique()):
        part = merged[merged["model"] == model]
        cov = metrics.coverage(part["actual"], part["lo"], part["hi"])
        width = metrics.mean_interval_width(part["lo"], part["hi"])
        required = _required_quantile_width(
            part["actual"].to_numpy(), part["point"].to_numpy(), part["level"].iloc[0]
        )
        print(
            f"{model:12s} coverage={cov:.3f} (nominal 0.80)  "
            f"mean_width=${width:.2f}  oracle_width=${required:.2f}  "
            f"ratio={width / required:.2f}"
        )

    out = {
        "by_model": by_model.to_dict(orient="records"),
        "by_ticker": by_ticker.to_dict(orient="records"),
        "by_year": by_year.to_dict(orient="records"),
        "by_vol_regime": by_regime.assign(
            vol_regime=lambda d: d["vol_regime"].astype(str)
        ).to_dict(orient="records"),
    }
    out_path = REPO_ROOT / "results" / "calibration_diagnosis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
