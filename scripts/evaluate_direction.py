"""Phase C: is the LSTM's directional call worth anything, scored as a probability rather than
a coin flip?

Run: ``python -m scripts.evaluate_direction``

Accuracy alone is close to meaningless at ~50% — it can't distinguish "no edge" from "a small,
genuine, well-calibrated edge" or from "overconfident nonsense that happens to land near
chance". This reframes direction as a classification problem: does the LSTM's predicted move
carry any information about **P(up over the horizon)**, scored properly with Brier score and
log loss against a base-rate baseline (always predict the historical up-rate), plus a
calibration curve (when the model says 60%, does it happen 60% of the time?).

Reuses the already-committed backtest ledger's point forecasts and actuals rather than
retraining anything: ``results/scoreboard_backtest.json`` already has, for every (as_of,
symbol) cycle, the LSTM's point forecast, the persistence point forecast (which is exactly the
anchor/previous close), and the real outcome. The 55 cycles are split chronologically in half
— the first half calibrates a simple binned probability mapping (bin by predicted return,
compute the empirical historical up-rate in that bin), the second half evaluates it. This is a
deliberately simple, honest probability estimator: the point of this arm is to test whether
directional information exists at all, not to build the best possible classifier around it.

Pre-registered expectation: no meaningful edge over the base rate. Reported honestly either
way.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_FILE = REPO_ROOT / "results" / "scoreboard_backtest.json"
OUT = REPO_ROOT / "results" / "direction_evaluation.json"

N_BINS = 5
EPS = 1e-9  # log-loss clipping, so a bin that was ever wrong in calibration never gives log(0)


def _load_paired_rows() -> pd.DataFrame:
    """One row per (as_of, symbol): the LSTM's predicted return, the actual return, both
    measured against the same anchor — persistence's point, which is exactly prev_close."""
    records = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    lstm = df[df["model"] == "lstm"][["as_of", "symbol", "point", "actual"]]
    lstm = lstm.rename(columns={"point": "lstm_point"})
    persistence = df[df["model"] == "persistence"][["as_of", "symbol", "point"]]
    persistence = persistence.rename(columns={"point": "anchor"})

    merged = lstm.merge(persistence, on=["as_of", "symbol"], how="inner")
    merged["predicted_return"] = (merged["lstm_point"] - merged["anchor"]) / merged["anchor"]
    merged["actual_return"] = (merged["actual"] - merged["anchor"]) / merged["anchor"]
    merged["actual_up"] = (merged["actual_return"] > 0).astype(int)
    merged["as_of"] = pd.to_datetime(merged["as_of"])
    return merged.sort_values("as_of")


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_binned_calibration(cal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges (quantiles of predicted_return in the calibration set) and each bin's
    empirical up-rate — a simple, transparent, non-parametric calibration map fit on
    calibration data only.
    """
    edges = np.quantile(cal["predicted_return"], np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.digitize(cal["predicted_return"], edges[1:-1])
    up_rates = np.array(
        [cal["actual_up"][bin_idx == b].mean() if (bin_idx == b).any() else np.nan
         for b in range(N_BINS)]
    )
    # A bin with no calibration data (possible with few bins/cycles) falls back to the
    # calibration set's overall base rate rather than propagating a NaN probability.
    base_rate = float(cal["actual_up"].mean())
    up_rates = np.where(np.isnan(up_rates), base_rate, up_rates)
    return edges, up_rates


def _apply_binned_calibration(edges: np.ndarray, up_rates: np.ndarray, x: np.ndarray) -> np.ndarray:
    bin_idx = np.digitize(x, edges[1:-1])
    return up_rates[bin_idx]


def main() -> None:
    df = _load_paired_rows()
    as_ofs = sorted(df["as_of"].unique())
    mid = len(as_ofs) // 2
    cal_dates, test_dates = set(as_ofs[:mid]), set(as_ofs[mid:])
    cal = df[df["as_of"].isin(cal_dates)]
    test = df[df["as_of"].isin(test_dates)].copy()

    base_rate = float(cal["actual_up"].mean())
    edges, up_rates = _fit_binned_calibration(cal)
    predicted = test["predicted_return"].to_numpy()
    test["model_p_up"] = _apply_binned_calibration(edges, up_rates, predicted)
    test["base_rate_p_up"] = base_rate

    model_brier = _brier(test["model_p_up"].to_numpy(), test["actual_up"].to_numpy())
    base_brier = _brier(test["base_rate_p_up"].to_numpy(), test["actual_up"].to_numpy())
    model_logloss = _log_loss(test["model_p_up"].to_numpy(), test["actual_up"].to_numpy())
    base_logloss = _log_loss(test["base_rate_p_up"].to_numpy(), test["actual_up"].to_numpy())

    # Reliability diagram: for each bin actually used in test, (mean predicted p, empirical
    # up-rate, n) -- read off directly from the test set's own outcomes, not re-derived from
    # the calibration fit.
    calibration_curve = []
    for b in range(N_BINS):
        mask = (test["model_p_up"] == up_rates[b]).to_numpy()
        if not mask.any():
            continue
        calibration_curve.append(
            {
                "predicted_p_up": float(up_rates[b]),
                "empirical_up_rate": float(test["actual_up"][mask].mean()),
                "n": int(mask.sum()),
            }
        )

    accuracy = float((np.sign(test["predicted_return"]) == np.sign(test["actual_return"])).mean())

    summary = {
        "n_calibration": len(cal),
        "n_test": len(test),
        "base_rate_up": base_rate,
        "directional_accuracy": accuracy,
        "brier_score": {"model": model_brier, "base_rate_baseline": base_brier},
        "log_loss": {"model": model_logloss, "base_rate_baseline": base_logloss},
        "brier_skill_vs_base_rate": 1.0 - model_brier / base_brier if base_brier > 0 else None,
        "calibration_curve": calibration_curve,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"n_calibration={summary['n_calibration']}  n_test={summary['n_test']}")
    print(f"base rate P(up)={base_rate:.3f}  directional accuracy={accuracy:.3f}")
    print(f"Brier: model={model_brier:.4f}  base-rate baseline={base_brier:.4f}")
    print(f"log loss: model={model_logloss:.4f}  base-rate baseline={base_logloss:.4f}")
    print("calibration curve (predicted -> empirical, n):")
    for row in calibration_curve:
        print(f"  {row['predicted_p_up']:.3f} -> {row['empirical_up_rate']:.3f}  (n={row['n']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
