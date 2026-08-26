"""Phase E: does a simple LSTM + persistence ensemble beat either arm alone?

Run: ``python -m scripts.evaluate_ensemble``

No retraining: reuses the final backtest ledger's point forecasts and intervals directly
(``results/scoreboard_backtest.json``, produced by the Phase A ``conformal_ewma`` backfill).
Two things are combined:

* **Point forecast** — the mean of the LSTM's and persistence's point forecasts.
* **Interval** — the union: ``lo = min(lo_lstm, lo_persistence)``,
  ``hi = max(hi_lstm, hi_persistence)``. This is the conservative combination rule (its
  coverage can only be greater than or equal to the better-covered input arm, since it
  contains both arms' intervals by construction) rather than an average, which has no such
  guarantee and could easily undercover in a way that looks fine on a summary table.

This is deliberately evaluated with the same skepticism as everything else in this plan: a
union interval trivially widens coverage by construction, which is exactly the "coverage fixed
only by making intervals enormous" failure mode the plan warns about elsewhere. The interval
score (which penalizes width, not just coverage) is what actually decides whether this
"helped".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_FILE = REPO_ROOT / "results" / "scoreboard_backtest.json"
OUT = REPO_ROOT / "results" / "ensemble_evaluation.json"


def _load_paired() -> pd.DataFrame:
    records = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    lstm = df[df["model"] == "lstm"][["as_of", "symbol", "point", "lo", "hi", "level", "actual"]]
    lstm = lstm.rename(columns={"point": "lstm_point", "lo": "lstm_lo", "hi": "lstm_hi"})
    persistence = df[df["model"] == "persistence"][["as_of", "symbol", "point", "lo", "hi"]]
    persistence = persistence.rename(
        columns={"point": "persistence_point", "lo": "persistence_lo", "hi": "persistence_hi"}
    )
    return lstm.merge(persistence, on=["as_of", "symbol"], how="inner")


def _score_arm(
    name: str, point: np.ndarray, lo: np.ndarray, hi: np.ndarray, actual: np.ndarray, level: float
) -> dict:
    return {
        "name": name,
        "rmse": metrics.rmse(actual, point),
        "coverage": metrics.coverage(actual, lo, hi),
        "mean_interval_width": metrics.mean_interval_width(lo, hi),
        "interval_score": metrics.interval_score(actual, lo, hi, level),
    }


def main() -> None:
    df = _load_paired()
    actual = df["actual"].to_numpy()
    level = float(df["level"].iloc[0])

    ensemble_point = (df["lstm_point"] + df["persistence_point"]) / 2
    ensemble_lo = np.minimum(df["lstm_lo"], df["persistence_lo"])
    ensemble_hi = np.maximum(df["lstm_hi"], df["persistence_hi"])

    arms = {
        "lstm": _score_arm(
            "lstm",
            df["lstm_point"].to_numpy(), df["lstm_lo"].to_numpy(), df["lstm_hi"].to_numpy(),
            actual, level,
        ),
        "persistence": _score_arm(
            "persistence",
            df["persistence_point"].to_numpy(), df["persistence_lo"].to_numpy(),
            df["persistence_hi"].to_numpy(), actual, level,
        ),
        "ensemble": _score_arm(
            "ensemble",
            ensemble_point.to_numpy(), ensemble_lo.to_numpy(), ensemble_hi.to_numpy(),
            actual, level,
        ),
    }

    baseline_rmse = arms["persistence"]["rmse"]
    for arm in arms.values():
        arm["skill_vs_persistence"] = metrics.skill_score(arm["rmse"], baseline_rmse)

    summary = {
        "n": len(df),
        "level": level,
        "arms": arms,
        "ensemble_beats_persistence_on_interval_score": (
            arms["ensemble"]["interval_score"] < arms["persistence"]["interval_score"]
        ),
        "ensemble_beats_persistence_on_price_skill": (
            arms["ensemble"]["skill_vs_persistence"] > 0
        ),
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, arm in arms.items():
        print(
            f"{name:12s} rmse=${arm['rmse']:.2f}  skill={arm['skill_vs_persistence']:+.4f}  "
            f"coverage={arm['coverage']:.3f}  width=${arm['mean_interval_width']:.2f}  "
            f"interval_score={arm['interval_score']:.2f}"
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
