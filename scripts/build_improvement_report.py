"""Phase E: assemble results/improvement.json — every arm tried in the model-improvement
plan, in one file, with the "what didn't work" list front and center.

Run: ``python -m scripts.build_improvement_report``

This does not compute anything new; it reads the already-written output of each evaluation
script (`diagnose_calibration`, `evaluate_volatility`, `evaluate_direction`,
`evaluate_features`, `evaluate_ensemble`) plus the before/after backtest coverage numbers, and
assembles them into one report — the single place the honest final outcome of the whole plan
lives, including every variant that was tried and rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "improvement.json"

# Before/after backtest coverage: hand-recorded from the two real (non-proxy) backfill runs —
# the "before" snapshot is no longer on disk (results/scoreboard_backtest.json now holds only
# the current, conformal_ewma-scored data, since backtest data is freely regenerable and not
# append-only the way live predictions are), but the exact numbers were computed from that run
# and are preserved here for the record. See model-improvement_PLAN.md Phase A / section 7 for
# the full narrative.
CALIBRATION_BEFORE_AFTER = {
    "lstm": {
        "before": {
            "method": "quantile", "coverage": 0.674, "mean_width": 51.19, "interval_score": 97.15,
        },
        "after": {
            "method": "conformal_ewma", "coverage": 0.777, "mean_width": 62.66,
            "interval_score": 92.75,
        },
    },
    "persistence": {
        "before": {
            "method": "quantile", "coverage": 0.678, "mean_width": 53.21, "interval_score": 99.00,
        },
        "after": {
            "method": "conformal_ewma", "coverage": 0.798, "mean_width": 61.08,
            "interval_score": 88.60,
        },
    },
    "nominal_level": 0.80,
    "pre_registered_success_band": [0.76, 0.84],
}


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run scripts.{path.stem} first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    diagnosis = _load("calibration_diagnosis.json")
    volatility = _load("volatility_evaluation.json")
    direction = _load("direction_evaluation.json")
    features = _load("feature_evaluation.json")
    ensemble = _load("ensemble_evaluation.json")

    what_didnt_work = [
        {
            "variant": "quantile (original method)",
            "phase": "A",
            "outcome": "undercovers by ~13 points (67-68% vs 80% nominal); kept only as the "
            "named baseline arm per the anti-overfitting protocol",
        },
        {
            "variant": "conformal (pooled, non-adaptive)",
            "phase": "A",
            "outcome": "fixes marginal coverage on the cheap proxy, but conformal_ewma was "
            "chosen instead — narrower at similar coverage, and closer to the plan's actual "
            "goal (predict the uncertainty, not just widen uniformly)",
        },
        {
            "variant": "GARCH(1,1) volatility",
            "phase": "A",
            "outcome": "not tried against the real backfill — EWMA already closed the "
            "coverage gap on the cheap proxy, meeting the pre-registered condition for "
            "skipping GARCH. Implemented and unit-tested regardless, in case a future "
            "session wants to revisit it.",
        },
        {
            "variant": "direction as a tradeable signal",
            "phase": "C",
            "outcome": "no edge; Brier score and log loss both slightly WORSE than a "
            "base-rate baseline, and the calibration curve doesn't even rank-order correctly",
        },
        {
            "variant": "technical indicator features (RSI/MACD/vol/volume-z/HL-range)",
            "phase": "D",
            "outcome": f"mean skill {features['groups']['technical']['skill_vs_baseline']:+.4f} "
            "across 4 sample as-of dates, but sign-flips twice across those dates "
            "(+0.09 to -0.06) — not distinguishable from noise at this sample size, rejected",
        },
        {
            "variant": "cross-sectional features (index-relative return)",
            "phase": "D",
            "outcome": "smallest and least consistent effect of the three feature groups "
            f"({features['groups']['cross_sectional']['skill_vs_baseline']:+.4f} mean skill, "
            "straddling zero every date), rejected",
        },
        {
            "variant": "calendar features (day-of-week, month-end)",
            "phase": "D",
            "outcome": "most consistent of the three "
            f"({features['groups']['calendar']['skill_vs_baseline']:+.4f} mean skill, 3/4 "
            "dates positive) but decaying toward zero on the two most recent dates — "
            "consistent with a good early draw regressing to the mean rather than a real "
            "effect, rejected",
        },
        {
            "variant": "LSTM + persistence ensemble (mean point, union interval)",
            "phase": "E",
            "outcome": f"coverage rises to {ensemble['arms']['ensemble']['coverage']:.3f} "
            "(union interval trivially widens coverage), but interval score "
            f"({ensemble['arms']['ensemble']['interval_score']:.2f}) is WORSE than "
            f"persistence alone ({ensemble['arms']['persistence']['interval_score']:.2f}), "
            "and price skill "
            f"({ensemble['arms']['ensemble']['skill_vs_persistence']:+.4f}) is still "
            "negative. Rejected — persistence alone remains the better standalone arm on "
            "both dimensions.",
        },
    ]

    report = {
        "calibration_before_after": CALIBRATION_BEFORE_AFTER,
        "calibration_diagnosis": diagnosis,
        "volatility_arm": {
            "skill_vs_persistence_of_volatility": volatility["skill_vs_persistence"],
            "tickers_beating_persistence": volatility["tickers_beating_persistence"],
            "n_tickers": volatility["n_tickers"],
            "units": volatility["units"],
        },
        "direction_arm": {
            "directional_accuracy": direction["directional_accuracy"],
            "brier_score": direction["brier_score"],
            "log_loss": direction["log_loss"],
            "base_rate_up": direction["base_rate_up"],
        },
        "final_production_configuration": {
            "interval_method": "conformal_ewma",
            "point_forecast_arms": ["lstm", "persistence"],
            "ensemble": "not used — did not beat persistence alone",
            "extra_lstm_features": "none — all 3 candidate groups rejected as noise (Phase D)",
        },
        "n_variants_tried": len(what_didnt_work) + 1,  # +1 for conformal_ewma, the one kept
        "what_didnt_work": what_didnt_work,
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(
        f"variants tried: {report['n_variants_tried']} "
        f"(1 kept, {len(what_didnt_work)} rejected/superseded)"
    )


if __name__ == "__main__":
    main()
