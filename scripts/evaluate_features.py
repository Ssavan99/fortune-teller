"""Phase D: evaluate candidate LSTM input feature groups, one at a time, on validation only.

Run: ``python -m scripts.evaluate_features``

For a small sample of as-of dates (this is a validation-only comparison, not the full 55-cycle
backfill — it doesn't need historical breadth, it needs a few honest before/after pairs), trains
the LSTM with each candidate feature group added to its input via
:func:`sequences.build`'s existing ``extra_features`` hook (the same mechanism the original
study's sentiment ablation used), and compares validation RMSE (in the ``return`` target's own
units) against a baseline with no extra features. A feature group is kept only if it improves
mean validation RMSE across the sample; the pre-registered expectation is that most do nothing,
and that is reported as plainly as a group that helps.

Every group is compared against the SAME baseline runs (same seed, same as-of dates, same
train/val split) so the only thing that differs between a baseline run and a feature-group run
is the extra input columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import features, metrics, sequences
from src.data import prices
from src.rolling import LOOKBACK, _chronological_split
from src.train import TrainConfig, predict, train

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "feature_evaluation.json"

HORIZON = 21
SEED = 20260822
SAMPLE_AS_OFS = [
    pd.Timestamp("2022-09-01"),
    pd.Timestamp("2023-09-01"),
    pd.Timestamp("2024-09-01"),
    pd.Timestamp("2025-09-01"),
]
GROUPS = ("technical", "cross_sectional", "calendar")


def _technical_extra(history: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for symbol, part in history.groupby("symbol"):
        part = part.sort_values("date")
        tech = features.technical_features(part)
        cols = [features.with_missing_indicator(tech[c]) for c in features.TECHNICAL_COLUMNS]
        out[symbol] = np.hstack(cols)
    return out


def _cross_sectional_extra(history: pd.DataFrame) -> dict[str, np.ndarray]:
    cs = features.cross_sectional_features(history)
    out = {}
    for symbol, part in history.groupby("symbol"):
        part = part.sort_values("date")
        aligned = cs[symbol].reindex(part["date"])
        out[symbol] = features.with_missing_indicator(aligned)
    return out


def _calendar_extra(history: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for symbol, part in history.groupby("symbol"):
        part = part.sort_values("date")
        cal = features.calendar_features(part["date"])
        out[symbol] = cal.to_numpy(dtype=float)
    return out


EXTRA_BUILDERS = {
    "technical": _technical_extra,
    "cross_sectional": _cross_sectional_extra,
    "calendar": _calendar_extra,
}


def _val_rmse(
    history: pd.DataFrame, as_of: pd.Timestamp, extra_features: dict | None, config_overrides: dict
) -> float:
    split = _chronological_split(history)
    config = TrainConfig(target="return", lookback=LOOKBACK, seed=SEED, **config_overrides)
    built = sequences.build(
        history, split, lookback=config.lookback, target="return", horizon=HORIZON,
        extra_features=extra_features,
    )
    model, norm, config = train(built["train"], built["val"], config, verbose=False)
    val_pred = sequences.to_dollars(predict(model, built["val"], norm), built["val"], "return")
    return metrics.rmse(built["val"].y_close, val_pred)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="full",
        choices=("full", "fast"),
        help="'fast' is a cheap wiring smoke test (tiny network); 'full' is the real "
        "evaluation used to decide which groups to keep",
    )
    args = parser.parse_args()
    config_overrides = (
        {"hidden": 8, "layers": 1, "max_epochs": 2, "patience": 2} if args.config == "fast" else {}
    )

    df = prices.load()
    results: dict[str, dict] = {}

    baseline_rmses = []
    group_rmses = {g: [] for g in GROUPS}

    for as_of in SAMPLE_AS_OFS:
        history = df[df["date"] < as_of].copy()
        if history.empty:
            continue
        split = _chronological_split(history)
        if split.train.empty or split.val.empty:
            continue

        baseline = _val_rmse(history, as_of, None, config_overrides)
        baseline_rmses.append(baseline)
        print(f"as_of={as_of.date()}  baseline val RMSE=${baseline:.3f}")

        for group in GROUPS:
            extra = EXTRA_BUILDERS[group](history)
            rmse = _val_rmse(history, as_of, extra, config_overrides)
            group_rmses[group].append(rmse)
            skill = metrics.skill_score(rmse, baseline)
            print(f"  + {group:16s} val RMSE=${rmse:.3f}  skill vs baseline={skill:+.4f}")

    mean_baseline = float(np.mean(baseline_rmses))
    for group in GROUPS:
        mean_rmse = float(np.mean(group_rmses[group]))
        results[group] = {
            "n_as_of_dates": len(group_rmses[group]),
            "mean_val_rmse": mean_rmse,
            "mean_baseline_val_rmse": mean_baseline,
            "skill_vs_baseline": metrics.skill_score(mean_rmse, mean_baseline),
            "helped": mean_rmse < mean_baseline,
        }

    summary = {
        "config": args.config,
        "as_of_dates": [str(d.date()) for d in SAMPLE_AS_OFS],
        "n_groups_tried": len(GROUPS),
        "mean_baseline_val_rmse": mean_baseline,
        "groups": results,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nmean baseline val RMSE: ${mean_baseline:.3f}")
    for group, r in results.items():
        verdict = "HELPED" if r["helped"] else "did not help"
        print(f"  {group:16s} skill={r['skill_vs_baseline']:+.4f}  ({verdict})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
