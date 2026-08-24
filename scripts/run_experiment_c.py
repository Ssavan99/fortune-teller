"""Experiment C — how much apparent skill does a pre-split scaler manufacture?

Run: ``python -m scripts.run_experiment_c``

Everything is held fixed except one line: whether the feature scaler is fitted on the training
rows or on the whole series. The leaky arm never sees a test *row* during training — it only
inherits the test period's minimum and maximum through the normalisation constants. That is
the whole defect, and it is subtle enough to survive code review.

Two numbers are reported side by side for each arm:

* the **scaled-unit** error, which is what gets reported when metrics are never inverted, and
* the **dollar** error and skill score, which is what a reader can actually check.

The gap between how those two move is the point of the experiment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src import metrics, sequences
from src.data import prices, splits
from src.train import TrainConfig, predict, train

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "experiment_c.json"

TARGETS = ("level", "return")


def run_arm(df, split, target: str, leaky: bool) -> dict:
    label = "leaky (scaler fitted on all data)" if leaky else "clean (scaler fitted on train)"
    print(f"\n=== {target} / {label} ===")

    config = TrainConfig(target=target)
    built = sequences.build(
        df, split, lookback=config.lookback, target=target, leaky_scaling=leaky
    )

    started = time.time()
    model, norm, config = train(built["train"], built["val"], config, verbose=False)
    raw = predict(model, built["test"], norm)

    inverter = (
        sequences.ScaledInverter(df, split, leaky_scaling=leaky) if target == "level" else None
    )
    y_pred = sequences.to_dollars(raw, built["test"], target, inverter)

    test = built["test"]

    # The scaled-unit number: error in whatever space the model was trained in. This is the
    # figure that looks reassuring and means nothing.
    scaled_rmse = float(np.sqrt(np.mean((raw - test.y) ** 2)))

    per_ticker = {}
    for symbol in sorted(set(test.symbols)):
        mask = test.symbols == symbol
        m = metrics.score(test.y_close[mask], y_pred[mask], test.prev_close[mask])
        b = metrics.score(test.y_close[mask], test.prev_close[mask], test.prev_close[mask])
        per_ticker[symbol] = {
            "rmse": m.rmse,
            "persistence_rmse": b.rmse,
            "skill_vs_persistence": metrics.skill_score(m.rmse, b.rmse),
        }

    skills = [v["skill_vs_persistence"] for v in per_ticker.values()]
    arm = {
        "label": label,
        "leaky": leaky,
        "scaled_rmse": scaled_rmse,
        "dollar_rmse_mean": float(np.mean([v["rmse"] for v in per_ticker.values()])),
        "mean_skill_vs_persistence": float(np.mean(skills)),
        "tickers_beating_persistence": int(sum(1 for s in skills if s > 0)),
        "n_tickers": len(skills),
        "best_val_loss": min(h["val_loss"] for h in config.history),
        "epochs_run": len(config.history),
        "train_seconds": round(time.time() - started, 1),
        "per_ticker": per_ticker,
    }
    print(
        f"  scaled RMSE {scaled_rmse:.5f} | mean dollar RMSE "
        f"{arm['dollar_rmse_mean']:.3f} | mean skill "
        f"{arm['mean_skill_vs_persistence']:+.4f}"
    )
    return arm


def render(report: dict) -> str:
    lines = [
        "",
        f"Held-out period: {report['period']['start']} -> {report['period']['end']}",
        "",
        f"{'target':<9}{'arm':<8}{'scaled RMSE':>13}{'mean $ RMSE':>13}{'mean skill':>12}",
        "-" * 55,
    ]
    for target in TARGETS:
        for key in ("clean", "leaky"):
            a = report["arms"][target][key]
            lines.append(
                f"{target:<9}{key:<8}{a['scaled_rmse']:>13.5f}"
                f"{a['dollar_rmse_mean']:>13.3f}{a['mean_skill_vs_persistence']:>12.4f}"
            )
        lines.append("")

    lines.append("Effect of the leak:")
    for target in TARGETS:
        clean = report["arms"][target]["clean"]
        leaky = report["arms"][target]["leaky"]
        scaled_change = (
            (leaky["scaled_rmse"] - clean["scaled_rmse"]) / clean["scaled_rmse"] * 100
        )
        lines.append(
            f"  {target:<7} scaled RMSE {scaled_change:+.1f}%   "
            f"mean skill {clean['mean_skill_vs_persistence']:+.4f} -> "
            f"{leaky['mean_skill_vs_persistence']:+.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    df = prices.load()
    split = splits.chronological_split(df)

    report = {
        "period": {
            "start": str(split.test["date"].min().date()),
            "end": str(split.test["date"].max().date()),
        },
        "arms": {
            target: {
                "clean": run_arm(df, split, target, leaky=False),
                "leaky": run_arm(df, split, target, leaky=True),
            }
            for target in TARGETS
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
