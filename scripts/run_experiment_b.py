"""Experiment B — does adding news sentiment change the answer?

Run: ``python -m scripts.run_experiment_b``

Both arms are trained on **identical dates, identical splits and identical seeds**. The only
difference is two extra input columns: the sentiment score and its missing-indicator. Anything
else moving between the arms would make the delta unattributable.

The window ends 2024-02-29, not at the sentiment snapshot's own end of 2024-03-25, so that no
architecture is ever selected on data overlapping Experiment A's held-out period.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src import metrics, sequences
from src.data import prices, sentiment, splits
from src.train import TrainConfig, predict, train

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "experiment_b.json"

TARGET = "return"  # the level arm is already known to fail structurally; see experiment A
SEED = 20260822


def run_arm(window, split, use_sentiment: bool) -> dict:
    """Train one arm.

    ``window`` must be the frame already truncated at ``SENTIMENT_END``. Passing the full
    price frame here would put everything after the sentiment window — the whole of
    Experiment A's held-out period — into this experiment's test bucket, since ``build``
    assigns to ``test`` anything that is not in train or val.
    """
    label = "price+sentiment" if use_sentiment else "price only"
    print(f"\n=== arm: {label} ===")

    if window["date"].max() > splits.SENTIMENT_END:
        raise ValueError("window extends past SENTIMENT_END; test bucket would be wrong")

    extra = sentiment.align_to_prices(window) if use_sentiment else None
    config = TrainConfig(target=TARGET, seed=SEED)
    built = sequences.build(
        window, split, lookback=config.lookback, target=TARGET, extra_features=extra
    )

    started = time.time()
    model, norm, config = train(built["train"], built["val"], config)
    raw = predict(model, built["test"], norm)
    y_pred = sequences.to_dollars(raw, built["test"], TARGET)

    test = built["test"]
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
        "n_features": int(built["train"].x.shape[2]),
        "per_ticker": per_ticker,
        "pooled": {
            "mean_skill_vs_persistence": float(np.mean(skills)),
            "median_skill_vs_persistence": float(np.median(skills)),
            "tickers_beating_persistence": int(sum(1 for s in skills if s > 0)),
            "n_tickers": len(skills),
        },
        "config": config.as_dict(),
        "epochs_run": len(config.history),
        "best_val_loss": min(h["val_loss"] for h in config.history),
        "train_seconds": round(time.time() - started, 1),
    }
    q = arm["pooled"]
    print(
        f"  {arm['n_features']} features | mean skill "
        f"{q['mean_skill_vs_persistence']:+.4f} | beats persistence on "
        f"{q['tickers_beating_persistence']}/{q['n_tickers']}"
    )
    return arm


def render(report: dict) -> str:
    a, b = report["arms"]["price_only"], report["arms"]["price_sentiment"]
    lines = [
        "",
        f"Window: {report['period']['start']} -> {report['period']['end']}"
        f"  ({report['period']['rows']:,} ticker-days)",
        "",
        f"{'ticker':<8}{'coverage':>10}{'price only':>12}{'+sentiment':>12}{'delta':>10}",
        "-" * 52,
    ]
    for sym in a["per_ticker"]:
        s0 = a["per_ticker"][sym]["skill_vs_persistence"]
        s1 = b["per_ticker"][sym]["skill_vs_persistence"]
        cov = report["coverage"][sym]["coverage"]
        lines.append(f"{sym:<8}{cov:>10.3f}{s0:>12.4f}{s1:>12.4f}{s1 - s0:>+10.4f}")

    deltas = [
        b["per_ticker"][s]["skill_vs_persistence"] - a["per_ticker"][s]["skill_vs_persistence"]
        for s in a["per_ticker"]
    ]
    lines += [
        "",
        f"  price only      mean skill {a['pooled']['mean_skill_vs_persistence']:+.4f}",
        f"  price+sentiment mean skill {b['pooled']['mean_skill_vs_persistence']:+.4f}",
        "",
        f"  delta: mean {np.mean(deltas):+.4f}, sd {np.std(deltas):.4f}, "
        f"range [{min(deltas):+.4f}, {max(deltas):+.4f}]",
        f"  sentiment helped on {sum(1 for d in deltas if d > 0)}/{len(deltas)} tickers",
    ]
    return "\n".join(lines)


def main() -> None:
    df = prices.load()
    split = splits.sentiment_window_split(df)
    window = df[df["date"] <= splits.SENTIMENT_END]

    report = {
        "period": {
            "start": str(split.test["date"].min().date()),
            "end": str(split.test["date"].max().date()),
            "rows": int(len(split.test)),
        },
        "coverage": sentiment.coverage_report(window),
        "arms": {
            "price_only": run_arm(window, split, use_sentiment=False),
            "price_sentiment": run_arm(window, split, use_sentiment=True),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
