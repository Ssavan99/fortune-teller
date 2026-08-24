"""Experiment A — does an LSTM beat persistence on the held-out period?

Run: ``python -m scripts.run_experiment_a``

Two target parameterisations are trained and reported:

* ``level``  — predict the next close directly, in the symbol's training-derived scale.
* ``return`` — predict the next close-to-close change, then convert back to a price.

The comparison against persistence is made on **exactly the same rows** as the model's
predictions: the persistence forecast is each sequence's own ``prev_close``, so the two can
never be scored on different alignments.
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
OUT = REPO_ROOT / "results" / "experiment_a.json"

TARGETS = ("level", "return")


def diagnose(seqs, y_pred_dollars, split, df) -> dict:
    """Per-ticker evidence for *why* an arm behaves as it does.

    Two questions a reader will ask, answered with numbers rather than assertion:

    * **Is the model stuck inside its training range?** A min-max scaled level target cannot
      express a price above the training maximum without extrapolating past 1.0, which an
      LSTM with a bounded-ish head does badly. ``pred_max`` against ``train_close_max`` and
      ``test_close_max`` shows whether that is what happened.
    * **Is the model predicting anything at all?** ``move_std`` is the spread of the predicted
      day-over-day move. If it is near zero the model has collapsed onto persistence;
      ``move_corr`` says whether the moves it does predict line up with reality.
    """
    train_dates = set(split.train["date"])
    out = {}
    for symbol in sorted(set(seqs.symbols)):
        mask = seqs.symbols == symbol
        part = df[df["symbol"] == symbol]
        train_close = part[part["date"].isin(train_dates)]["close"]

        pred_move = y_pred_dollars[mask] - seqs.prev_close[mask]
        actual_move = seqs.y_close[mask] - seqs.prev_close[mask]
        # corrcoef returns nan if either series is constant, so guard both.
        corr = (
            float(np.corrcoef(pred_move, actual_move)[0, 1])
            if np.std(pred_move) > 1e-12 and np.std(actual_move) > 1e-12
            else None
        )

        out[symbol] = {
            "train_close_max": float(train_close.max()),
            "test_close_max": float(seqs.y_close[mask].max()),
            "pred_max": float(y_pred_dollars[mask].max()),
            "pred_move_std": float(np.std(pred_move)),
            "actual_move_std": float(np.std(actual_move)),
            "move_corr": corr,
        }
    return out


def score_arm(seqs, y_pred_dollars) -> dict:
    """Per-ticker and pooled scores for one trained model, against persistence."""
    per_ticker = {}
    for symbol in sorted(set(seqs.symbols)):
        mask = seqs.symbols == symbol
        model = metrics.score(seqs.y_close[mask], y_pred_dollars[mask], seqs.prev_close[mask])
        base = metrics.score(seqs.y_close[mask], seqs.prev_close[mask], seqs.prev_close[mask])
        per_ticker[symbol] = {
            "lstm": model.as_dict(),
            "persistence": base.as_dict(),
            "skill_vs_persistence": metrics.skill_score(model.rmse, base.rmse),
        }

    skills = [v["skill_vs_persistence"] for v in per_ticker.values()]
    pooled = {
        "mean_skill_vs_persistence": float(np.mean(skills)),
        "median_skill_vs_persistence": float(np.median(skills)),
        "worst_skill": float(np.min(skills)),
        "best_skill": float(np.max(skills)),
        "tickers_beating_persistence": int(sum(1 for s in skills if s > 0)),
        "n_tickers": len(skills),
        "mean_lstm_mape": float(np.mean([v["lstm"]["mape"] for v in per_ticker.values()])),
        "mean_persistence_mape": float(
            np.mean([v["persistence"]["mape"] for v in per_ticker.values()])
        ),
    }
    return {"per_ticker": per_ticker, "pooled": pooled}


def run_target(df, split, target: str) -> dict:
    print(f"\n=== target: {target} ===")
    config = TrainConfig(target=target)
    built = sequences.build(df, split, lookback=config.lookback, target=target)

    started = time.time()
    model, norm, config = train(built["train"], built["val"], config)
    elapsed = time.time() - started

    inverter = sequences.ScaledInverter(df, split) if target == "level" else None
    raw = predict(model, built["test"], norm)
    y_pred = sequences.to_dollars(raw, built["test"], target, inverter)

    arm = score_arm(built["test"], y_pred)
    arm["diagnostics"] = diagnose(built["test"], y_pred, split, df)
    arm["config"] = config.as_dict()
    arm["train_seconds"] = round(elapsed, 1)
    arm["epochs_run"] = len(config.history)
    arm["best_val_loss"] = min(h["val_loss"] for h in config.history)

    q = arm["pooled"]
    print(
        f"  mean skill vs persistence: {q['mean_skill_vs_persistence']:+.4f}   "
        f"beats persistence on {q['tickers_beating_persistence']}/{q['n_tickers']} tickers"
    )
    print(
        f"  LSTM mean MAPE {q['mean_lstm_mape']:.3f}%  vs  "
        f"persistence {q['mean_persistence_mape']:.3f}%"
    )
    return arm


def render(report: dict) -> str:
    lines = [
        "",
        f"Held-out period: {report['period']['start']} -> {report['period']['end']}"
        f"  ({report['period']['rows']:,} scored ticker-days)",
        "",
        f"{'ticker':<8}{'persist RMSE':>14}{'LSTM level':>13}{'skill':>9}"
        f"{'LSTM return':>14}{'skill':>9}",
        "-" * 67,
    ]
    lvl = report["arms"]["level"]["per_ticker"]
    ret = report["arms"]["return"]["per_ticker"]
    for sym in lvl:
        lines.append(
            f"{sym:<8}{lvl[sym]['persistence']['rmse']:>14.3f}"
            f"{lvl[sym]['lstm']['rmse']:>13.3f}{lvl[sym]['skill_vs_persistence']:>9.4f}"
            f"{ret[sym]['lstm']['rmse']:>14.3f}{ret[sym]['skill_vs_persistence']:>9.4f}"
        )
    lines.append("")
    for target in TARGETS:
        q = report["arms"][target]["pooled"]
        lines.append(
            f"  {target:<7} mean skill {q['mean_skill_vs_persistence']:+.4f}  "
            f"(worst {q['worst_skill']:+.4f}, best {q['best_skill']:+.4f})  "
            f"beats persistence on {q['tickers_beating_persistence']}/{q['n_tickers']}"
        )

    lines += ["", "Why the level arm fails - training range vs held-out range:", ""]
    lines.append(
        f"{'ticker':<8}{'train max':>11}{'test max':>10}{'pred max':>10}"
        f"{'pred/train':>12}{'move corr':>11}"
    )
    lines.append("-" * 62)
    dl = report["arms"]["level"]["diagnostics"]
    for sym, d in dl.items():
        corr = f"{d['move_corr']:+.3f}" if d["move_corr"] is not None else "  n/a"
        lines.append(
            f"{sym:<8}{d['train_close_max']:>11.2f}{d['test_close_max']:>10.2f}"
            f"{d['pred_max']:>10.2f}{d['pred_max'] / d['train_close_max']:>12.2f}{corr:>11}"
        )

    lines += ["", "Return arm - is it predicting moves, or collapsing onto persistence?", ""]
    lines.append(f"{'ticker':<8}{'pred move sd':>14}{'actual move sd':>16}{'move corr':>11}")
    lines.append("-" * 49)
    dr = report["arms"]["return"]["diagnostics"]
    for sym, d in dr.items():
        corr = f"{d['move_corr']:+.3f}" if d["move_corr"] is not None else "  n/a"
        lines.append(
            f"{sym:<8}{d['pred_move_std']:>14.3f}{d['actual_move_std']:>16.3f}{corr:>11}"
        )
    return "\n".join(lines)


def main() -> None:
    df = prices.load()
    split = splits.chronological_split(df)

    report = {
        "period": {
            "start": str(split.test["date"].min().date()),
            "end": str(split.test["date"].max().date()),
            "rows": int(len(split.test)),
        },
        "arms": {target: run_target(df, split, target) for target in TARGETS},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
