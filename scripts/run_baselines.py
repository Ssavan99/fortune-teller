"""Score the trivial forecasters on the held-out period and write results/baselines.json.

Run: ``python -m scripts.run_baselines``

Dollar errors are not comparable across tickers — a $3 error on NVDA and a $3 error on INTC
are very different mistakes — so the pooled figures are MAPE and skill score, both scale-free,
and the per-ticker dollar figures are reported alongside rather than averaged into one number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import baselines, metrics
from src.data import prices, splits

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "baselines.json"

MODELS = ("persistence", "drift", "ar5")


def evaluate(df, split) -> dict:
    per_ticker: dict[str, dict] = {}

    for symbol in sorted(df["symbol"].unique()):
        forecasts = {
            "persistence": baselines.persistence(df, symbol, split.test),
            "drift": baselines.drift(df, symbol, split.test, split.train),
            "ar5": baselines.autoregressive(df, symbol, split.test, split.train, order=5),
        }

        base_rmse = metrics.rmse(
            forecasts["persistence"].y_true, forecasts["persistence"].y_pred
        )

        entry = {}
        for name, fc in forecasts.items():
            s = metrics.score(fc.y_true, fc.y_pred, fc.last_close)
            entry[name] = s.as_dict() | {
                "skill_vs_persistence": metrics.skill_score(s.rmse, base_rmse)
            }
        entry["mean_close"] = float(np.mean(forecasts["persistence"].y_true))
        per_ticker[symbol] = entry

    pooled = {}
    for name in MODELS:
        skills = [per_ticker[s][name]["skill_vs_persistence"] for s in per_ticker]
        mapes = [per_ticker[s][name]["mape"] for s in per_ticker]
        pooled[name] = {
            "mean_skill_vs_persistence": float(np.mean(skills)),
            "median_skill_vs_persistence": float(np.median(skills)),
            "mean_mape": float(np.mean(mapes)),
            "tickers_beating_persistence": int(sum(1 for s in skills if s > 0)),
            "n_tickers": len(skills),
        }

    return {
        "period": {
            "start": str(split.test["date"].min().date()),
            "end": str(split.test["date"].max().date()),
            "rows": int(len(split.test)),
        },
        "per_ticker": per_ticker,
        "pooled": pooled,
    }


def render(report: dict) -> str:
    p = report["period"]
    lines = [
        f"Held-out period: {p['start']} -> {p['end']}  ({p['rows']:,} ticker-days)",
        "",
        f"{'ticker':<7}{'mean $':>9}{'persist RMSE':>14}{'drift RMSE':>12}{'AR(5) RMSE':>12}"
        f"{'drift skill':>13}{'AR skill':>10}",
        "-" * 77,
    ]
    for sym, e in report["per_ticker"].items():
        lines.append(
            f"{sym:<7}{e['mean_close']:>9.2f}{e['persistence']['rmse']:>14.3f}"
            f"{e['drift']['rmse']:>12.3f}{e['ar5']['rmse']:>12.3f}"
            f"{e['drift']['skill_vs_persistence']:>13.4f}"
            f"{e['ar5']['skill_vs_persistence']:>10.4f}"
        )

    lines += ["", "Pooled:"]
    for name in MODELS:
        q = report["pooled"][name]
        lines.append(
            f"  {name:<12} mean skill {q['mean_skill_vs_persistence']:+.4f}   "
            f"mean MAPE {q['mean_mape']:.3f}%   "
            f"beats persistence on {q['tickers_beating_persistence']}/{q['n_tickers']} tickers"
        )
    return "\n".join(lines)


def main() -> None:
    df = prices.load()
    split = splits.chronological_split(df)
    report = evaluate(df, split)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(render(report))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
