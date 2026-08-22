"""Assemble docs/data/report.json for the results page.

GitHub Pages serves ``docs/`` as the site root, so the page cannot reach ``results/`` above
it. This copies the three experiment reports into one payload the page can fetch, and derives
the handful of summary figures the page shows above the fold — so those numbers come from the
committed results rather than being typed into the HTML by hand.

Run: ``python -m scripts.build_site``
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "docs" / "data" / "report.json"

SOURCES = {
    "baselines": "baselines.json",
    "experiment_a": "experiment_a.json",
    "experiment_b": "experiment_b.json",
    "experiment_c": "experiment_c.json",
}


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run the experiment scripts before building the site."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(data: dict) -> dict:
    a = data["experiment_a"]
    b = data["experiment_b"]
    ret = a["arms"]["return"]["pooled"]
    lvl = a["arms"]["level"]["pooled"]

    deltas = [
        b["arms"]["price_sentiment"]["per_ticker"][s]["skill_vs_persistence"]
        - b["arms"]["price_only"]["per_ticker"][s]["skill_vs_persistence"]
        for s in b["arms"]["price_only"]["per_ticker"]
    ]
    mean_delta = sum(deltas) / len(deltas)
    sd_delta = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5

    return {
        "period": a["period"],
        "headline_skill": ret["mean_skill_vs_persistence"],
        "headline_beats": ret["tickers_beating_persistence"],
        "headline_n": ret["n_tickers"],
        "level_skill": lvl["mean_skill_vs_persistence"],
        "level_beats": lvl["tickers_beating_persistence"],
        "lstm_mape": ret["mean_lstm_mape"],
        "persistence_mape": ret["mean_persistence_mape"],
        "sentiment_delta_mean": mean_delta,
        "sentiment_delta_sd": sd_delta,
        "sentiment_helped": sum(1 for d in deltas if d > 0),
    }


def main() -> None:
    data = {key: load(name) for key, name in SOURCES.items()}
    data["summary"] = summarise(data)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

    s = data["summary"]
    print(f"wrote {OUT}")
    print(f"  headline skill {s['headline_skill']:+.4f} on {s['period']['start']}"
          f" -> {s['period']['end']}")
    print(f"  beats persistence on {s['headline_beats']}/{s['headline_n']} tickers")


if __name__ == "__main__":
    main()
