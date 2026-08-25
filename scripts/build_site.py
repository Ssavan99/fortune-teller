"""Assemble docs/data/report.json and docs/data/scoreboard.json for the two site pages.

GitHub Pages serves ``docs/`` as the site root, so the page cannot reach ``results/`` above
it. This copies the committed result files into payloads the pages can fetch, and derives the
handful of summary figures each page shows above the fold — so those numbers come from the
committed results rather than being typed into the HTML by hand.

Run: ``python -m scripts.build_site``
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import metrics
from src.data.prices import TICKERS

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "docs" / "data" / "report.json"
SCOREBOARD_OUT = REPO_ROOT / "docs" / "data" / "scoreboard.json"

BACKTEST_FILE = RESULTS / "scoreboard_backtest.json"
LIVE_FILE = RESULTS / "scoreboard_live.json"

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


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts.backfill_scoreboard and/or "
            "scripts.run_monthly before building the site."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _model_summary(records: list[dict]) -> dict:
    """Coverage, width and interval score for one (mode, model) slice — never mixed with any
    other slice. ``n_scored`` can legitimately be 0 (e.g. the live ledger's first month, before
    anything has had 21 trading days to mature) — that is reported honestly, not hidden."""
    scored = [r for r in records if r["actual"] is not None]
    summary = {"n": len(records), "n_scored": len(scored)}
    if not scored:
        return summary

    y = np.array([r["actual"] for r in scored])
    lo = np.array([r["lo"] for r in scored])
    hi = np.array([r["hi"] for r in scored])
    level = scored[0]["level"]

    summary["coverage"] = metrics.coverage(y, lo, hi)
    summary["mean_interval_width"] = metrics.mean_interval_width(lo, hi)
    summary["mean_abs_error"] = float(np.mean([r["abs_error"] for r in scored]))
    # The LLM arm's range isn't a calibrated quantile (level=None) — interval_score needs a
    # numeric alpha, so it's reported only for the arms that actually have one.
    summary["interval_score"] = metrics.interval_score(y, lo, hi, level) if level else None
    return summary


def _summarise_by_model(records: list[dict]) -> dict:
    models = sorted({r["model"] for r in records})
    return {m: _model_summary([r for r in records if r["model"] == m]) for m in models}


def _open_predictions(live_records: list[dict]) -> list[dict]:
    """Live rows with no outcome yet — sorted soonest-maturing first, since that's what a
    reader checking in on the scoreboard cares about most."""
    today = datetime.now(timezone.utc).date()
    open_rows = []
    for r in live_records:
        if r["actual"] is not None:
            continue
        target = datetime.strptime(r["target_date"], "%Y-%m-%d").date()
        open_rows.append({**r, "days_remaining": max((target - today).days, 0)})
    return sorted(open_rows, key=lambda r: (r["target_date"], r["symbol"], r["model"]))


def _series_by_ticker(backtest_records: list[dict], live_records: list[dict]) -> dict:
    """Per-ticker prediction history for the chart. Each point keeps its own ``mode`` so the
    page can render backtest and live bands distinctly rather than blending them visually into
    one line — the chart shows *all* individual predictions, it does not aggregate any."""
    series: dict[str, list[dict]] = {ticker: [] for ticker in TICKERS}
    for r in backtest_records + live_records:
        series.setdefault(r["symbol"], []).append(
            {
                "as_of": r["as_of"],
                "target_date": r["target_date"],
                "model": r["model"],
                "mode": r["mode"],
                "point": r["point"],
                "lo": r["lo"],
                "hi": r["hi"],
                "actual": r["actual"],
            }
        )
    for ticker in series:
        series[ticker].sort(key=lambda r: (r["target_date"], r["model"]))
    return series


def build_scoreboard() -> dict:
    backtest = _load_records(BACKTEST_FILE)
    live = _load_records(LIVE_FILE)

    # Defense in depth: the same rule tests/test_integrity.py enforces (Phase 7), checked again
    # here so a corrupted results file fails the build loudly rather than reaching the page.
    backtest_modes = {r["mode"] for r in backtest}
    live_modes = {r["mode"] for r in live}
    if backtest_modes - {"backtest"}:
        raise ValueError(f"{BACKTEST_FILE} contains non-backtest rows: {backtest_modes}")
    if live_modes - {"live"}:
        raise ValueError(f"{LIVE_FILE} contains non-live rows: {live_modes}")

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_days": 21,
        "level": 0.80,
        "tickers": list(TICKERS),
        "open_predictions": _open_predictions(live),
        "live_summary": _summarise_by_model(live),
        "backtest_summary": _summarise_by_model(backtest),
        "series": _series_by_ticker(backtest, live),
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

    scoreboard = build_scoreboard()
    SCOREBOARD_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCOREBOARD_OUT.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")

    print(f"wrote {SCOREBOARD_OUT}")
    print(f"  {len(scoreboard['open_predictions'])} open predictions")
    for label, summary in (
        ("live", scoreboard["live_summary"]),
        ("backtest", scoreboard["backtest_summary"]),
    ):
        for model, model_summary in summary.items():
            if model_summary["n_scored"] == 0:
                print(f"  {label}/{model}: {model_summary['n']} rows, none scored yet")
            else:
                print(
                    f"  {label}/{model}: coverage={model_summary['coverage']:.2f} "
                    f"n_scored={model_summary['n_scored']}"
                )


if __name__ == "__main__":
    main()
