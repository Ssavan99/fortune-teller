"""Generate the README figure from committed results.

Run: ``python -m scripts.make_figures``

Everything drawn here is read from ``results/``. The figure is regenerable, so it cannot
drift away from the numbers it is illustrating the way a pasted screenshot can.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "docs" / "skill_by_ticker.png"

INK = "#14171f"
MUTED = "#5b6270"
GOOD = "#1e7f4f"
BAD = "#c0392b"
LINE = "#d9dde4"


def main() -> None:
    a = json.loads((RESULTS / "experiment_a.json").read_text(encoding="utf-8"))
    ret = a["arms"]["return"]
    lvl = a["arms"]["level"]
    period = a["period"]

    tickers = list(ret["per_ticker"])
    skills = [ret["per_ticker"][t]["skill_vs_persistence"] for t in tickers]
    order = sorted(range(len(tickers)), key=lambda i: skills[i])
    tickers = [tickers[i] for i in order]
    skills = [skills[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), facecolor="white")

    # ---- left: per-ticker skill, return arm ----
    colours = [GOOD if s > 0 else BAD for s in skills]
    ax1.barh(tickers, skills, color=colours, alpha=0.85, height=0.68)
    ax1.axvline(0, color=INK, linewidth=1.1)
    ax1.set_xlabel("skill score vs persistence   (>0 beats the baseline)", color=MUTED)
    ax1.set_title(
        f"LSTM (return target) vs persistence\nmean skill "
        f"{ret['pooled']['mean_skill_vs_persistence']:+.4f}   ·   "
        f"beats baseline on {ret['pooled']['tickers_beating_persistence']}"
        f"/{ret['pooled']['n_tickers']}",
        color=INK, fontsize=12, pad=12,
    )
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.spines["bottom"].set_color(LINE)
    ax1.tick_params(colors=MUTED)
    ax1.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.7)
    ax1.set_axisbelow(True)

    # ---- right: level arm cannot leave its training range ----
    diag = lvl["diagnostics"]
    names = sorted(diag, key=lambda t: diag[t]["test_close_max"] / diag[t]["train_close_max"])
    train_max = [diag[t]["train_close_max"] for t in names]
    test_max = [diag[t]["test_close_max"] for t in names]
    pred_max = [diag[t]["pred_max"] for t in names]

    reach = [p / tr for p, tr in zip(pred_max, train_max, strict=True)]
    needed = [te / tr for te, tr in zip(test_max, train_max, strict=True)]

    y = range(len(names))
    ax2.barh(list(y), needed, color=LINE, height=0.68, label="where prices actually went")
    ax2.barh(list(y), reach, color="#1f4fd8", alpha=0.85, height=0.4,
             label="furthest the model predicted")
    ax2.axvline(1.0, color=INK, linewidth=1.1, linestyle="--")
    ax2.text(1.02, len(names) - 0.4, "training max", color=MUTED, fontsize=9)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(names)
    ax2.set_xlabel("multiple of the training-set maximum close", color=MUTED)
    ax2.set_title(
        "Why the level target fails\nit cannot express a price above what it trained on",
        color=INK, fontsize=12, pad=12,
    )
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.spines["bottom"].set_color(LINE)
    ax2.tick_params(colors=MUTED)
    ax2.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=MUTED)

    fig.suptitle(
        f"Held-out {period['start']} to {period['end']}  ·  "
        f"{period['rows']:,} scored ticker-days  ·  15 tickers",
        color=MUTED, fontsize=10.5, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
