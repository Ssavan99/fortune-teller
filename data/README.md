# Data

Two committed snapshots. Both are checked in so that every result in this repository
reproduces offline and in CI with no network access and no API key.

## `prices.csv`

Daily split- and dividend-adjusted OHLCV bars.

| | |
|---|---|
| Source | Yahoo Finance, via [`yfinance`](https://github.com/ranaroussi/yfinance) — free, no key |
| Rows | 21,450 |
| Symbols | 15 (AAPL, ADBE, AMD, AMZN, BABA, DELL, GOOG, IBM, INTC, META, MSFT, NVDA, ORCL, SAP, TSLA) |
| Coverage | 2020-12-10 → 2026-08-21, 1,430 trading days per symbol, no gaps |
| Columns | `date, symbol, open, high, low, close, volume` |
| Regenerate | `python -m src.data.prices --refresh` |

**Adjusted, deliberately.** Fetched with `auto_adjust=True`. NVDA split 4:1 in July 2021 and
10:1 in June 2024, both inside this window; on unadjusted bars those show up as overnight
drops of 75% and 90%. A next-day model scored on unadjusted prices is scored partly on
corporate actions.

Re-fetching will not always reproduce this file byte for byte — vendors revise history. The
snapshot is the reference; `--refresh` is for extending the window.

## `sentiment_daily.csv`

Daily aggregate news-sentiment scores, one column per ticker, wide format.

| | |
|---|---|
| Source | FinBrain daily sentiment aggregates, retrieved via RapidAPI in 2024 |
| Rows | 838 |
| Symbols | 14 — the price universe **minus DELL**, which the source had no coverage for |
| Coverage | 2020-12-21 → 2024-03-25 |
| Range | roughly −0.55 to +0.67; per-ticker means are all positive (+0.07 to +0.35) |
| Columns | `date` plus one column per ticker |
| Regenerate | **Not regenerable.** The source API requires a paid key. This is a fixed snapshot. |

### Consequences for the study

- Sentiment runs to **2024-03-25**, which reaches **17 dates past the start of the main
  held-out period**. Experiment B is therefore clamped to end at 2024-02-29 — the same cutoff
  as the main validation split — so that no architecture is ever selected on a window
  overlapping the held-out period. Seventeen days of ablation data is a cheap price for
  keeping the headline result clean. The clamp is asserted at import time in
  `src/data/splits.py`.
- Experiment B's no-sentiment arm is restricted to exactly the same dates as its sentiment
  arm, so the difference between them is attributable to the feature and not to the window.
- **Missingness is whole-row, not cell-level.** There are zero empty cells in the file. What
  is missing is entire dates: of the 819 trading days inside the coverage window, **14 have no
  sentiment row at all**. In the other direction, **33 of the 838 rows fall on non-trading
  days** (2 of them weekends) and drop out on a merge against prices. So the missing-indicator
  feature has to be constructed by reindexing onto the trading calendar — there is nothing at
  cell level for it to detect. It is never filled with `0`: an absence of news is not neutral
  news, and conflating the two hands the model a fake signal.
- The per-ticker means are all positive, from +0.068 (TSLA) to +0.345 (SAP). Whatever this
  series measures, it is not centred, and a level shift is not evidence of sentiment.
