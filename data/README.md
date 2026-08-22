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

- Sentiment ends **2024-03-25**, so it cannot be used on the main held-out period. Experiment B
  is therefore run on its own window, with the no-sentiment arm restricted to exactly the same
  dates so the comparison is attributable.
- Coverage inside the window is by calendar date, not trading date. Dates with no score are
  handled with an **explicit missing-indicator feature**, never filled with `0` — an absence of
  news is not the same as neutral news, and conflating them hands the model a fake signal.
- The per-ticker means are all positive. Whatever this series measures, it is not centred, and
  a level shift is not evidence of sentiment.
