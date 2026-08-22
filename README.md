# stock-lstm-vs-persistence

**Does an LSTM beat `tomorrow's close = today's close`?** On 15 large-cap tech tickers, over
two and a half years of held-out data, measured in dollars: **no.**

📊 **[Results page →](https://ssavan99.github.io/stock-lstm-vs-persistence/)**

Next-day close prediction is a standard first deep-learning-on-finance project, and it is
usually reported without a baseline and in normalised units — a mean absolute error of
`0.0258` on MinMax-scaled closes looks small and means nothing. This repository does the
comparison the task actually calls for: the same model, scored in dollars, against the
trivial forecast that tomorrow's price is today's.

![Per-ticker skill against persistence, and why the level target fails](docs/skill_by_ticker.png)

## The result

Held out: **2024-03-01 → 2026-08-21**, 9,315 scored ticker-days, never touched during
fitting, scaling, early stopping or model selection.

| Model | Mean skill vs persistence | Beats persistence | Mean MAPE |
|---|---|---|---|
| Persistence (baseline) | — | — | 1.909% |
| Random walk with drift | −0.0002 | 11 / 15 * | 1.909% |
| AR(5) on returns | −0.0068 | 0 / 15 | 1.923% |
| **Bi-LSTM, return target** | **−0.0067** | **6 / 15** | 1.929% |
| Bi-LSTM, level target | −8.0148 | 0 / 15 | 16.966% |

Skill score is `1 − RMSE_model / RMSE_persistence`. Positive beats the baseline; negative
means the trivial forecast was better.

\* Drift "wins" on 11 of 15 tickers with a mean skill of −0.0002. A win count is not a
result — the wins are noise, and it loses on the tickers that matter more.

## Why each model fails

Both failures have an identifiable mechanism, reported rather than glossed:

**The level model cannot leave its training range.** A min-max scaled level target can only
express prices up to the training maximum, and this universe roughly doubled after the
cutoff. NVDA is the clearest case:

| | Training max | Held-out max | Largest prediction |
|---|---|---|---|
| NVDA | $43.72 | $235.47 | $68.26 |

The price went 5.4× beyond the training range; the model reached 1.6×. That is not
undertraining, and no amount of tuning fixes a target parameterisation that cannot represent
the answer.

**The return model mostly predicts "no change".** Its predicted daily moves have a standard
deviation 5–30× smaller than real ones — for NVDA, $0.38 predicted against $4.21 actual. It
has largely collapsed onto persistence, then added noise.

It is not pure noise, though. The correlation between predicted and actual direction is
positive on **13 of 15 tickers**, mean **+0.046**. There is a faint signal there. It is not
enough to beat doing nothing, and reporting it as a win would require ignoring the RMSE.

## Does news sentiment help?

No detectable effect. Adding a daily sentiment score plus a missing-indicator changes mean
skill by **−0.0049**, with a standard deviation across tickers of **0.0089** — the effect is
smaller than its own spread, so this is "no signal", not "sentiment hurts". It helped on 3 of
15 tickers. Neither arm beats persistence.

Both arms share dates, splits and seed; only the two extra input columns differ. DELL acts as
a control — the sentiment source never covered it.

## How the evaluation is kept honest

The point of this repository is the evaluation, so the safeguards are the substance, not
boilerplate:

- **Scaler fitted on training rows only.** Fitting before the split is the classic defect
  here: the model never sees a test *row*, but the test period's minimum and maximum set the
  normalisation for everything. `TrainOnlyScaler` refuses to be refitted and refuses to
  transform before fitting. A test builds a series that is flat in training and spikes only
  after the cutoff, then asserts the scaler never learns the spike — and separately asserts
  that the broken version *would* be caught.
- **Fixed calendar cutoffs, not ratios.** Ratio splits move every boundary when data is
  extended, making two runs incomparable.
- **Windows end strictly before their target.** Verified row by row across all 9,315 test
  sequences.
- **The baseline is scored on identical rows** — it is each sequence's own previous close, so
  the two can never be misaligned.
- **Adjusted prices.** NVDA split 4:1 in 2021 and 10:1 in 2024, both inside the window; on
  unadjusted bars those are −75% and −90% overnight "moves".
- **Everything in dollars.** No scaled-unit metric is reported anywhere.

## What the pre-split scaler is worth

Experiment C runs the identical code path with one flag changed: whether the scaler is fitted
on the training rows or on the whole series. No test row enters training in either case — only
the normalisation constants differ.

| Target | Scaler | Scaled-unit RMSE | Mean $ RMSE | Mean skill |
|---|---|---|---|---|
| level | train only | 1.71209 | $57.67 | −8.0148 |
| level | **all data** | **0.02444** | **$7.85** | **−0.1545** |
| return | train only | 0.02923 | $6.86 | −0.0067 |
| return | **all data** | **0.02902** | **$6.81** | **+0.0006** |

Two things worth sitting with:

1. **The leak flips the sign.** The return arm goes from −0.0067 to **+0.0006** — from losing
   to the baseline to beating it. A paper reporting +0.0006 as skill would be reporting the
   leak, not the model.
2. **It cuts the level arm's scaled-unit RMSE by 98.6%**, from 1.712 to 0.024. That is the
   number a scaled-unit report would print, and it looks like a working model.

This is why nothing here is reported in scaled units, and why the scaler is a class that
refuses to be refitted rather than a convention to be remembered.

## Install and run

Requires Python 3.10+. Data snapshots are committed, so no network access and no API key are
needed.

```bash
git clone https://github.com/Ssavan99/stock-lstm-vs-persistence
cd stock-lstm-vs-persistence

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.run_baselines      # baseline table, ~5 seconds
python -m scripts.run_experiment_a   # LSTM vs persistence, ~6 minutes on CPU
python -m scripts.run_experiment_b   # sentiment ablation, ~3 minutes
python -m scripts.run_experiment_c   # leakage demonstration, ~10 minutes
```

Each script prints its table and writes JSON to `results/`. To rebuild the results page from
those files:

```bash
python -m scripts.build_site
```

To refresh market data to today (extends the held-out period):

```bash
python -m src.data.prices --refresh
```

Tests and linting:

```bash
pytest -q
ruff check src scripts tests
```

## Layout

```
src/
  data/prices.py       yfinance fetch, committed snapshot, schema
  data/sentiment.py    sentiment aligned onto the trading calendar
  data/splits.py       fixed-cutoff chronological splits
  preprocessing.py     TrainOnlyScaler — fits once, on training rows
  sequences.py         windowing, target construction, dollar inversion
  baselines.py         persistence, drift, AR(p)
  metrics.py           RMSE / MAE / MAPE in dollars, skill score
  models/lstm.py       bidirectional LSTM regressor
  train.py             training loop, early stopping, seeding
scripts/               experiment runners and site builder
results/               committed JSON output
docs/                  static results page (GitHub Pages)
```

## Limitations

- **One universe, one period, one architecture.** The finding is about this task as posed,
  not about LSTMs in general.
- **Daily bars only.** No intraday data, no order book, no fundamentals, no macro.
- **Sentiment is a fixed 2020-12 → 2024-03 snapshot** from a source that now requires a paid
  key, so the ablation cannot run on the main held-out period and cannot be refreshed.
- **No hyperparameter search.** One configuration, chosen up front and held fixed across arms
  so the comparisons stay attributable. A tuned model might close some of the return arm's
  0.7% gap; nothing suggests it would close the level arm's.
- **Point forecasts only.** No uncertainty intervals.
- **Not a trading system and not advice.** No costs, slippage, sizing or execution — nothing
  here was backtested as a strategy.

## License

MIT — see [LICENSE](LICENSE).
