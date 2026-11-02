# CLAUDE.md — VaR and ES Backtesting System for Brazilian Equity Portfolios

Read `projeto_mestrado_VaR_ES.pdf` first. It is the specification: estimators, formulas,
tests, regimes and experimental design are there (chapters 2 and 5, written in Portuguese).
This file says **how** to build it. When in doubt, follow the PDF and ask before inventing.

## Goal in one sentence

The user enters a portfolio of B3 stocks (tickers and weights). The system downloads data from
Yahoo Finance, computes VaR and ES with six estimators, backtests them separately in calm and
crisis regimes, recommends the most reliable estimator and **explains every calculation** at
several levels of detail, in **Portuguese or English** (user's choice). The same code runs the
thesis experiment and saves all intermediate data.

## Principles (apply to all code)

1. **Simple and readable.** Code a master's student can read and explain to a committee.
   Short functions, clear names, no tricks.
2. **Pure functions.** Input → output, no global state, no side effects. Only `data.py`
   (network and disk) and `app.py` (screen) have side effects.
3. **Use what already exists.** Do not reimplement what a library already does well
   (GARCH in `arch`, distributions in `scipy.stats`, GPD in `scipy.stats.genpareto`).
4. **Optimize with NumPy, not parallelism.** Vectorize; avoid loops over pandas rows.
   **Do not use** multiprocessing, joblib, threads, Ray, Dask or async.
5. **No heavy frameworks.** Forbidden: Django, FastAPI, Flask, React, Vue, Node, ORMs,
   databases, Celery, Docker, Kubernetes, i18n frameworks (gettext/Babel). No class
   inheritance; use functions and `@dataclass(frozen=True)` to group results.
6. **Reproducible.** Every random step takes an explicit `seed`. Every saved result records
   data version, seeds and package versions.
7. **Do not grow the scope.** Six estimators, the tests in the PDF, one interface. Nothing
   else unless asked.

## Language rules

- **Code in English:** module, function, variable and column names, comments, docstrings,
  commit messages, README.
- **User-facing text in two languages:** every string shown on screen (labels, summaries,
  explanations, warnings, chart titles) lives in `riskbt/locales/en.json` and
  `riskbt/locales/pt.json`, with **identical keys**. Never hardcode UI text in Python.
- A tiny helper does the lookup, nothing more:

  ```python
  def t(key: str, lang: str, **values) -> str:
      """Return the translated text for `key`, filling {placeholders} with `values`."""
  ```
- Language selector at the top of the app (`Português` / `English`), default `en`.
- Number formatting follows the language: `pt` → `R$ 3.490,00` and `3,49%`;
  `en` → `R$ 3,490.00` and `3.49%`. Currency stays BRL in both.
- Formulas (`st.latex`) are the same in both languages; only the surrounding text changes.
- Saved data files (Parquet/CSV) always use **English column names**, so results are
  language-independent. Only the downloadable CSV in the app may use translated headers.
- A test checks that `en.json` and `pt.json` have exactly the same keys.

## Dependencies (only these)

```
python >= 3.11
numpy, pandas, scipy, statsmodels   # computation
arch                                 # GARCH(1,1)-t
yfinance                             # data
pyarrow                              # Parquet cache
streamlit                            # interface
altair                               # charts (ships with Streamlit)
pytest                               # tests
```

Need another library? **Ask before adding it.**

## Structure

```
riskbt/
  config.py         # fixed parameters (table below)
  data.py           # download, validate, Parquet cache, freeze a data version
  portfolio.py      # daily log return of a constant-weight portfolio
  regimes.py        # 21-day realized volatility of Ibovespa, expanding 80th percentile
  estimators.py     # hs, normal, garch_t, fhs, evt_pot, caviar
  rolling.py        # rolling window: 1000 days, refit every 20, forecast t+1
  backtest.py       # kupiec, christoffersen, traffic_light, z2, fz0_loss, diebold_mariano
  recommend.py      # picks the estimator for the portfolio (rule below)
  explain.py        # builds the layered explanations from `details` + locale texts
  i18n.py           # t() and number formatting
  locales/en.json
  locales/pt.json
  experiment.py     # 36 portfolios of the factorial design + Ibovespa
app.py              # Streamlit interface
scripts/run_experiment.py
tests/
data/               # cache (git-ignored)
results/            # experiment outputs (git-ignored)
```

## Fixed parameters (from the PDF; kept in `riskbt/config.py`)

| Parameter | Value |
|---|---|
| Period | 2008-01-01 to the data freeze date |
| Estimation window | 1000 trading days |
| Parameter refit | every 20 days (volatility updated daily) |
| Levels | VaR 99% (α = 0.01); VaR and ES 97.5% (α = 0.025) |
| Regime | crisis if 21-day RV at t−1 > 80th percentile of all history up to t−1; at least 250 days before classifying |
| EVT | threshold u = 90% quantile of standardized losses |
| CAViaR | asymmetric slope (Engle and Manganelli, 2004), VaR only |
| Z2 | simulated p-value, 1000 replications, fixed seed |
| Diebold–Mariano | Newey–West variance |
| Traffic light | rolling 250-day windows: green 0–4, yellow 5–9, red ≥ 10 |

## Estimator contract

Every estimator is a pure function with the same signature:

```python
def garch_t(returns: np.ndarray, alpha: float) -> Forecast: ...

@dataclass(frozen=True)
class Forecast:
    var: float            # positive loss
    es: float | None      # None for CAViaR
    details: dict         # intermediate values used by the explanation
```

`details` stores **everything the explanation must show**: e.g. for GARCH-t, ω, a, b, ν,
σ̂ₜ₊₁, the t quantile, persistence a+b and half-life. Nothing is recomputed on screen.

Each function has a docstring with: what it does, the formula (same as in the PDF) and the
reference (author, year).

## Recommendation rule (simple and explainable)

1. Keep only estimators that pass (p ≥ 0.05) Kupiec and Christoffersen in the current regime
   and, if they provide ES, also Z2.
2. Among them, recommend the one with the lowest mean FZ0 loss in the current regime (CAViaR
   competes on quantile loss only if no other estimator passes).
3. If none passes, say so clearly and show the least bad one, with a warning.

The screen always shows **why** that estimator was chosen.

## Interface: layered explanations

One Streamlit page, lightweight, no heavy custom CSS. Every block has **three layers**:

1. **Summary** in plain language for non-technical users. E.g. (en) "On 99 out of 100 days,
   the portfolio is not expected to lose more than R$ 3,490."
2. **Chart**, always with one sentence saying what it shows.
3. **"See more" / "Ver mais"** (`st.expander`) for technical users:
   - the formula in `st.latex`;
   - the step-by-step numbers with the portfolio's actual values (from `details`);
   - assumptions and limitations of the method;
   - the bibliographic reference.

Page blocks, in this order:
1. Language selector; input: `.SA` tickers, weights (must sum to 1), amount invested in BRL.
2. Main result: VaR 99%, ES 97.5%, current regime, recommended estimator.
3. Comparison of the six estimators (table + "See more" for each).
4. Backtesting: violations over time with crisis periods shaded; traffic light; tests by regime.
5. Data: source, version, gaps corrected.
6. Button to download a CSV with all forecasts and tests for the portfolio.

Accessibility and readability:
- Never use color alone to convey meaning (traffic light also has text).
- Adequate contrast; short sentences; no jargon in the summary layer.
- Fixed notice: "Risk measurement tool. Not investment advice." / "Ferramenta de medição de
  risco. Não é recomendação de investimento."

## Experiment outputs (for the thesis)

`scripts/run_experiment.py` writes to `results/<run_date>/`:

- `forecasts.parquet`: portfolio, estimator, date, regime, return, var, es, hit, fz0_loss;
- `parameters.parquet`: parameters of each refit (e.g. omega, a, b, nu, xi, beta, u);
- `tests.csv`: every test by portfolio × estimator × regime (statistic and p-value);
- `portfolios.csv`: composition, design factors, HHI, beta, commodity weight;
- `metadata.json`: data hash, seeds, package versions, `config.py` parameters.

Nothing is aggregated without keeping the raw data it came from.

## Required tests (`pytest`)

- Estimators recover theoretical VaR and ES on simulated series (normal, t, known GARCH):
  relative error < 2%.
- Statistical tests: rejection rate between 3% and 7% under H0 at the 5% level.
- Reproducibility: two runs with the same data and seeds give identical results.
- No look-ahead: the t+1 forecast does not change if data after t is altered.
- Locales: `en.json` and `pt.json` have the same keys and every key used in code exists.

## Work order

Do **one step at a time**. At the end of each step: run the tests, show a short summary of what
was done and **stop for my review** before continuing.

1. `config.py`, `data.py`, `portfolio.py` + tests. Download and freeze one data version.
2. `regimes.py` + a simple check chart of the crisis periods.
3. Estimators `hs`, `normal`, `garch_t`, `fhs` + tests against theoretical values.
4. Estimators `evt_pot` and `caviar` + tests.
5. `rolling.py` and `backtest.py` + no-look-ahead test.
6. `recommend.py`, `i18n.py`, `locales/` and `explain.py`.
7. `app.py` with the three layers and the language selector.
8. `experiment.py` and `scripts/run_experiment.py`; run it and produce `results/`.
9. Short README in English: install, run the app, run the experiment.

## Style

- Type hints on all public functions.
- Functions up to ~30 lines; split if longer.
- Comment only where the math is not obvious, pointing to the PDF equation number.
- No dead code, no debug prints, no notebooks as main code.
