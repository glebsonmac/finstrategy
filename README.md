# riskbt

VaR and Expected Shortfall backtesting for Brazilian equity portfolios. Enter B3 tickers and
weights, and the system downloads prices from Yahoo Finance, computes one-day VaR and ES with six
estimators, backtests them separately in calm and crisis regimes, recommends the most reliable
estimator and explains every calculation, in English or Portuguese.

Risk measurement tool. Not investment advice.

## Install

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the app

```bash
streamlit run app.py --server.address localhost
```

A new portfolio takes about a minute: each trading day since 2012 gets a forecast.

## Run the experiment

Download and freeze the data, then run the Ibovespa and the 36 planned portfolios on it. Data is
not stored in git, and any machine can recreate it with these commands.

```bash
python scripts/freeze_data.py      # version named after today, in data/frozen/
python scripts/check_regimes.py <version>   # optional chart of the crisis periods
python scripts/run_experiment.py   # uses the latest version, writes results/<run_date>/
```

A frozen version is never replaced, and its hash is checked every time it is loaded. Yahoo revises
adjusted prices, so a new extraction may differ slightly from an older one. To compare machines
exactly, copy `data/frozen/<version>/` and run `python scripts/run_experiment.py <version>`.

| Output | Content |
|---|---|
| `forecasts.parquet` | portfolio, estimator, alpha, date, regime, return, var, es, hit, fz0_loss, quantile_loss |
| `parameters.parquet` | parameters of each refit |
| `tests.csv` | every test by portfolio, estimator, level and regime |
| `portfolios.csv` | composition, design factors, HHI, beta, commodity weight |
| `metadata.json` | data hash, seeds, package versions, parameters, data report |

## Run the tests

```bash
pytest
```

The tests need no network. They check estimators against theoretical values, the size of the
statistical tests under the null, the absence of look-ahead, reproducibility, the locale files and
the page itself.

## Layout

```
riskbt/   config, data, portfolio, regimes, estimators, rolling, backtest,
          recommend, explain, i18n, locales, experiment
app.py    Streamlit interface
scripts/  freeze_data, check_regimes, run_experiment
tests/
```

`projeto_mestrado_VaR_ES.pdf` (Portuguese) holds the formulas and the experimental design.
`src/finstrategy/pricing/garch_montecarlo.py` and its tests belong to an earlier project and remain
only because a notebook in `notebooks/` imports that module.
