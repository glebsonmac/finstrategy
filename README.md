# riskbt

VaR and Expected Shortfall backtesting for Brazilian equity portfolios. Enter B3 tickers and
weights, and the system downloads prices from Yahoo Finance, computes one-day VaR and ES with six
estimators, backtests them separately in calm and crisis regimes, recommends the most reliable
estimator and explains every calculation, in English or Portuguese.

Risk measurement tool. Not investment advice.

## Install

Python 3.11 or later. The tests passed in new environments with Python 3.11, 3.12 and 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the tests passed with. After `source`, `python` and `pip` come
from the environment, and the commands below assume it is active.

If the app or a script stops with `No module named ...`, the active environment is missing the
dependencies: activate it and run the `pip install` line again. If `.venv` was copied from another
machine or the project folder moved, delete `.venv` and create it again.

## Run the app

```bash
streamlit run app.py --server.address localhost
```

A new portfolio can take up to a minute: each trading day since 2012 gets a forecast. Results
stay in memory until the app is closed. Weights accept a dot or a comma as the decimal separator.

## Examples by sector

Three example portfolios. In the app, type each table into the portfolio grid and click
**Calculate risk**. The app accepts tickers with or without `.SA`.

| Electric utilities | Weight | | Electric utilities and oil and gas | Weight | | Mixed | Weight |
|---|---|---|---|---|---|---|---|
| CMIG4 | 0.2 | | CMIG4 | 0.2 | | CMIG4 | 0.2 |
| CPFE3 | 0.2 | | EGIE3 | 0.2 | | PETR4 | 0.2 |
| EGIE3 | 0.2 | | EQTL3 | 0.2 | | VALE3 | 0.2 |
| EQTL3 | 0.2 | | PETR3 | 0.1 | | ITUB4 | 0.2 |
| TAEE11 | 0.2 | | PETR4 | 0.2 | | ABEV3 | 0.1 |
| | | | PRIO3 | 0.1 | | WEGE3 | 0.1 |

The mixed portfolio combines electric utilities, oil and gas, mining, banking, consumer goods and
industrials. These are examples of how to run the system, not suggested allocations.

To run all three without the app, freeze the data once with `python scripts/freeze_data.py`, save
this code as `sector_examples.py` in the project root and run `python sector_examples.py`. It takes
less than a minute for the three portfolios.

```python
from riskbt import config, data, experiment

PORTFOLIOS = {
    "electric": {"CMIG4.SA": 0.2, "CPFE3.SA": 0.2, "EGIE3.SA": 0.2, "EQTL3.SA": 0.2, "TAEE11.SA": 0.2},
    "electric_oil_gas": {
        "CMIG4.SA": 0.2, "EGIE3.SA": 0.2, "EQTL3.SA": 0.2,
        "PETR3.SA": 0.1, "PETR4.SA": 0.2, "PRIO3.SA": 0.1,
    },
    "mixed": {
        "CMIG4.SA": 0.2, "PETR4.SA": 0.2, "VALE3.SA": 0.2,
        "ITUB4.SA": 0.2, "ABEV3.SA": 0.1, "WEGE3.SA": 0.1,
    },
}

prices = data.load_frozen(data.latest_frozen_version()).prices
for name, weights in PORTFOLIOS.items():
    analysis = experiment.analyze_portfolio(prices, weights)
    choice = analysis.recommendation
    var_99 = analysis.latest[(choice.estimator, config.ALPHA_VAR)].var
    print(f"{name}: {choice.estimator} ({choice.status}), regime {choice.regime}, VaR 99% {var_99:.2%}")
```

## Run the experiment

Download and freeze the data, then run the Ibovespa and the 36 planned portfolios on it. Data is
not stored in git, and any machine can recreate it with these commands.

```bash
python scripts/freeze_data.py      # version named after today, in data/frozen/
python scripts/check_regimes.py <version>   # optional chart of the crisis periods
python scripts/run_experiment.py   # uses the latest version, writes results/<run_date>/
```

The experiment prints one line per portfolio and takes about 6 minutes. An existing results folder
is never replaced: to run again on the same day, name another one with
`python scripts/run_experiment.py <version> <run_date>`.

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
