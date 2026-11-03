"""Run the thesis experiment and write results/<run_date>/.

Usage: python scripts/run_experiment.py [data_version] [run_date]

The data version defaults to the most recent frozen one, and the run date to today.
"""

import json
import sys
import time
from dataclasses import asdict
from datetime import date
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskbt import config, data, experiment  # noqa: E402

FORECAST_COLUMNS = ["portfolio", "estimator", "alpha", "date", "regime", "return", "var", "es", "hit", "fz0_loss", "quantile_loss"]


def config_values() -> dict:
    """Upper-case values of config.py, in a form JSON can store."""
    values = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    breaks = {f"{ticker} {day}": reason for (ticker, day), reason in config.KNOWN_PRICE_BREAKS.items()}
    return json.loads(json.dumps({**values, "KNOWN_PRICE_BREAKS": breaks}, default=lambda v: sorted(v) if isinstance(v, frozenset) else str(v)))


def write_metadata(folder: Path, frozen: data.PriceData, universe: dict[str, str], seconds: dict) -> None:
    metadata = {
        "data_version": frozen.version,
        "data_sha256": frozen.sha256,
        "seeds": {"experiment": config.EXPERIMENT_SEED, "z2": config.Z2_SEED, "caviar": config.CAVIAR_SEED},
        "package_versions": {name: package_version(name) for name in data.RECORDED_PACKAGES},
        "config": config_values(),
        "eligible_universe": universe,
        "data_report": asdict(frozen.report),
        "seconds_per_portfolio": seconds,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(data_version: str, run_date: str) -> None:
    frozen = data.load_frozen(data_version)
    universe = experiment.eligible_universe(frozen.prices)
    specs = [experiment.ibovespa_spec(), *experiment.build_design(universe)]
    folder = config.RESULTS_DIR / run_date
    if folder.exists():
        sys.exit(f"{folder} already exists and is never replaced. Choose another run date:\n"
                 f"  python scripts/run_experiment.py {data_version} {run_date}-2")
    folder.mkdir(parents=True)

    forecasts, parameters, tests, portfolios, seconds = [], [], [], [], {}
    for position, spec in enumerate(specs, start=1):
        started = time.perf_counter()
        analysis = experiment.analyze_portfolio(frozen.prices, spec.weights)
        seconds[spec.name] = round(time.perf_counter() - started, 1)
        forecasts.append(analysis.forecasts.assign(portfolio=spec.name)[FORECAST_COLUMNS])
        parameters.append(analysis.parameters.assign(portfolio=spec.name))
        tests.append(analysis.tests.assign(portfolio=spec.name))
        portfolios.append(experiment.portfolio_characteristics(frozen.prices, spec, universe))
        print(f"{position}/{len(specs)} {spec.name}: {seconds[spec.name]} s", flush=True)

    pd.concat(forecasts, ignore_index=True).to_parquet(folder / "forecasts.parquet", index=False)
    pd.concat(parameters, ignore_index=True).to_parquet(folder / "parameters.parquet", index=False)
    pd.concat(tests, ignore_index=True).to_csv(folder / "tests.csv", index=False)
    pd.DataFrame(portfolios).to_csv(folder / "portfolios.csv", index=False)
    write_metadata(folder, frozen, universe, seconds)
    print(f"Results at {folder}")


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else data.latest_frozen_version()
    main(version, sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat())
