"""Portfolio analysis shared by the app and the experiment, and the 36-portfolio design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskbt import backtest, config
from riskbt.estimators import Forecast
from riskbt.portfolio import portfolio_log_returns
from riskbt.recommend import Recommendation, recommend
from riskbt.regimes import classify_regimes, next_day_regime
from riskbt.rolling import latest_forecasts, rolling_forecasts


class InsufficientHistory(Exception):
    """The portfolio has too few common days for one estimation window."""

    def __init__(self, days: int, needed: int) -> None:
        super().__init__(f"{days} days of history, {needed} needed.")
        self.days, self.needed = days, needed


@dataclass(frozen=True)
class PortfolioAnalysis:
    returns: pd.Series
    regimes: pd.Series
    forecasts: pd.DataFrame  # with regime and both losses
    parameters: pd.DataFrame
    tests: pd.DataFrame
    latest: dict[tuple[str, float], Forecast]
    recommendation: Recommendation


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    n_assets: int
    weighting: str
    profile: str
    draw: int
    weights: dict[str, float]


def index_returns(prices: pd.DataFrame) -> pd.Series:
    return np.log(prices[config.IBOVESPA_TICKER]).diff().dropna()


def analyze_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    window: int = config.ESTIMATION_WINDOW,
    refit_every: int = config.REFIT_EVERY,
) -> PortfolioAnalysis:
    """Run the full protocol on one portfolio: forecasts, tests by regime, recommendation."""
    returns = portfolio_log_returns(prices, weights)
    if len(returns) <= window:
        raise InsufficientHistory(len(returns), window + 1)
    market = index_returns(prices)
    regimes = classify_regimes(market)
    rolling = rolling_forecasts(returns, window=window, refit_every=refit_every)
    forecasts = backtest.add_losses(rolling.forecasts)
    tests = backtest.evaluate(forecasts, rolling.tail_losses, regimes)
    regime = next_day_regime(market)
    forecasts = forecasts.assign(regime=regimes.reindex(forecasts["date"]).to_numpy())
    latest = latest_forecasts(returns, window=window)
    return PortfolioAnalysis(returns, regimes, forecasts, rolling.parameters, tests, latest, recommend(tests, regime))


def eligible_universe(prices: pd.DataFrame, universe: dict[str, str] = config.UNIVERSE) -> dict[str, str]:
    """Keep assets listed early enough and with few missing days after their first price."""
    last_start = pd.Timestamp(config.UNIVERSE_LAST_START)
    eligible = {}
    for ticker, sector in universe.items():
        if ticker not in prices.columns or prices[ticker].first_valid_index() is None:
            continue
        series = prices[ticker].loc[prices[ticker].first_valid_index() :]
        if series.index[0] <= last_start and series.isna().mean() <= config.UNIVERSE_MAX_MISSING_SHARE:
            eligible[ticker] = sector
    return eligible


def _draw_commodity(universe: dict[str, str], size: int, rng: np.random.Generator) -> list[str]:
    """Draw 70% of the assets from commodity sectors and the rest from the other sectors."""
    commodity = sorted(t for t, s in universe.items() if s in config.COMMODITY_SECTORS)
    others = sorted(t for t, s in universe.items() if s not in config.COMMODITY_SECTORS)
    n_commodity = round(config.COMMODITY_PROFILE_SHARE * size)
    if n_commodity > len(commodity):
        raise ValueError(f"Only {len(commodity)} commodity assets for {n_commodity} requested.")
    chosen = rng.choice(commodity, n_commodity, replace=False).tolist()
    return chosen + rng.choice(others, size - n_commodity, replace=False).tolist()


def _draw_diversified(universe: dict[str, str], size: int, rng: np.random.Generator) -> list[str]:
    """Take one asset from each sector in random order, going around until `size` is reached."""
    pools = {
        sector: rng.permutation(sorted(t for t, s in universe.items() if s == sector)).tolist()
        for sector in sorted(set(universe.values()))
    }
    chosen: list[str] = []
    while len(chosen) < size:
        for sector in rng.permutation(sorted(pools)):
            if pools[sector] and len(chosen) < size:
                chosen.append(pools[sector].pop())
    return chosen


def build_design(universe: dict[str, str], seed: int = config.EXPERIMENT_SEED) -> list[PortfolioSpec]:
    """Build the 3 x 2 x 2 factorial design with 3 draws per configuration (table 5.2)."""
    rng = np.random.default_rng(seed)
    draw_assets = {"commodity": _draw_commodity, "diversified": _draw_diversified}
    specs = []
    for size in config.PORTFOLIO_SIZES:
        for weighting in config.WEIGHT_SCHEMES:
            for profile in config.SECTOR_PROFILES:
                for draw in range(1, config.DRAWS_PER_CONFIGURATION + 1):
                    tickers = draw_assets[profile](universe, size, rng)
                    raw = np.full(size, 1 / size) if weighting == "equal" else rng.dirichlet(np.ones(size))
                    weights = dict(zip(sorted(tickers), (float(w) for w in raw)))
                    specs.append(PortfolioSpec(f"n{size}_{weighting}_{profile}_{draw}", size, weighting, profile, draw, weights))
    return specs


def ibovespa_spec() -> PortfolioSpec:
    return PortfolioSpec("ibovespa", 1, "index", "index", 0, {config.IBOVESPA_TICKER: 1.0})


def portfolio_characteristics(prices: pd.DataFrame, spec: PortfolioSpec, universe: dict[str, str]) -> dict:
    """Design factors plus HHI, beta against the Ibovespa and commodity weight (section 5.3)."""
    returns = portfolio_log_returns(prices, spec.weights)
    market = index_returns(prices).reindex(returns.index)
    beta = np.cov(returns, market)[0, 1] / market.var()
    commodity = sum(w for t, w in spec.weights.items() if universe.get(t) in config.COMMODITY_SECTORS)
    return {
        "portfolio": spec.name, "n_assets": spec.n_assets, "weighting": spec.weighting,
        "profile": spec.profile, "draw": spec.draw, "hhi": sum(w**2 for w in spec.weights.values()),
        "beta": float(beta), "commodity_weight": commodity, "first_date": returns.index[0],
        "composition": ";".join(f"{t}={w:.6f}" for t, w in spec.weights.items()),
    }


def export_csv(analysis: PortfolioAnalysis) -> str:
    """All forecasts and tests of a portfolio in one CSV, told apart by the `section` column."""
    forecasts = analysis.forecasts.assign(section="forecast")
    tests = analysis.tests.assign(section="test")
    return pd.concat([forecasts, tests], ignore_index=True).to_csv(index=False)
