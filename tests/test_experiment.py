import numpy as np
import pandas as pd
import pytest

from riskbt import config, experiment


def simulated_prices(days=380, assets=4, seed=41):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=days)
    returns = rng.standard_t(5, (days, assets + 1)) * 0.012
    columns = [f"S{i}.SA" for i in range(assets)] + [config.IBOVESPA_TICKER]
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=columns)


UNIVERSE = {
    **{f"C{i}.SA": "steel" for i in range(9)}, **{f"M{i}.SA": "mining" for i in range(9)},
    **{f"B{i}.SA": "financials" for i in range(8)}, **{f"U{i}.SA": "utilities" for i in range(8)},
    **{f"H{i}.SA": "health" for i in range(8)},
}


@pytest.fixture(scope="module")
def design():
    return experiment.build_design(UNIVERSE)


def test_design_is_full_factorial(design):
    assert len(design) == 36
    assert len({spec.name for spec in design}) == 36
    for size in config.PORTFOLIO_SIZES:
        assert sum(spec.n_assets == size for spec in design) == 12
    assert sum(spec.weighting == "dirichlet" for spec in design) == 18
    assert sum(spec.profile == "commodity" for spec in design) == 18


def test_weights_are_valid(design):
    for spec in design:
        assert len(spec.weights) == spec.n_assets
        assert sum(spec.weights.values()) == pytest.approx(1)
        if spec.weighting == "equal":
            assert set(np.round(list(spec.weights.values()), 10)) == {round(1 / spec.n_assets, 10)}


def test_commodity_profile_share(design):
    for spec in design:
        share = np.mean([UNIVERSE[t] in config.COMMODITY_SECTORS for t in spec.weights])
        if spec.profile == "commodity":
            assert share == pytest.approx(round(0.7 * spec.n_assets) / spec.n_assets)
        else:
            assert len({UNIVERSE[t] for t in spec.weights}) == min(spec.n_assets, 5)


def test_design_depends_only_on_the_seed(design):
    assert experiment.build_design(UNIVERSE) == design
    assert experiment.build_design(UNIVERSE, seed=1) != design


def test_draws_differ(design):
    compositions = {tuple(sorted(spec.weights)) for spec in design if spec.n_assets == 20 and spec.profile == "commodity"}
    assert len(compositions) > 1


def test_eligible_universe_applies_both_rules():
    index = pd.bdate_range("2010-01-01", "2013-12-31")
    prices = pd.DataFrame({"OLD": 1.0, "LATE": 1.0, "GAPPY": 1.0}, index=index)
    prices.loc[:"2012-06-01", "LATE"] = np.nan
    prices.iloc[100:200, 2] = np.nan
    assert experiment.eligible_universe(prices, {"OLD": "x", "LATE": "x", "GAPPY": "x", "NONE": "x"}) == {"OLD": "x"}


def test_characteristics():
    prices = simulated_prices()
    spec = experiment.PortfolioSpec("p", 2, "equal", "commodity", 1, {"S0.SA": 0.6, "S1.SA": 0.4})
    result = experiment.portfolio_characteristics(prices, spec, {"S0.SA": "steel", "S1.SA": "health"})
    assert result["hhi"] == pytest.approx(0.52)
    assert result["commodity_weight"] == pytest.approx(0.6)
    index_spec = experiment.ibovespa_spec()
    assert experiment.portfolio_characteristics(prices, index_spec, {})["beta"] == pytest.approx(1)


def test_full_analysis_is_reproducible_and_exports():
    prices = simulated_prices()
    weights = {"S0.SA": 0.5, "S1.SA": 0.5}
    first = experiment.analyze_portfolio(prices, weights, window=300, refit_every=40)
    second = experiment.analyze_portfolio(prices, weights, window=300, refit_every=40)
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts)
    pd.testing.assert_frame_equal(first.tests, second.tests)
    pd.testing.assert_frame_equal(first.parameters, second.parameters)
    assert first.recommendation.estimator == second.recommendation.estimator
    assert experiment.export_csv(first) == experiment.export_csv(second)
    assert "section" in experiment.export_csv(first).splitlines()[0]


def test_short_history_is_refused():
    with pytest.raises(experiment.InsufficientHistory) as error:
        experiment.analyze_portfolio(simulated_prices(days=100), {"S0.SA": 1.0}, window=300)
    assert error.value.needed == 301
