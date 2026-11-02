"""Estimators must recover theoretical VaR and ES within 2% on simulated series."""

import numpy as np
import pytest
from scipy import stats

from riskbt import estimators as est
from riskbt.config import ALPHA_ES, ALPHA_VAR

MAX_RELATIVE_ERROR = 0.02
NU = 6
GARCH = {"mu": 0.0003, "omega": 2e-6, "a": 0.08, "b": 0.90, "nu": NU}


def normal_truth(mu, sigma, alpha):
    z = stats.norm.ppf(alpha)
    return -(mu + sigma * z), -mu + sigma * stats.norm.pdf(z) / alpha


def assert_close(forecast, var, es=None):
    assert forecast.var == pytest.approx(var, rel=MAX_RELATIVE_ERROR)
    if es is not None:
        assert forecast.es == pytest.approx(es, rel=MAX_RELATIVE_ERROR)


def simulate_garch(n, params, seed):
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(params["nu"], n) * np.sqrt((params["nu"] - 2) / params["nu"])
    variance = params["omega"] / (1 - params["a"] - params["b"])
    returns = np.empty(n)
    for i in range(n):
        epsilon = np.sqrt(variance) * shocks[i]
        returns[i] = params["mu"] + epsilon
        variance = params["omega"] + params["a"] * epsilon**2 + params["b"] * variance
    return returns, float(np.sqrt(variance))


@pytest.fixture(scope="module")
def garch_series():
    return simulate_garch(30_000, GARCH, seed=11)


@pytest.mark.parametrize("alpha", [ALPHA_VAR, ALPHA_ES])
def test_hs_on_normal(alpha):
    returns = np.random.default_rng(1).normal(0.001, 0.02, 1_000_000)
    assert_close(est.hs(returns, alpha), *normal_truth(0.001, 0.02, alpha))


def test_hs_formula_on_small_sample():
    returns = np.arange(1.0, 101.0) / 100 - 0.5  # -0.49, -0.48, ... with n = 100
    forecast = est.hs(returns, 0.025)
    assert forecast.details["k"] == 2
    assert forecast.var == pytest.approx(0.48)
    assert forecast.es == pytest.approx(0.485)


@pytest.mark.parametrize("alpha", [ALPHA_VAR, ALPHA_ES])
def test_hs_on_student_t(alpha):
    nu = 5
    returns = np.random.default_rng(2).standard_t(nu, 2_000_000)
    quantile = stats.t.ppf(alpha, nu)
    es = stats.t.pdf(quantile, nu) / alpha * (nu + quantile**2) / (nu - 1)
    assert_close(est.hs(returns, alpha), -quantile, es)


@pytest.mark.parametrize("alpha", [ALPHA_VAR, ALPHA_ES])
def test_normal_on_normal(alpha):
    returns = np.random.default_rng(3).normal(0.001, 0.02, 200_000)
    assert_close(est.normal(returns, alpha), *normal_truth(0.001, 0.02, alpha))


def test_values_printed_in_the_pdf_figure_1():
    sigma = 0.015
    t_var975, t_es975 = est.student_t_tail(4, 0.025)
    t_var99, _ = est.student_t_tail(4, 0.01)
    assert -sigma * t_var975 == pytest.approx(0.0294, abs=5e-5)
    assert -sigma * t_es975 == pytest.approx(0.0424, abs=5e-5)
    assert -sigma * t_var99 == pytest.approx(0.0397, abs=5e-5)
    assert normal_truth(0, sigma, 0.025)[1] == pytest.approx(0.0351, abs=5e-5)
    assert normal_truth(0, sigma, 0.01)[0] == pytest.approx(0.0349, abs=5e-5)


def test_student_t_tail_matches_numerical_integral():
    nu, alpha = 5, 0.025
    quantile, tail_mean = est.student_t_tail(nu, alpha)
    scale = np.sqrt((nu - 2) / nu)
    draws = stats.t.ppf(np.linspace(1e-7, alpha, 2_000_001), nu) * scale
    assert quantile == pytest.approx(draws[-1], rel=1e-6)
    assert tail_mean == pytest.approx(draws.mean(), rel=1e-2)


def test_garch_parameters_are_recovered(garch_series):
    fit = est.fit_garch(garch_series[0])
    assert fit["converged"]
    assert fit["a"] == pytest.approx(GARCH["a"], abs=0.015)
    assert fit["b"] == pytest.approx(GARCH["b"], abs=0.02)
    assert fit["nu"] == pytest.approx(GARCH["nu"], abs=0.7)


def test_garch_volatility_recursion(garch_series):
    returns, true_next = garch_series
    path = est.garch_volatility(returns, GARCH)
    assert len(path) == len(returns) + 1
    assert path[-1] == pytest.approx(true_next, rel=1e-6)
    manual = GARCH["omega"] + GARCH["a"] * (returns[-1] - GARCH["mu"]) ** 2 + GARCH["b"] * path[-2] ** 2
    assert path[-1] ** 2 == pytest.approx(manual)


@pytest.fixture(scope="module")
def garch_samples():
    return [simulate_garch(30_000, GARCH, seed) for seed in range(6)]


@pytest.mark.parametrize("alpha", [ALPHA_VAR, ALPHA_ES])
@pytest.mark.parametrize("estimator", [est.garch_t, est.fhs, est.evt_pot])
def test_garch_based_estimators_recover_known_garch(garch_samples, estimator, alpha):
    """The mean relative error over seeds measures bias; one seed alone also carries sampling noise."""
    quantile, tail_mean = est.student_t_tail(GARCH["nu"], alpha)
    errors = []
    for returns, sigma in garch_samples:
        forecast = estimator(returns, alpha)
        errors.append((
            forecast.var / -(GARCH["mu"] + sigma * quantile) - 1,
            forecast.es / -(GARCH["mu"] + sigma * tail_mean) - 1,
        ))
    assert np.all(np.abs(np.mean(errors, axis=0)) < MAX_RELATIVE_ERROR)


def test_garch_details_are_complete(garch_series):
    details = est.garch_t(garch_series[0][:2000], ALPHA_ES).details
    for key in ("omega", "a", "b", "nu", "sigma_next", "quantile", "persistence", "half_life"):
        assert key in details
    assert details["half_life"] == pytest.approx(np.log(0.5) / np.log(details["a"] + details["b"]))


def test_half_life_example_from_the_pdf():
    returns = np.random.default_rng(5).normal(0, 0.01, 500)
    _, details = est._garch_details(returns, {"mu": 0.0, "omega": 1e-6, "a": 0.08, "b": 0.90, "nu": 5.0})
    assert details["half_life"] == pytest.approx(34.3, abs=0.1)


def test_gpd_measures_on_exact_gpd_tail():
    xi, beta, u = 0.2, 0.5, 1.0
    draws = u + stats.genpareto.rvs(xi, scale=beta, size=2_000_000, random_state=6)
    # all observations are above u, so the tail share is 1
    quantile, es = est.gpd_tail_measures(u, xi, beta, 1.0, 0.01)
    assert quantile == pytest.approx(np.quantile(draws, 0.99), rel=0.01)
    assert es == pytest.approx(draws[draws > quantile].mean(), rel=0.02)


def test_gpd_exponential_limit():
    with_small_xi = est.gpd_tail_measures(1.0, 1e-7, 0.5, 0.1, 0.01)[0]
    at_zero = est.gpd_tail_measures(1.0, 0.0, 0.5, 0.1, 0.01)[0]
    assert with_small_xi == pytest.approx(at_zero, rel=1e-5)


@pytest.mark.parametrize("alpha", [ALPHA_VAR, ALPHA_ES])
def test_caviar_on_iid_normal(alpha):
    returns = np.random.default_rng(7).normal(0.0, 0.02, 40_000)
    forecast = est.caviar(returns, alpha)
    assert forecast.es is None
    assert_close(forecast, normal_truth(0.0, 0.02, alpha)[0])


@pytest.mark.parametrize("alpha", [ALPHA_VAR, ALPHA_ES])
def test_caviar_recovers_its_own_process(alpha):
    """A return series whose quantile truly follows eq. 2.11 must give the true forecast."""
    rng = np.random.default_rng(8)
    z_alpha = stats.norm.ppf(alpha)
    beta = np.array([-0.03, 0.92, -0.05, -0.12]) * [abs(z_alpha) / 2.326, 1, abs(z_alpha) / 2.326, abs(z_alpha) / 2.326]
    n, q, returns = 40_000, -2.0, []
    for shock in rng.normal(size=n):
        r = q * shock / z_alpha
        returns.append(r)
        q = beta[0] + beta[1] * q + beta[2] * max(r, 0) + beta[3] * max(-r, 0)
    forecast = est.caviar(np.array(returns) / 100, alpha)
    assert forecast.var * 100 == pytest.approx(-q, rel=0.05)


def test_caviar_coverage_on_garch(garch_series):
    returns = garch_series[0][:6000]
    fit = est.fit_caviar(returns[:4000], ALPHA_ES)
    beta = np.array([fit["beta1"], fit["beta2"], fit["beta3"], fit["beta4"]])
    path = est._caviar_path(beta, 100 * returns, fit["q_start"])[:-1] / 100
    assert np.mean(returns[4000:] < path[4000:]) == pytest.approx(ALPHA_ES, abs=0.01)


@pytest.mark.parametrize("name", ["hs", "normal", "garch_t", "fhs", "evt_pot"])
def test_tail_grid_is_consistent_with_var_and_es(garch_series, name):
    returns = garch_series[0][:1000]
    spec = est.ESTIMATORS[name]
    forecast = spec.forecast(returns, spec.fit(returns, ALPHA_ES), ALPHA_ES)
    tail = forecast.details["tail_losses"]
    assert tail.mean() == pytest.approx(forecast.es, rel=1e-9)
    assert tail.min() >= forecast.var * (1 - 1e-9)
    assert forecast.es > forecast.var > 0


def test_every_public_estimator_follows_the_contract(garch_series):
    returns = garch_series[0][:1000]
    for name in est.ESTIMATORS:
        forecast = getattr(est, name)(returns, ALPHA_VAR)
        assert isinstance(forecast, est.Forecast)
        assert forecast.var > 0
        assert forecast.details
