import numpy as np
import pytest

from finstrategy.pricing.garch_montecarlo import fit_garch_t, risk_metrics, simulate_prices_at_expiration


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(7)
    # retornos com volatilidade variando no tempo, para o GARCH ter algo a capturar
    variances = 1e-4 * (1 + 0.5 * np.sin(np.linspace(0, 20, 1500)) ** 2)
    return rng.standard_t(5, size=1500) * np.sqrt(variances)


def test_fit_garch_t_produces_sane_parameters(synthetic_returns):
    garch = fit_garch_t(synthetic_returns)
    assert garch.next_day_variance > 0
    assert 0 <= garch.alpha < 0.3
    assert 0 <= garch.beta < 1
    assert garch.alpha + garch.beta < 1  # estacionariedade
    assert garch.degrees_of_freedom > 2
    assert garch.current_annual_volatility > 0


def test_simulate_prices_is_risk_neutral_by_construction(synthetic_returns):
    garch = fit_garch_t(synthetic_returns)
    rng = np.random.default_rng(42)
    annual_drift = 0.05
    trading_days = 30
    start_price = 100.0
    prices = simulate_prices_at_expiration(garch, start_price, trading_days, annual_drift, 5_000, rng)
    expected_mean = start_price * np.exp(annual_drift * trading_days / 252)
    # a centralização no código força E[preço simulado] = preço inicial * e^(drift*T)
    # exatamente sobre a amostra gerada, não só no limite assintótico
    assert prices.mean() == pytest.approx(expected_mean, rel=1e-9)
    assert prices.min() > 0


def test_simulate_prices_zero_drift_centers_around_start_price(synthetic_returns):
    garch = fit_garch_t(synthetic_returns)
    rng = np.random.default_rng(1)
    prices = simulate_prices_at_expiration(garch, 50.0, 20, 0.0, 5_000, rng)
    assert prices.mean() == pytest.approx(50.0, rel=1e-9)


def test_risk_metrics_basic_case():
    # 10% dos cenários perdem tudo (profit = -1 * cost), 90% ganham 10% do custo
    cost_per_share = 10.0
    profits = np.concatenate([np.full(100, -cost_per_share), np.full(900, cost_per_share * 0.1)])
    metrics = risk_metrics(profits, cost_per_share)
    assert metrics["P(perda)"] == pytest.approx(0.1)
    # ES 97,5% olha só a cauda de perda (profit < 0 vira loss_pct > 0); aqui a cauda
    # inteira (2,5% piores) está dentro dos 10% que perderam tudo
    assert metrics["ES 97,5%"] == pytest.approx(1.0, abs=1e-6)


def test_fit_garch_t_requires_nonconstant_returns_but_does_not_crash_on_near_constant():
    # retorno quase constante é um caso degenerado; só garante que não lança exceção
    rng = np.random.default_rng(3)
    tiny_noise = rng.normal(0, 1e-6, 300)
    garch = fit_garch_t(tiny_noise)
    assert np.isfinite(garch.next_day_variance)
