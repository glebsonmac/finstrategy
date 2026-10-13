import numpy as np
import pytest

from finstrategy.pricing.garch_montecarlo import fit_garch_t, simulate_price_paths


@pytest.fixture
def garch():
    rng = np.random.default_rng(9)
    variances = 1e-4 * (1 + 0.5 * np.sin(np.linspace(0, 20, 1200)) ** 2)
    returns = rng.standard_t(5, size=1200) * np.sqrt(variances)
    return fit_garch_t(returns)


def test_simulate_price_paths_shape_and_start_column(garch):
    rng = np.random.default_rng(1)
    paths = simulate_price_paths(garch, 100.0, trading_days=15, annual_drift=0.0, number_of_paths=500, rng=rng)
    assert paths.shape == (500, 16)
    assert np.allclose(paths[:, 0], 100.0)
    assert (paths > 0).all()


def test_simulate_price_paths_terminal_mean_matches_drift(garch):
    rng = np.random.default_rng(2)
    annual_drift = 0.08
    trading_days = 20
    start_price = 50.0
    paths = simulate_price_paths(garch, start_price, trading_days, annual_drift, 4_000, rng)
    expected_mean = start_price * np.exp(annual_drift * trading_days / 252)
    assert paths[:, -1].mean() == pytest.approx(expected_mean, rel=1e-9)
