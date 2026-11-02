"""Backtests: formulas on small cases, and rejection rate of 3-7% at the 5% level under H0."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskbt import backtest as bt

MIN_RATE, MAX_RATE = 0.03, 0.07


def test_kupiec_known_value():
    hits = np.r_[np.ones(20, dtype=bool), np.zeros(980, dtype=bool)]
    pi = 0.02
    expected = -2 * (980 * np.log(0.99) + 20 * np.log(0.01) - 980 * np.log(1 - pi) - 20 * np.log(pi))
    result = bt.kupiec(hits, 0.01)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(stats.chi2.sf(expected, 1))


def test_kupiec_with_no_violations():
    result = bt.kupiec(np.zeros(500, dtype=bool), 0.01)
    assert result.statistic == pytest.approx(-2 * 500 * np.log(0.99))


def test_kupiec_empty_sample():
    assert np.isnan(bt.kupiec(np.array([], dtype=bool), 0.01).p_value)


def test_christoffersen_detects_clusters():
    clustered = np.zeros(1000, dtype=bool)
    clustered[100:110] = clustered[500:510] = True
    spread = np.zeros(1000, dtype=bool)
    spread[::50] = True
    assert bt.christoffersen_independence(clustered).p_value < 0.001
    assert bt.christoffersen_independence(spread).p_value > 0.05


def test_christoffersen_known_transitions():
    hits = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    assert bt._transition_counts(hits, None) == (4, 2, 2, 1)


def test_christoffersen_ignores_jumps_between_regime_blocks():
    hits = np.array([0, 1, 1, 0], dtype=bool)
    consecutive = np.array([True, False, True])
    assert bt._transition_counts(hits, consecutive) == (0, 1, 1, 0)


def test_conditional_coverage_adds_both_statistics():
    hits = np.random.default_rng(1).random(1000) < 0.02
    total = bt.christoffersen(hits, 0.01).statistic
    parts = bt.kupiec(hits, 0.01).statistic + bt.christoffersen_independence(hits).statistic
    assert total == pytest.approx(parts)


def test_traffic_light_zones():
    hits = np.zeros(260, dtype=bool)
    hits[:4] = True
    series = pd.Series(hits, index=pd.bdate_range("2020-01-01", periods=260))
    assert bt.traffic_light(series)["zone"].iloc[0] == "green"
    hits[:9] = True
    assert bt.traffic_light(pd.Series(hits, index=series.index))["zone"].iloc[0] == "yellow"
    hits[:10] = True
    lights = bt.traffic_light(pd.Series(hits, index=series.index))
    assert lights["zone"].iloc[0] == "red" and lights["violations"].iloc[0] == 10
    assert lights["zone"].iloc[-1] == "green"


def test_z2_formula():
    returns = np.array([-0.05, 0.01, 0.01, 0.01])
    var, es = np.full(4, 0.02), np.full(4, 0.04)
    assert bt.z2_statistic(returns, var, es, 0.25) == pytest.approx(-0.05 / 0.04 / (4 * 0.25) + 1)


def test_z2_has_zero_mean_under_h0():
    rng = np.random.default_rng(2)
    alpha, sigma, days = 0.025, 0.01, 4_000_000  # standard error of about 0.003
    z = stats.norm.ppf(alpha)
    returns = rng.normal(0, sigma, days)
    var, es = np.full(days, -z * sigma), np.full(days, sigma * stats.norm.pdf(z) / alpha)
    assert bt.z2_statistic(returns, var, es, alpha) == pytest.approx(0, abs=0.012)


def normal_tail(sigma, alpha, days):
    points = (np.arange(50) + 0.5) / 50
    return np.tile(-stats.norm.ppf(alpha * points) * sigma, (days, 1))


def test_z2_test_detects_underestimated_es():
    rng = np.random.default_rng(3)
    alpha, days, sigma = 0.025, 1000, 0.01
    z = stats.norm.ppf(alpha)
    var, es = np.full(days, -z * sigma), np.full(days, sigma * stats.norm.pdf(z) / alpha)
    assert bt.z2_test(rng.normal(0, 2 * sigma, days), var, es, normal_tail(sigma, alpha, days), alpha).p_value < 0.01


def test_z2_seed_gives_identical_p_values():
    rng = np.random.default_rng(4)
    alpha, days, sigma = 0.025, 300, 0.01
    z = stats.norm.ppf(alpha)
    var, es = np.full(days, -z * sigma), np.full(days, sigma * stats.norm.pdf(z) / alpha)
    returns, tail = rng.normal(0, sigma, days), normal_tail(sigma, alpha, days)
    assert bt.z2_test(returns, var, es, tail, alpha) == bt.z2_test(returns, var, es, tail, alpha)


def test_fz0_formula_and_propriety():
    returns, var, es, alpha = np.array([-0.05]), np.array([0.02]), np.array([0.03]), 0.025
    v, e = -0.02, -0.03
    expected = -1 / (alpha * e) * (v + 0.05) + v / e + np.log(-e) - 1
    assert bt.fz0_loss(returns, var, es, alpha)[0] == pytest.approx(expected)
    sample = np.random.default_rng(5).normal(0, 0.01, 500_000)
    alpha_z = stats.norm.ppf(alpha)
    true_var, true_es = -alpha_z * 0.01, 0.01 * stats.norm.pdf(alpha_z) / alpha
    mean_loss = lambda scale: bt.fz0_loss(sample, np.full(sample.size, true_var * scale), np.full(sample.size, true_es * scale), alpha).mean()
    assert mean_loss(1.0) < mean_loss(1.3)
    assert mean_loss(1.0) < mean_loss(0.7)


def test_quantile_loss_is_minimized_at_the_true_quantile():
    sample = np.random.default_rng(6).normal(0, 0.01, 500_000)
    true_var = -stats.norm.ppf(0.01) * 0.01
    mean_loss = lambda var: bt.quantile_loss(sample, np.full(sample.size, var), 0.01).mean()
    assert mean_loss(true_var) < mean_loss(true_var * 1.2)
    assert mean_loss(true_var) < mean_loss(true_var * 0.8)


def test_dm_matches_hand_formula_without_lags():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    d = a - b
    expected = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / 500)
    assert bt.diebold_mariano(a, b, lags=0).statistic == pytest.approx(expected)


def test_dm_signs_and_degenerate_cases():
    rng = np.random.default_rng(8)
    good, bad = rng.normal(0, 1, 800), rng.normal(0.5, 1, 800)
    assert bt.diebold_mariano(good, bad).statistic < -2
    assert bt.diebold_mariano(good, good).p_value == 1.0
    assert np.isnan(bt.diebold_mariano(good[:1], bad[:1]).p_value)


def rejection_rate(p_values):
    return np.mean(np.asarray(p_values) < 0.05)


def test_kupiec_size_under_h0():
    rng = np.random.default_rng(10)
    p_values = [bt.kupiec(rng.random(1500) < 0.025, 0.025).p_value for _ in range(3000)]
    assert MIN_RATE <= rejection_rate(p_values) <= MAX_RATE


def test_christoffersen_size_under_h0():
    rng = np.random.default_rng(11)
    p_values = [bt.christoffersen(rng.random(1500) < 0.05, 0.05).p_value for _ in range(3000)]
    assert MIN_RATE <= rejection_rate(p_values) <= MAX_RATE


def test_z2_size_under_h0_normal():
    rng = np.random.default_rng(12)
    alpha, days, sigma = 0.025, 750, 0.01
    z = stats.norm.ppf(alpha)
    var, es = np.full(days, -z * sigma), np.full(days, sigma * stats.norm.pdf(z) / alpha)
    tail = normal_tail(sigma, alpha, days)
    p_values = [bt.z2_test(rng.normal(0, sigma, days), var, es, tail, alpha, seed=s).p_value for s in range(800)]
    assert MIN_RATE <= rejection_rate(p_values) <= MAX_RATE


def test_dm_size_under_h0():
    rng = np.random.default_rng(13)
    p_values = [bt.diebold_mariano(rng.normal(size=750), rng.normal(size=750)).p_value for _ in range(1500)]
    assert MIN_RATE <= rejection_rate(p_values) <= MAX_RATE


def test_evaluate_produces_every_test_by_regime():
    from riskbt.regimes import CALM, CRISIS
    from riskbt.rolling import rolling_forecasts

    values = np.random.default_rng(14).normal(0, 0.01, 420)
    returns = pd.Series(values, index=pd.bdate_range("2015-01-01", periods=420))
    result = rolling_forecasts(returns, names=("hs", "normal", "caviar"), window=300, refit_every=60)
    labels = np.where(np.arange(420) % 2 == 0, CALM, CRISIS)
    regimes = pd.Series(labels, index=returns.index)
    table = bt.evaluate(bt.add_losses(result.forecasts), result.tail_losses, regimes)
    assert set(table["regime"]) == {"all", CALM, CRISIS}
    assert {"kupiec", "christoffersen", "christoffersen_independence", "z2", "mean_fz0_loss", "diebold_mariano_fz0"} <= set(table["test"])
    assert table[(table.estimator == "caviar") & (table.test == "z2")].empty
    assert table[table.test == "mean_quantile_loss"]["estimator"].nunique() == 3
    assert table[(table.estimator == "hs") & (table.test == "kupiec") & (table.regime == CALM)]["n_obs"].iloc[0] == 60
