import numpy as np
import pandas as pd
import pytest

from riskbt.config import ALPHA_ES, ALPHA_VAR, TAIL_GRID_POINTS
from riskbt.estimators import ESTIMATORS
from riskbt.rolling import latest_forecasts, rolling_forecasts

WINDOW, REFIT = 300, 5


@pytest.fixture(scope="module")
def returns():
    values = np.random.default_rng(21).standard_t(5, WINDOW + 30) * 0.01
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=len(values)))


@pytest.fixture(scope="module")
def result(returns):
    return rolling_forecasts(returns, window=WINDOW, refit_every=REFIT)


def test_shape_and_alignment(result, returns):
    assert len(result.forecasts) == 30 * len(ESTIMATORS) * 2
    one = result.forecasts[(result.forecasts.estimator == "hs") & (result.forecasts.alpha == ALPHA_ES)]
    assert list(one["date"]) == list(returns.index[WINDOW:])
    np.testing.assert_allclose(one["return"], returns.iloc[WINDOW:])
    assert (one["hit"] == (one["return"] < -one["var"])).all()


def test_caviar_has_no_es_and_others_do(result):
    forecasts = result.forecasts
    assert forecasts[forecasts.estimator == "caviar"]["es"].isna().all()
    assert forecasts[forecasts.estimator != "caviar"]["es"].notna().all()
    assert ("caviar", ALPHA_ES) not in result.tail_losses
    assert result.tail_losses[("fhs", ALPHA_VAR)].shape == (30, TAIL_GRID_POINTS)


def test_refit_schedule_applies_to_every_estimator(result):
    counts = result.parameters.groupby(["estimator", "alpha"]).size()
    assert (counts == 6).all() and len(counts) == len(ESTIMATORS) * 2
    assert {"omega", "a", "b", "nu", "xi", "beta", "u", "converged"} <= set(result.parameters.columns)


def test_parameters_stay_fixed_but_volatility_moves_between_refits(result):
    forecasts = result.forecasts
    garch = forecasts[(forecasts.estimator == "garch_t") & (forecasts.alpha == ALPHA_ES)]["var"].to_numpy()
    hs = forecasts[(forecasts.estimator == "hs") & (forecasts.alpha == ALPHA_ES)]["var"].to_numpy()
    assert len(set(garch[:REFIT])) == REFIT
    assert len(set(np.round(hs[:REFIT], 12))) == 1


def test_no_look_ahead(returns, result):
    """Altering data after day t must not change any forecast up to t + 1."""
    cut = WINDOW + 17
    altered = returns.copy()
    altered.iloc[cut:] = altered.iloc[cut:] * 7 + 0.05
    other = rolling_forecasts(altered, window=WINDOW, refit_every=REFIT).forecasts
    keep = result.forecasts["date"] <= returns.index[cut]
    columns = ["estimator", "alpha", "date", "var", "es"]
    pd.testing.assert_frame_equal(result.forecasts.loc[keep, columns], other.loc[keep, columns])


def test_reproducible(returns, result):
    again = rolling_forecasts(returns, window=WINDOW, refit_every=REFIT)
    pd.testing.assert_frame_equal(result.forecasts, again.forecasts)
    pd.testing.assert_frame_equal(result.parameters, again.parameters)


def test_series_too_short_is_refused(returns):
    with pytest.raises(ValueError, match="Need more than"):
        rolling_forecasts(returns.iloc[:WINDOW], window=WINDOW)


def test_latest_forecast_uses_the_last_window(returns):
    latest = latest_forecasts(returns, window=WINDOW)
    assert set(latest) == {(name, alpha) for name in ESTIMATORS for alpha in (ALPHA_VAR, ALPHA_ES)}
    tail = returns.iloc[-WINDOW:].sort_values()
    assert latest[("hs", ALPHA_VAR)].var == pytest.approx(-tail.iloc[2])
