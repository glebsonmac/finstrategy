import numpy as np
import pandas as pd

from riskbt.config import ALPHA_ES
from riskbt.recommend import CAVIAR_ONLY, NONE_PASSED, PASSED, recommend

GOOD, BAD = 0.5, 0.001


def make_tests(rows, regime="crisis", n_obs=1000):
    """rows: name -> (kupiec p, christoffersen p, z2 p or None, fz0 loss or None)."""
    records = []
    for name, (kupiec, chris, z2, fz0) in rows.items():
        entries = [("kupiec", np.nan, kupiec), ("christoffersen", np.nan, chris), ("mean_quantile_loss", 0.01, np.nan)]
        if z2 is not None:
            entries += [("z2", -0.1, z2), ("mean_fz0_loss", fz0, np.nan)]
        for test, statistic, p_value in entries:
            records.append({"estimator": name, "alpha": ALPHA_ES, "regime": regime, "test": test, "statistic": statistic, "p_value": p_value, "n_obs": n_obs, "comparator": ""})
    records.append({"estimator": "hs", "alpha": ALPHA_ES, "regime": regime, "test": "diebold_mariano_fz0", "statistic": 1.0, "p_value": 0.1, "n_obs": n_obs, "comparator": "normal"})
    return pd.DataFrame(records)


def test_lowest_loss_among_those_that_pass():
    tests = make_tests({"hs": (BAD, GOOD, GOOD, -5.0), "fhs": (GOOD, GOOD, GOOD, -2.0), "evt_pot": (GOOD, GOOD, GOOD, -3.0), "caviar": (GOOD, GOOD, None, None)})
    result = recommend(tests, "crisis")
    assert result.estimator == "evt_pot" and result.status == PASSED


def test_z2_failure_disqualifies_an_estimator_with_es():
    tests = make_tests({"fhs": (GOOD, GOOD, BAD, -9.0), "evt_pot": (GOOD, GOOD, GOOD, -1.0)})
    assert recommend(tests, "crisis").estimator == "evt_pot"


def test_caviar_only_competes_when_no_other_passes():
    tests = make_tests({"fhs": (BAD, GOOD, GOOD, -2.0), "caviar": (GOOD, GOOD, None, None)})
    result = recommend(tests, "crisis")
    assert result.estimator == "caviar" and result.status == CAVIAR_ONLY


def test_none_passed_names_the_least_bad():
    tests = make_tests({
        "hs": (BAD, BAD, BAD, -9.0), "fhs": (BAD, GOOD, GOOD, -1.0), "evt_pot": (GOOD, BAD, GOOD, -2.0), "caviar": (BAD, BAD, None, None),
    })
    result = recommend(tests, "crisis")
    assert result.estimator == "evt_pot" and result.status == NONE_PASSED


def test_only_the_current_regime_counts():
    tests = pd.concat([
        make_tests({"fhs": (GOOD, GOOD, GOOD, -1.0), "hs": (BAD, BAD, BAD, -9.0)}, regime="calm"),
        make_tests({"fhs": (BAD, BAD, BAD, -1.0), "hs": (GOOD, GOOD, GOOD, -2.0)}, regime="crisis"),
    ])
    assert recommend(tests, "calm").estimator == "fhs"
    assert recommend(tests, "crisis").estimator == "hs"


def test_low_power_flag():
    rows = {"fhs": (GOOD, GOOD, GOOD, -1.0)}
    assert recommend(make_tests(rows, n_obs=100), "crisis").low_power
    assert not recommend(make_tests(rows, n_obs=1000), "crisis").low_power
