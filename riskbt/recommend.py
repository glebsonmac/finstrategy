"""Choose the estimator for a portfolio from its backtests in the current regime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskbt import config

PASSED = "passed"
CAVIAR_ONLY = "caviar_only"
NONE_PASSED = "none_passed"
MIN_EXPECTED_VIOLATIONS = 5.0  # below this the regime sample says little (PDF, section 2.4.6)


@dataclass(frozen=True)
class Recommendation:
    estimator: str
    status: str  # PASSED, CAVIAR_ONLY or NONE_PASSED
    regime: str
    table: pd.DataFrame  # one row per estimator with the evidence behind the choice
    low_power: bool  # too few days in this regime for the tests to detect much


def _evidence(tests: pd.DataFrame, regime: str, alpha: float) -> pd.DataFrame:
    """Pivot one regime and level into columns: kupiec_p, christoffersen_p, z2_p and losses."""
    subset = tests[(tests["regime"] == regime) & (tests["alpha"] == alpha) & (tests["comparator"] == "")]
    p_values = subset.pivot(index="estimator", columns="test", values="p_value")
    statistics = subset.pivot(index="estimator", columns="test", values="statistic")
    table = pd.DataFrame({
        "kupiec_p": p_values["kupiec"],
        "christoffersen_p": p_values["christoffersen"],
        "z2_p": p_values["z2"],
        "mean_fz0_loss": statistics["mean_fz0_loss"],
        "mean_quantile_loss": statistics["mean_quantile_loss"],
        "n_obs": subset[subset["test"] == "kupiec"].set_index("estimator")["n_obs"],
    })
    return table


def _with_verdicts(table: pd.DataFrame) -> pd.DataFrame:
    """Count failed tests. Missing p-values fail, except Z2 for CAViaR, which has no ES."""
    level = config.SIGNIFICANCE_LEVEL
    has_es = table["mean_fz0_loss"].notna()
    failed = (
        ~(table["kupiec_p"] >= level)
    ).astype(int) + (~(table["christoffersen_p"] >= level)).astype(int) + (
        (~(table["z2_p"] >= level)) & has_es
    ).astype(int)
    return table.assign(tests_failed=failed, passed=failed == 0)


def recommend(tests: pd.DataFrame, regime: str, alpha: float = config.ALPHA_ES) -> Recommendation:
    """Apply the rule of CLAUDE.md to the tests of the current regime at 97.5%.

    1. Keep estimators passing Kupiec and Christoffersen (p >= 0.05) and, with ES, also Z2.
    2. Pick the lowest mean FZ0 among them; CAViaR competes only if no other estimator passes.
    3. If none passes, name the one failing the fewest tests, breaking ties by FZ0.

    The 97.5% level is used because 99% has too few violations per regime (section 2.4.6).
    """
    table = _with_verdicts(_evidence(tests, regime, alpha))
    low_power = bool(np.nan_to_num(table["n_obs"].max()) * alpha < MIN_EXPECTED_VIOLATIONS)
    with_es = table[table["mean_fz0_loss"].notna()]
    passing = with_es[with_es["passed"]]
    if not passing.empty:
        return Recommendation(str(passing["mean_fz0_loss"].idxmin()), PASSED, regime, table, low_power)
    if "caviar" in table.index and table.loc["caviar", "passed"]:
        return Recommendation("caviar", CAVIAR_ONLY, regime, table, low_power)
    least_bad = with_es.sort_values(["tests_failed", "mean_fz0_loss"]).index[0]
    return Recommendation(str(least_bad), NONE_PASSED, regime, table, low_power)
