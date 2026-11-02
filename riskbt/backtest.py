"""Backtests of VaR and ES forecasts (PDF, section 2.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special, stats

from riskbt import config
from riskbt.regimes import CALM, CRISIS


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float


def _bernoulli_log_likelihood(count: float, total: float, probability: float) -> float:
    """ln[(1 - p)^(total - count) * p^count], with 0 * ln(0) = 0."""
    return float(special.xlogy(total - count, 1 - probability) + special.xlogy(count, probability))


def kupiec(hits: np.ndarray, alpha: float) -> TestResult:
    """Unconditional coverage LR test: H0 is P(hit) = alpha, chi2(1) (eq. 2.14; Kupiec, 1995)."""
    total, count = len(hits), int(np.sum(hits))
    if total == 0:
        return TestResult(np.nan, np.nan)
    restricted = _bernoulli_log_likelihood(count, total, alpha)
    unrestricted = _bernoulli_log_likelihood(count, total, count / total)
    statistic = max(-2 * (restricted - unrestricted), 0.0)
    return TestResult(statistic, float(stats.chi2.sf(statistic, 1)))


def _transition_counts(hits: np.ndarray, consecutive: np.ndarray | None) -> tuple[int, int, int, int]:
    """Counts n00, n01, n10 and n11 over pairs of days allowed by `consecutive`."""
    previous, current = hits[:-1], hits[1:]
    if consecutive is not None:
        previous, current = previous[consecutive], current[consecutive]
    n11 = int(np.sum(previous & current))
    n10 = int(np.sum(previous & ~current))
    n01 = int(np.sum(~previous & current))
    n00 = int(np.sum(~previous & ~current))
    return n00, n01, n10, n11


def christoffersen_independence(hits: np.ndarray, consecutive: np.ndarray | None = None) -> TestResult:
    """First-order Markov independence LR test, chi2(1) (eq. 2.15; Christoffersen, 1998).

    `consecutive[i]` says whether days i and i+1 are neighbours in the sample being tested;
    a regime subset has jumps in calendar time that are not real transitions.
    """
    hits = np.asarray(hits, dtype=bool)
    if len(hits) < 2:
        return TestResult(np.nan, np.nan)
    n00, n01, n10, n11 = _transition_counts(hits, consecutive)
    pairs = n00 + n01 + n10 + n11
    if pairs == 0:
        return TestResult(np.nan, np.nan)
    pi = (n01 + n11) / pairs
    restricted = _bernoulli_log_likelihood(n01 + n11, pairs, pi)
    from_calm = _bernoulli_log_likelihood(n01, n00 + n01, n01 / (n00 + n01)) if n00 + n01 else 0.0
    from_hit = _bernoulli_log_likelihood(n11, n10 + n11, n11 / (n10 + n11)) if n10 + n11 else 0.0
    statistic = max(-2 * (restricted - from_calm - from_hit), 0.0)
    return TestResult(statistic, float(stats.chi2.sf(statistic, 1)))


def christoffersen(hits: np.ndarray, alpha: float, consecutive: np.ndarray | None = None) -> TestResult:
    """Conditional coverage: LR_cc = LR_uc + LR_ind, chi2(2) (Christoffersen, 1998)."""
    coverage = kupiec(hits, alpha)
    independence = christoffersen_independence(hits, consecutive)
    statistic = coverage.statistic + independence.statistic
    return TestResult(statistic, float(stats.chi2.sf(statistic, 2)))


def traffic_light(hits: pd.Series, window: int = config.TRAFFIC_LIGHT_WINDOW) -> pd.DataFrame:
    """Basel zones over rolling windows: green 0-4, yellow 5-9, red 10+ (BCBS, 1996)."""
    counts = hits.astype(float).rolling(window).sum().dropna()
    zones = np.select(
        [counts <= config.TRAFFIC_LIGHT_GREEN_MAX, counts <= config.TRAFFIC_LIGHT_YELLOW_MAX],
        ["green", "yellow"], default="red",
    )
    return pd.DataFrame({"violations": counts.astype(int), "zone": zones}, index=counts.index)


def z2_statistic(returns: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Z2 = sum_t r_t * I_t / (T * alpha * ES_t) + 1 (eq. 2.16; Acerbi and Szekely, 2014)."""
    hits = returns < -var
    return float(np.sum(returns[hits] / es[hits]) / (len(returns) * alpha) + 1)


def z2_test(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    tail_losses: np.ndarray,
    alpha: float,
    replications: int = config.Z2_REPLICATIONS,
    seed: int = config.Z2_SEED,
) -> TestResult:
    """Left-tailed Z2 test, with the p-value simulated from the forecast distributions.

    Each replication asks the model what could have happened: a day violates VaR with
    probability alpha, and its loss is then drawn from that day's predicted tail (`tail_losses`,
    days x points). Low p-values mean ES was underestimated.
    """
    days = len(returns)
    if days == 0:
        return TestResult(np.nan, np.nan)
    rng = np.random.default_rng(seed)
    observed = z2_statistic(returns, var, es, alpha)
    simulated_hits = rng.random((replications, days)) < alpha
    points = rng.integers(tail_losses.shape[1], size=(replications, days))
    relative_losses = tail_losses[np.arange(days), points] / es
    simulated = 1 - np.sum(simulated_hits * relative_losses, axis=1) / (days * alpha)
    p_value = (1 + np.sum(simulated <= observed)) / (replications + 1)
    return TestResult(observed, float(p_value))


def fz0_loss(returns: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """FZ0 joint loss for (VaR, ES) (eq. 2.17; Patton, Ziegel and Chen, 2019). Lower is better."""
    v, e = -var, -es
    return -((returns <= v) * (v - returns)) / (alpha * e) + v / e + np.log(-e) - 1


def quantile_loss(returns: np.ndarray, var: np.ndarray, alpha: float) -> np.ndarray:
    """Quantile (pinball) loss of the VaR forecast (eq. 2.12)."""
    errors = returns + var
    return (alpha - (errors < 0)) * errors


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int | None = None) -> TestResult:
    """DM test of equal mean loss with Newey-West variance, N(0, 1) (Diebold and Mariano, 1995).

    Negative statistics favour A. The default lag count is floor(4 * (T / 100)^(2/9)).
    """
    difference = np.asarray(loss_a) - np.asarray(loss_b)
    days = len(difference)
    if days < 2:
        return TestResult(np.nan, np.nan)
    if np.allclose(difference, difference[0]):
        return TestResult(0.0, 1.0) if np.isclose(difference[0], 0) else TestResult(np.nan, np.nan)
    lags = int(np.floor(4 * (days / 100) ** (2 / 9))) if lags is None else lags
    fit = sm.OLS(difference, np.ones(days)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}, use_t=False,
    )
    return TestResult(float(fit.tvalues[0]), float(fit.pvalues[0]))


def add_losses(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Add the fz0_loss and quantile_loss columns; CAViaR has no ES, so its FZ0 is missing."""
    fz0 = np.full(len(forecasts), np.nan)
    has_es = forecasts["es"].notna().to_numpy()
    alpha, realized = forecasts["alpha"].to_numpy(), forecasts["return"].to_numpy()
    var, es = forecasts["var"].to_numpy(), forecasts["es"].to_numpy()
    fz0[has_es] = fz0_loss(realized[has_es], var[has_es], es[has_es], alpha[has_es])
    return forecasts.assign(fz0_loss=fz0, quantile_loss=quantile_loss(realized, var, alpha))


def _row(estimator, alpha, regime, test, result, n_obs, violations=np.nan, comparator="", detail="") -> dict:
    return {
        "estimator": estimator, "alpha": alpha, "regime": regime, "test": test,
        "statistic": result.statistic, "p_value": result.p_value, "n_obs": n_obs,
        "violations": violations, "comparator": comparator, "detail": detail,
    }


def _single_estimator_rows(group: pd.DataFrame, tail: np.ndarray | None, regime: str, positions: np.ndarray) -> list[dict]:
    """Coverage, independence, Z2 and mean losses of one estimator and level on one regime."""
    name, alpha = group["estimator"].iloc[0], group["alpha"].iloc[0]
    subset = group.iloc[positions]
    hits, n_obs = subset["hit"].to_numpy(), len(subset)
    count = int(hits.sum())
    consecutive = np.diff(positions) == 1
    rows = [
        _row(name, alpha, regime, "kupiec", kupiec(hits, alpha), n_obs, count),
        _row(name, alpha, regime, "christoffersen_independence", christoffersen_independence(hits, consecutive), n_obs, count),
        _row(name, alpha, regime, "christoffersen", christoffersen(hits, alpha, consecutive), n_obs, count),
        _row(name, alpha, regime, "mean_quantile_loss", TestResult(float(subset["quantile_loss"].mean()), np.nan), n_obs, count),
    ]
    if tail is not None:
        realized, var, es = (subset[column].to_numpy() for column in ("return", "var", "es"))
        rows.append(_row(name, alpha, regime, "z2", z2_test(realized, var, es, tail[positions], alpha), n_obs, count))
        rows.append(_row(name, alpha, regime, "mean_fz0_loss", TestResult(float(subset["fz0_loss"].mean()), np.nan), n_obs, count))
    return rows


def _pairwise_rows(frame: pd.DataFrame, alpha: float, regime: str, loss: str, test: str) -> list[dict]:
    """Diebold-Mariano between each pair of estimators that have this loss."""
    wide = frame.pivot(index="date", columns="estimator", values=loss).dropna(axis=1, how="all")
    names = list(wide.columns)
    return [
        _row(first, alpha, regime, test, diebold_mariano(wide[first].to_numpy(), wide[second].to_numpy()), len(wide), comparator=second)
        for position, first in enumerate(names) for second in names[position + 1 :]
    ]


def _traffic_light_rows(frame: pd.DataFrame) -> list[dict]:
    """Zone of each estimator's VaR 99% in the most recent 250 days."""
    rows = []
    for name, group in frame[frame["alpha"] == config.ALPHA_VAR].groupby("estimator", sort=False):
        lights = traffic_light(group.set_index("date")["hit"])
        if not lights.empty:
            last = lights.iloc[-1]
            result = TestResult(float(last["violations"]), np.nan)
            rows.append(_row(name, config.ALPHA_VAR, "all", "traffic_light", result, config.TRAFFIC_LIGHT_WINDOW, int(last["violations"]), detail=last["zone"]))
    return rows


def evaluate(forecasts: pd.DataFrame, tail_losses: dict, regimes: pd.Series) -> pd.DataFrame:
    """Run every test on the full sample and on calm and crisis days, in long format."""
    forecasts = forecasts.assign(regime=regimes.reindex(forecasts["date"]).to_numpy())
    rows: list[dict] = []
    for (name, alpha), group in forecasts.groupby(["estimator", "alpha"], sort=False):
        labels = group["regime"].to_numpy()
        for regime in ("all", CALM, CRISIS):
            positions = np.arange(len(group)) if regime == "all" else np.flatnonzero(labels == regime)
            rows += _single_estimator_rows(group, tail_losses.get((name, alpha)), regime, positions)
    for alpha, level in forecasts.groupby("alpha", sort=False):
        for regime in ("all", CALM, CRISIS):
            subset = level if regime == "all" else level[level["regime"] == regime]
            rows += _pairwise_rows(subset, alpha, regime, "fz0_loss", "diebold_mariano_fz0")
            rows += _pairwise_rows(subset, alpha, regime, "quantile_loss", "diebold_mariano_quantile")
    return pd.DataFrame(rows + _traffic_light_rows(forecasts))
