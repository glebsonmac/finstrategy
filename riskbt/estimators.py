"""The six VaR and ES estimators (PDF, section 2.3). VaR and ES are positive losses.

Each estimator has a public function `name(returns, alpha) -> Forecast`, built from two parts:
`fit_*` estimates the parameters and `forecast_*` applies them to the latest data. The rolling
window uses the parts separately: it refits every 20 days and forecasts every day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from arch import arch_model
from scipy import optimize, signal, stats

from riskbt import config

PERCENT = 100.0  # arch and CAViaR optimize better with returns in percent


@dataclass(frozen=True)
class Forecast:
    var: float  # positive loss
    es: float | None  # None for CAViaR
    details: dict  # intermediate values used by the explanation


@dataclass(frozen=True)
class EstimatorSpec:
    fit: Callable[..., dict]
    forecast: Callable[[np.ndarray, dict, float], Forecast]
    uses_garch: bool


def _tail_count(n: int, alpha: float) -> int:
    """k = floor(n * alpha), at least one observation. The small term absorbs float error."""
    return max(int(np.floor(n * alpha + 1e-9)), 1)


def _tail_levels() -> np.ndarray:
    """Midpoints of equal slices of the tail, as fractions of alpha."""
    points = config.TAIL_GRID_POINTS
    return (np.arange(points) + 0.5) / points


def _matched_tail(values: np.ndarray, quantile: float, tail_mean: float) -> np.ndarray:
    """Rescale tail points beyond `quantile` so their mean is exactly `tail_mean`."""
    excess = values - quantile
    return quantile + excess * (tail_mean - quantile) / excess.mean()


def _location_scale_forecast(
    mu: float, sigma: float, quantile: float, tail_mean: float, tail: np.ndarray, details: dict,
) -> Forecast:
    """VaR = -(mu + sigma * q) and ES = -(mu + sigma * e) (section 2.2)."""
    var = -(mu + sigma * quantile)
    es = -(mu + sigma * tail_mean)
    tail_losses = -(mu + sigma * tail)
    return Forecast(float(var), float(es), {**details, "tail_losses": tail_losses})


def _empirical_tail(sample: np.ndarray, alpha: float) -> tuple[int, float, float, np.ndarray]:
    """Return k, the k-th smallest value, the mean of the k smallest and the tail grid."""
    k = _tail_count(len(sample), alpha)
    worst = np.partition(sample, k - 1)[:k]
    quantile, tail_mean = float(worst.max()), float(worst.mean())
    if k == 1 or worst.min() == quantile:
        return k, quantile, tail_mean, np.full(config.TAIL_GRID_POINTS, quantile)
    grid = np.quantile(worst, _tail_levels())
    return k, quantile, tail_mean, _matched_tail(grid, quantile, tail_mean)


# --- Historical simulation -----------------------------------------------------------------


def fit_hs(returns: np.ndarray, alpha: float, garch: dict | None = None) -> dict:
    k, quantile, tail_mean, tail = _empirical_tail(returns, alpha)
    return {"n": len(returns), "k": k, "quantile": quantile, "tail_mean": tail_mean, "tail": tail}


def forecast_hs(returns: np.ndarray, fit: dict, alpha: float) -> Forecast:
    details = {key: fit[key] for key in ("n", "k", "quantile", "tail_mean")}
    return _location_scale_forecast(0.0, 1.0, fit["quantile"], fit["tail_mean"], fit["tail"], details)


def hs(returns: np.ndarray, alpha: float) -> Forecast:
    """Historical simulation: the empirical tail of the window (eq. 2.4).

    VaR = -r_(k) and ES = -(1/k) * sum_{i<=k} r_(i), with k = floor(n * alpha).
    Reference: McNeil, Frey and Embrechts (2015).
    """
    return forecast_hs(returns, fit_hs(returns, alpha), alpha)


# --- Normal --------------------------------------------------------------------------------


def fit_normal(returns: np.ndarray, alpha: float, garch: dict | None = None) -> dict:
    z_alpha = float(stats.norm.ppf(alpha))
    tail_mean = float(-stats.norm.pdf(z_alpha) / alpha)
    grid = stats.norm.ppf(alpha * _tail_levels())
    return {
        "mu": float(returns.mean()), "sigma": float(returns.std(ddof=1)), "z_alpha": z_alpha,
        "tail_mean": tail_mean, "tail": _matched_tail(grid, z_alpha, tail_mean),
    }


def forecast_normal(returns: np.ndarray, fit: dict, alpha: float) -> Forecast:
    details = {key: fit[key] for key in ("mu", "sigma", "z_alpha", "tail_mean")}
    return _location_scale_forecast(fit["mu"], fit["sigma"], fit["z_alpha"], fit["tail_mean"], fit["tail"], details)


def normal(returns: np.ndarray, alpha: float) -> Forecast:
    """Normal baseline with the window's mean and standard deviation (eq. 2.5).

    VaR = -(mu + sigma * z_alpha) and ES = -mu + sigma * phi(z_alpha) / alpha.
    Reference: McNeil, Frey and Embrechts (2015).
    """
    return forecast_normal(returns, fit_normal(returns, alpha), alpha)


# --- GARCH(1,1) filter, shared by GARCH-t, FHS and EVT-POT -----------------------------------


def fit_garch(returns: np.ndarray) -> dict:
    """Fit a constant-mean GARCH(1,1) with Student t innovations by maximum likelihood (arch)."""
    model = arch_model(PERCENT * returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    result = model.fit(disp="off", show_warning=False)
    params = result.params
    return {
        "mu": float(params["mu"] / PERCENT),
        "omega": float(params["omega"] / PERCENT**2),
        "a": float(params["alpha[1]"]),
        "b": float(params["beta[1]"]),
        "nu": float(params["nu"]),
        "converged": bool(result.convergence_flag == 0),
    }


def garch_volatility(returns: np.ndarray, garch: dict) -> np.ndarray:
    """Conditional volatility from eq. 2.6; the last of the n + 1 values is the t+1 forecast.

    The recursion starts at the window's sample variance.
    """
    shocks = returns - garch["mu"]
    initial = float(returns.var())
    driver = garch["omega"] + garch["a"] * shocks**2
    variance, _ = signal.lfilter([1.0], [1.0, -garch["b"]], driver, zi=[garch["b"] * initial])
    return np.sqrt(np.concatenate(([initial], variance)))


def _garch_details(returns: np.ndarray, garch: dict) -> tuple[float, dict]:
    """Next-day volatility and the values every GARCH-based explanation shows (eq. 2.7)."""
    sigma_next = float(garch_volatility(returns, garch)[-1])
    persistence = garch["a"] + garch["b"]
    half_life = float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else float("inf")
    long_run = float(np.sqrt(garch["omega"] / (1 - persistence))) if persistence < 1 else float("inf")
    details = {
        "mu": garch["mu"], "omega": garch["omega"], "a": garch["a"], "b": garch["b"],
        "nu": garch["nu"], "sigma_next": sigma_next, "persistence": persistence,
        "half_life": half_life, "long_run_volatility": long_run,
    }
    return sigma_next, details


def standardized_residuals(returns: np.ndarray, garch: dict) -> np.ndarray:
    """z_i = (r_i - mu) / sigma_i (section 2.3.4)."""
    return (returns - garch["mu"]) / garch_volatility(returns, garch)[:-1]


# --- GARCH-t -------------------------------------------------------------------------------


def student_t_tail(nu: float, alpha: float) -> tuple[float, float]:
    """Quantile and tail mean of the unit-variance Student t, the terms of eq. 2.8."""
    scale = np.sqrt((nu - 2) / nu)
    t_alpha = stats.t.ppf(alpha, nu)
    density = stats.t.pdf(t_alpha, nu)
    return float(scale * t_alpha), float(-scale * density / alpha * (nu + t_alpha**2) / (nu - 1))


def fit_garch_t(returns: np.ndarray, alpha: float, garch: dict | None = None) -> dict:
    garch = fit_garch(returns) if garch is None else garch
    quantile, tail_mean = student_t_tail(garch["nu"], alpha)
    scale = np.sqrt((garch["nu"] - 2) / garch["nu"])
    grid = scale * stats.t.ppf(alpha * _tail_levels(), garch["nu"])
    return {**garch, "quantile": quantile, "tail_mean": tail_mean, "tail": _matched_tail(grid, quantile, tail_mean)}


def forecast_garch_t(returns: np.ndarray, fit: dict, alpha: float) -> Forecast:
    sigma_next, details = _garch_details(returns, fit)
    details |= {"quantile": fit["quantile"], "tail_mean": fit["tail_mean"]}
    return _location_scale_forecast(fit["mu"], sigma_next, fit["quantile"], fit["tail_mean"], fit["tail"], details)


def garch_t(returns: np.ndarray, alpha: float) -> Forecast:
    """GARCH(1,1) with standardized Student t innovations (eq. 2.6 to 2.8).

    VaR = -(mu + sigma_t+1 * s_nu * t_nu,alpha) and
    ES = -mu + sigma_t+1 * s_nu * g_nu(t_nu,alpha) / alpha * (nu + t_nu,alpha^2) / (nu - 1).
    References: Bollerslev (1986, 1987).
    """
    return forecast_garch_t(returns, fit_garch_t(returns, alpha), alpha)


# --- Filtered historical simulation ----------------------------------------------------------


def fit_fhs(returns: np.ndarray, alpha: float, garch: dict | None = None) -> dict:
    garch = fit_garch(returns) if garch is None else garch
    k, quantile, tail_mean, tail = _empirical_tail(standardized_residuals(returns, garch), alpha)
    return {**garch, "n": len(returns), "k": k, "quantile": quantile, "tail_mean": tail_mean, "tail": tail}


def forecast_fhs(returns: np.ndarray, fit: dict, alpha: float) -> Forecast:
    sigma_next, details = _garch_details(returns, fit)
    details |= {key: fit[key] for key in ("n", "k", "quantile", "tail_mean")}
    return _location_scale_forecast(fit["mu"], sigma_next, fit["quantile"], fit["tail_mean"], fit["tail"], details)


def fhs(returns: np.ndarray, alpha: float) -> Forecast:
    """Filtered historical simulation: GARCH volatility and empirical residual tail (eq. 2.9).

    VaR = -(mu + sigma_t+1 * z_(k)) and ES = -(mu + sigma_t+1 * (1/k) * sum_{i<=k} z_(i)).
    Reference: Barone-Adesi, Giannopoulos and Vosper (1999).
    """
    return forecast_fhs(returns, fit_fhs(returns, alpha), alpha)


# --- EVT peaks over threshold ----------------------------------------------------------------


def gpd_tail_measures(
    threshold: float, xi: float, beta: float, tail_share: float, alpha: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Quantile and ES of standardized losses from eq. 2.10; `tail_share` is N_u / n."""
    ratio = np.asarray(alpha) / tail_share
    if abs(xi) < 1e-9:
        quantile = threshold - beta * np.log(ratio)
    else:
        quantile = threshold + beta / xi * (ratio ** (-xi) - 1)
    return quantile, quantile / (1 - xi) + (beta - xi * threshold) / (1 - xi)


def fit_evt_pot(returns: np.ndarray, alpha: float, garch: dict | None = None) -> dict:
    garch = fit_garch(returns) if garch is None else garch
    losses = -standardized_residuals(returns, garch)
    threshold = float(np.quantile(losses, config.EVT_THRESHOLD_QUANTILE))
    excesses = losses[losses > threshold] - threshold
    xi, _, beta = stats.genpareto.fit(excesses, floc=0)
    xi = min(float(xi), 0.95)  # eq. 2.10 requires xi < 1
    tail_share = len(excesses) / len(losses)
    y_alpha, e_y = gpd_tail_measures(threshold, xi, float(beta), tail_share, alpha)
    grid, _ = gpd_tail_measures(threshold, xi, float(beta), tail_share, alpha * _tail_levels())
    return {
        **garch, "u": threshold, "xi": xi, "beta": float(beta), "n": len(losses),
        "n_u": len(excesses), "y_alpha": float(y_alpha), "e_y": float(e_y),
        "tail": -_matched_tail(grid, float(y_alpha), float(e_y)),
    }


def forecast_evt_pot(returns: np.ndarray, fit: dict, alpha: float) -> Forecast:
    sigma_next, details = _garch_details(returns, fit)
    details |= {key: fit[key] for key in ("u", "xi", "beta", "n", "n_u", "y_alpha", "e_y")}
    return _location_scale_forecast(fit["mu"], sigma_next, -fit["y_alpha"], -fit["e_y"], fit["tail"], details)


def evt_pot(returns: np.ndarray, alpha: float) -> Forecast:
    """GPD fitted to GARCH-standardized losses above the 90% quantile (eq. 2.10).

    y_alpha = u + beta / xi * ((n / N_u * alpha)^(-xi) - 1) and
    e_y = y_alpha / (1 - xi) + (beta - xi * u) / (1 - xi); VaR = -mu + sigma_t+1 * y_alpha.
    Reference: McNeil and Frey (2000).
    """
    return forecast_evt_pot(returns, fit_evt_pot(returns, alpha), alpha)


# --- CAViaR --------------------------------------------------------------------------------


def _caviar_path(beta: np.ndarray, returns_pct: np.ndarray, start: float) -> np.ndarray:
    """Quantile path from eq. 2.11, in percent; the last of the n + 1 values is q_t+1."""
    driver = beta[0] + beta[2] * np.maximum(returns_pct, 0) + beta[3] * np.maximum(-returns_pct, 0)
    path, _ = signal.lfilter([1.0], [1.0, -beta[1]], driver, zi=[beta[1] * start])
    return np.concatenate(([start], path))


def _quantile_objective(beta: np.ndarray, returns_pct: np.ndarray, start: float, alpha: float) -> float:
    """Quantile loss of eq. 2.12; explosive quantile dynamics are ruled out."""
    if abs(beta[1]) >= 1:
        return 1e12
    errors = returns_pct - _caviar_path(beta, returns_pct, start)[:-1]
    return float(np.sum((alpha - (errors < 0)) * errors))


def _caviar_candidates(level: float) -> np.ndarray:
    """Random starting points whose implied long-run quantile is near the empirical one."""
    rng = np.random.default_rng(config.CAVIAR_SEED)
    count = config.CAVIAR_CANDIDATES
    persistence = rng.uniform(0.5, 0.99, count)
    slopes = rng.uniform(-0.6, 0.1, (count, 2))
    return np.column_stack((level * (1 - persistence), persistence, slopes))


def fit_caviar(returns: np.ndarray, alpha: float, garch: dict | None = None) -> dict:
    returns_pct = PERCENT * returns
    start = float(np.quantile(returns_pct[: min(100, len(returns_pct))], alpha))
    level = float(np.quantile(returns_pct, alpha))
    candidates = _caviar_candidates(level)
    losses = [_quantile_objective(beta, returns_pct, start, alpha) for beta in candidates]
    results = [
        optimize.minimize(
            _quantile_objective, candidates[index], args=(returns_pct, start, alpha),
            method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800},
        )
        for index in np.argsort(losses)[: config.CAVIAR_REFINED_STARTS]
    ]
    best = min(results, key=lambda result: result.fun)
    return {
        "beta1": float(best.x[0]), "beta2": float(best.x[1]), "beta3": float(best.x[2]),
        "beta4": float(best.x[3]), "q_start": start, "loss": float(best.fun),
    }


def forecast_caviar(returns: np.ndarray, fit: dict, alpha: float) -> Forecast:
    beta = np.array([fit["beta1"], fit["beta2"], fit["beta3"], fit["beta4"]])
    path = _caviar_path(beta, PERCENT * returns, fit["q_start"])
    q_next, q_last = path[-1] / PERCENT, path[-2] / PERCENT
    details = {
        "beta1": fit["beta1"], "beta2": fit["beta2"], "beta3": fit["beta3"], "beta4": fit["beta4"],
        "q_next": float(q_next), "q_last": float(q_last), "last_return": float(returns[-1]),
    }
    return Forecast(float(-q_next), None, details)


def caviar(returns: np.ndarray, alpha: float) -> Forecast:
    """Asymmetric slope CAViaR: models the quantile directly and gives VaR only (eq. 2.11, 2.12).

    q_t = b1 + b2 * q_t-1 + b3 * max(r_t-1, 0) + b4 * max(-r_t-1, 0), by quantile regression.
    Reference: Engle and Manganelli (2004).
    """
    return forecast_caviar(returns, fit_caviar(returns, alpha), alpha)


ESTIMATORS = {
    "hs": EstimatorSpec(fit_hs, forecast_hs, uses_garch=False),
    "normal": EstimatorSpec(fit_normal, forecast_normal, uses_garch=False),
    "garch_t": EstimatorSpec(fit_garch_t, forecast_garch_t, uses_garch=True),
    "fhs": EstimatorSpec(fit_fhs, forecast_fhs, uses_garch=True),
    "evt_pot": EstimatorSpec(fit_evt_pot, forecast_evt_pot, uses_garch=True),
    "caviar": EstimatorSpec(fit_caviar, forecast_caviar, uses_garch=False),
}
