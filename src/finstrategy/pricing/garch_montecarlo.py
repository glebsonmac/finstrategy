"""Risco com Monte Carlo via GARCH(1,1)-t — extraído da seção 6 do notebook
`put_de_protecao_DQN_montecarlos.ipynb`, sem mudar a lógica original.

A volatilidade de ações muda com o tempo e as caudas são mais gordas que a normal;
o GARCH-t captura as duas coisas. Ajustamos ao histórico real e simulamos o preço no
vencimento partindo da volatilidade atual, com drift neutro ao risco (juros menos
dividendos) — assim as diferenças entre strikes vêm só do prêmio e da forma da put,
não de uma aposta de alta/baixa embutida no modelo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal
from scipy.special import gammaln

TRADING_DAYS_PER_YEAR = 252


@dataclass
class GarchParameters:
    mean_daily_return: float
    omega: float
    alpha: float
    beta: float
    degrees_of_freedom: float
    next_day_variance: float

    @property
    def current_annual_volatility(self) -> float:
        return float(np.sqrt(self.next_day_variance * TRADING_DAYS_PER_YEAR))


def fit_garch_t(daily_log_returns: np.ndarray) -> GarchParameters:
    """Máxima verossimilhança do GARCH(1,1) com inovações t de Student."""
    returns_pct = daily_log_returns * 100

    def unpack(raw):
        mean, log_omega, alpha_logit, beta_logit, log_dof = raw
        alpha = 0.3 / (1 + np.exp(-alpha_logit))
        beta = (0.999 - alpha) / (1 + np.exp(-beta_logit))
        return mean, np.exp(log_omega), alpha, beta, 2.1 + np.exp(log_dof)

    def conditional_variances(mean, omega, alpha, beta):
        shocks = returns_pct - mean
        initial_variance = shocks.var()
        drivers = omega + alpha * np.concatenate([[initial_variance], shocks[:-1] ** 2])
        variances, _ = signal.lfilter([1.0], [1.0, -beta], drivers, zi=[beta * initial_variance])
        return shocks, np.maximum(variances, 1e-8)

    def negative_log_likelihood(raw):
        mean, omega, alpha, beta, dof = unpack(raw)
        shocks, variances = conditional_variances(mean, omega, alpha, beta)
        log_density = (
            gammaln((dof + 1) / 2)
            - gammaln(dof / 2)
            - 0.5 * np.log((dof - 2) * np.pi * variances)
            - (dof + 1) / 2 * np.log1p(shocks ** 2 / ((dof - 2) * variances))
        )
        return -log_density.sum()

    start = [returns_pct.mean(), np.log(returns_pct.var() * 0.05), -1.0, 2.0, np.log(4.0)]
    solution = optimize.minimize(
        negative_log_likelihood, start, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6}
    )
    mean, omega, alpha, beta, dof = unpack(solution.x)
    shocks, variances = conditional_variances(mean, omega, alpha, beta)
    next_variance = omega + alpha * shocks[-1] ** 2 + beta * variances[-1]
    return GarchParameters(mean / 100, omega / 1e4, alpha, beta, dof, next_variance / 1e4)


def simulate_prices_at_expiration(
    garch: GarchParameters,
    start_price: float,
    trading_days: int,
    annual_drift: float,
    number_of_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Preço final de cada trajetória; drift ajustado para E[preço final] = preço inicial × e^(drift × prazo)."""
    variance = np.full(number_of_paths, garch.next_day_variance)
    standardized = np.sqrt((garch.degrees_of_freedom - 2) / garch.degrees_of_freedom)
    cumulative_log_return = np.zeros(number_of_paths)
    for _ in range(trading_days):
        shock = np.sqrt(variance) * rng.standard_t(garch.degrees_of_freedom, number_of_paths) * standardized
        cumulative_log_return += shock
        variance = garch.omega + garch.alpha * shock ** 2 + garch.beta * variance
    centered = cumulative_log_return - np.log(np.mean(np.exp(cumulative_log_return)))
    return start_price * np.exp(centered + annual_drift * trading_days / TRADING_DAYS_PER_YEAR)


def simulate_price_paths(
    garch: GarchParameters,
    start_price: float,
    trading_days: int,
    annual_drift: float,
    number_of_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Como `simulate_prices_at_expiration`, mas devolve a trajetória diária inteira
    (shape `(number_of_paths, trading_days + 1)`, coluna 0 = `start_price`), não só o
    preço final. Usado pela DQN de saída (rl/exit_dqn.py), que precisa observar o
    preço em cada dia do caminho simulado para decidir quando vender a put — a versão
    só-terminal (função acima, extraída do notebook) continua intacta para não
    alterar o comportamento já validado do menu de proteção."""
    variance = np.full(number_of_paths, garch.next_day_variance)
    standardized = np.sqrt((garch.degrees_of_freedom - 2) / garch.degrees_of_freedom)
    cumulative_log_return = np.zeros(number_of_paths)
    log_paths = np.zeros((number_of_paths, trading_days + 1))
    for day in range(1, trading_days + 1):
        shock = np.sqrt(variance) * rng.standard_t(garch.degrees_of_freedom, number_of_paths) * standardized
        cumulative_log_return += shock
        variance = garch.omega + garch.alpha * shock ** 2 + garch.beta * variance
        log_paths[:, day] = cumulative_log_return
    # mesma centralização de risco-neutro usada na versão só-terminal, aplicada a cada coluna
    terminal_mean_log = np.log(np.mean(np.exp(log_paths[:, -1]))) if trading_days > 0 else 0.0
    drift_per_day = annual_drift / TRADING_DAYS_PER_YEAR
    days = np.arange(trading_days + 1)
    centered_log_paths = log_paths - (terminal_mean_log / max(trading_days, 1)) * days[None, :] + drift_per_day * days[None, :]
    return start_price * np.exp(centered_log_paths)


def risk_metrics(profit_per_share: np.ndarray, cost_per_share: float) -> dict:
    loss_pct = -profit_per_share / cost_per_share
    var_975 = np.percentile(loss_pct, 97.5)
    return {
        "P(perda)": float((profit_per_share < 0).mean()),
        "resultado esperado %": float(-loss_pct.mean()),
        "VaR 95%": float(np.percentile(loss_pct, 95)),
        "ES 97,5%": float(loss_pct[loss_pct >= var_975].mean()),
    }
