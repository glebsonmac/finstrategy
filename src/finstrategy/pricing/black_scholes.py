"""Black-Scholes para opções europeias: preço, Greeks e volatilidade implícita.

Módulo novo (não existia no notebook original, que só usava GARCH Monte Carlo para
risco). Serve de base para `strategy/exit_timing_bs.py` e `rl/exit_dqn.py`: a
volatilidade implícita comparada com a vol. GARCH indica se a opção está cara/barata, e
o theta decay projetado ao longo do tempo é o sinal central para decidir a janela de
saída do derivativo.

Opções americanas (ações dos EUA) podem ser exercidas antes do vencimento, mas o preço
europeu de Black-Scholes é a aproximação padrão usada aqui (ver limitações no notebook,
seção 11) — o erro de não considerar o exercício antecipado é pequeno para puts fora do
dinheiro/próximas do dinheiro, que é o caso típico de uma put de proteção.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import norm

_MIN_TIME_TO_EXPIRY_YEARS = 1e-6
_MIN_VOLATILITY = 1e-6


@dataclass
class OptionGreeks:
    price: float
    delta: float
    gamma: float
    theta: float  # por ano; dividir por 365 para "por dia"
    vega: float  # por 1.00 (100 p.p.) de vol; dividir por 100 para "por 1 p.p."
    rho: float


def _d1_d2(spot: float, strike: float, time_years: float, risk_free_rate: float, dividend_yield: float, volatility: float):
    volatility = max(volatility, _MIN_VOLATILITY)
    time_years = max(time_years, _MIN_TIME_TO_EXPIRY_YEARS)
    d1 = (
        np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_years
    ) / (volatility * np.sqrt(time_years))
    d2 = d1 - volatility * np.sqrt(time_years)
    return d1, d2


def _price_and_greeks_raw(
    option_type: str,
    spot,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> dict:
    """Núcleo vetorizável: `spot` pode ser escalar ou `np.ndarray`, o resto são
    escalares (um preço/Greek por ponto de `spot`). Usado tanto pela versão escalar
    (`price_and_greeks`, devolve `OptionGreeks`) quanto pela vetorizada
    (`price_and_greeks_array`, devolve arrays) — a DQN de saída (rl/exit_dqn.py)
    precisa repreçar milhares de caminhos simulados por dia e não pode pagar o custo
    de um laço Python chamando a versão escalar uma vez por caminho."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type precisa ser 'call' ou 'put'")

    time_years = max(time_years, _MIN_TIME_TO_EXPIRY_YEARS)
    volatility = max(volatility, _MIN_VOLATILITY)
    d1, d2 = _d1_d2(spot, strike, time_years, risk_free_rate, dividend_yield, volatility)
    discount_div = np.exp(-dividend_yield * time_years)
    discount_rate = np.exp(-risk_free_rate * time_years)
    sqrt_t = np.sqrt(time_years)

    gamma = discount_div * norm.pdf(d1) / (spot * volatility * sqrt_t)
    vega = spot * discount_div * norm.pdf(d1) * sqrt_t

    if option_type == "call":
        price = spot * discount_div * norm.cdf(d1) - strike * discount_rate * norm.cdf(d2)
        delta = discount_div * norm.cdf(d1)
        theta = (
            -spot * discount_div * norm.pdf(d1) * volatility / (2 * sqrt_t)
            - risk_free_rate * strike * discount_rate * norm.cdf(d2)
            + dividend_yield * spot * discount_div * norm.cdf(d1)
        )
        rho = strike * time_years * discount_rate * norm.cdf(d2)
    else:
        price = strike * discount_rate * norm.cdf(-d2) - spot * discount_div * norm.cdf(-d1)
        delta = -discount_div * norm.cdf(-d1)
        theta = (
            -spot * discount_div * norm.pdf(d1) * volatility / (2 * sqrt_t)
            + risk_free_rate * strike * discount_rate * norm.cdf(-d2)
            - dividend_yield * spot * discount_div * norm.cdf(-d1)
        )
        rho = -strike * time_years * discount_rate * norm.cdf(-d2)

    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def price_and_greeks(
    option_type: str,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> OptionGreeks:
    """Preço e Greeks de uma call ou put europeia (Black-Scholes-Merton, com dividendo contínuo).

    `option_type`: "call" ou "put". `time_years`: prazo até o vencimento em anos
    (dias corridos ÷ 365, ou dias úteis ÷ 252 — manter consistente com `volatility`).
    `spot` aqui é escalar; para repreçar um array de preços de uma vez (ex. vários
    caminhos simulados), usar `price_and_greeks_array`.
    """
    raw = _price_and_greeks_raw(option_type, spot, strike, time_years, risk_free_rate, volatility, dividend_yield)
    return OptionGreeks(**{key: float(value) for key, value in raw.items()})


def price_and_greeks_array(
    option_type: str,
    spot: np.ndarray,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> dict:
    """Mesma fórmula de `price_and_greeks`, vetorizada sobre um array de `spot`.

    Devolve um dict de arrays numpy (`price`, `delta`, `gamma`, `theta`, `vega`,
    `rho`), na mesma forma de `spot` — usado quando é preciso repreçar muitos
    caminhos simulados de uma vez (ex. `rl/exit_dqn.py`), onde chamar a versão
    escalar em laço Python por caminho seria o gargalo de desempenho do treino.
    """
    return _price_and_greeks_raw(option_type, np.asarray(spot, dtype=float), strike, time_years, risk_free_rate, volatility, dividend_yield)


def implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    time_years: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """Volatilidade implícita a partir do preço de mercado, via Brent (robusto a chute inicial ruim).

    Devolve NaN se o preço de mercado estiver fora da faixa possível (ex. abaixo do
    valor intrínseco) — quem chama decide como tratar (normalmente: ignorar a opção).
    """
    time_years = max(time_years, _MIN_TIME_TO_EXPIRY_YEARS)

    def price_difference(vol: float) -> float:
        return price_and_greeks(option_type, spot, strike, time_years, risk_free_rate, vol, dividend_yield).price - market_price

    low, high = 1e-4, 5.0
    if price_difference(low) > 0 or price_difference(high) < 0:
        return float("nan")
    try:
        return float(optimize.brentq(price_difference, low, high, xtol=1e-6, maxiter=200))
    except ValueError:
        return float("nan")
