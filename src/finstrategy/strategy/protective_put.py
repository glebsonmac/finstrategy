"""Ação + put de proteção: menu de strikes, custo, perda máxima, break-even.

Extraído das seções 4, 5 e 7 do notebook `put_de_protecao_DQN_montecarlos.ipynb`, sem
mudar a lógica original. `analyze_ticker` é o ponto de entrada que junta dados (via
`MarketDataProvider`), GARCH Monte Carlo e o menu de proteção para um ticker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config_model import AnalysisSettings
from ..data.errors import MarketDataError
from ..pricing.garch_montecarlo import TRADING_DAYS_PER_YEAR, fit_garch_t, risk_metrics, simulate_prices_at_expiration


@dataclass
class ProtectivePutPlan:
    ticker: str
    contract_symbol: str
    expiration: pd.Timestamp
    days_to_expiration: int
    stock_price: float
    strike: float
    put_premium: float
    quote_source: str
    shares: int
    contracts: int
    cost_per_share: float
    total_cost: float
    guaranteed_sale_price: float
    guaranteed_sale_value: float
    max_loss_per_share: float
    max_loss_total: float
    max_loss_pct: float
    breakeven_price: float
    breakeven_change_pct: float
    protection_cost_pct: float
    annualized_protection_cost_pct: float
    implied_volatility: float
    open_interest: int


def shares_for_ticker(ticker: str, settings: AnalysisSettings) -> int:
    lot = settings.shares_per_option_contract
    if ticker in settings.shares_per_ticker:
        return int(settings.shares_per_ticker[ticker] // lot * lot)
    return settings.contracts_per_ticker * lot


def choose_expiration(expirations: list, today: pd.Timestamp, settings: AnalysisSettings) -> pd.Timestamp:
    long_enough = [date for date in expirations if (date - today).days >= settings.minimum_protection_days]
    if not long_enough:
        raise MarketDataError("nenhum vencimento com prazo mínimo")
    return min(long_enough, key=lambda date: abs((date - today).days - settings.target_protection_days))


def prepare_liquid_puts(raw_puts: pd.DataFrame, stock_price: float, settings: AnalysisSettings) -> pd.DataFrame:
    puts = raw_puts.copy()
    lowest_strike, highest_strike = (stock_price * bound for bound in settings.strike_range_vs_price)
    puts = puts[(puts["strike"] >= lowest_strike) & (puts["strike"] <= highest_strike)]

    has_live_quote = (puts["bid"] > 0) & (puts["ask"] > 0)
    puts["premium"] = np.where(has_live_quote, puts["ask"], puts["lastPrice"])
    puts["quote_source"] = np.where(has_live_quote, "ask ao vivo", "último negócio")
    mid_price = (puts["bid"] + puts["ask"]) / 2
    puts["bid_ask_spread_pct"] = np.where(has_live_quote, (puts["ask"] - puts["bid"]) / mid_price, np.nan)

    is_liquid = puts["openInterest"].fillna(0) >= settings.minimum_open_interest
    spread_ok = ~has_live_quote | (puts["bid_ask_spread_pct"] <= settings.maximum_bid_ask_spread_pct)
    puts = puts[is_liquid & spread_ok & (puts["premium"] > 0)]
    if puts.empty:
        raise MarketDataError("nenhuma put líquida na faixa de strikes")
    columns = [
        "contractSymbol", "strike", "premium", "quote_source", "bid", "ask", "lastPrice",
        "bid_ask_spread_pct", "openInterest", "volume", "impliedVolatility",
    ]
    return puts[columns].sort_values("strike").reset_index(drop=True)


def build_protective_put_plan(
    ticker: str,
    put_row: pd.Series,
    stock_price: float,
    expiration: pd.Timestamp,
    today: pd.Timestamp,
    shares: int,
    settings: AnalysisSettings,
) -> ProtectivePutPlan:
    contracts = shares // settings.shares_per_option_contract
    option_fee_per_share = settings.option_commission_per_contract / settings.shares_per_option_contract
    cost_per_share = stock_price * (1 + settings.stock_commission_pct) + put_row["premium"] + option_fee_per_share

    strike = float(put_row["strike"])
    net_sale_at_strike = strike * (1 - settings.stock_commission_pct)
    max_loss_per_share = cost_per_share - net_sale_at_strike
    breakeven_price = cost_per_share / (1 - settings.stock_commission_pct)
    days = max((expiration - today).days, 1)
    protection_cost_pct = put_row["premium"] / stock_price

    return ProtectivePutPlan(
        ticker=ticker, contract_symbol=put_row["contractSymbol"], expiration=expiration, days_to_expiration=days,
        stock_price=stock_price, strike=strike, put_premium=float(put_row["premium"]),
        quote_source=put_row["quote_source"], shares=shares, contracts=contracts,
        cost_per_share=cost_per_share, total_cost=cost_per_share * shares,
        guaranteed_sale_price=strike, guaranteed_sale_value=net_sale_at_strike * shares,
        max_loss_per_share=max_loss_per_share, max_loss_total=max_loss_per_share * shares,
        max_loss_pct=max_loss_per_share / cost_per_share,
        breakeven_price=breakeven_price, breakeven_change_pct=breakeven_price / stock_price - 1,
        protection_cost_pct=protection_cost_pct,
        annualized_protection_cost_pct=protection_cost_pct * 365 / days,
        implied_volatility=float(put_row["impliedVolatility"]), open_interest=int(put_row["openInterest"]),
    )


def profit_per_share_at_expiration(plan: ProtectivePutPlan, prices_at_expiration: np.ndarray, settings: AnalysisSettings) -> np.ndarray:
    sale_price = np.maximum(prices_at_expiration, plan.strike)
    return sale_price * (1 - settings.stock_commission_pct) - plan.cost_per_share


def stock_only_profit_per_share(stock_price: float, prices_at_expiration: np.ndarray, settings: AnalysisSettings) -> np.ndarray:
    cost = stock_price * (1 + settings.stock_commission_pct)
    return prices_at_expiration * (1 - settings.stock_commission_pct) - cost


def build_strike_menu(plans: list, simulated_prices: np.ndarray, settings: AnalysisSettings) -> pd.DataFrame:
    stock_price = plans[0].stock_price
    unprotected = risk_metrics(stock_only_profit_per_share(stock_price, simulated_prices, settings), stock_price * (1 + settings.stock_commission_pct))
    rows = []
    for plan in plans:
        protected = risk_metrics(profit_per_share_at_expiration(plan, simulated_prices, settings), plan.cost_per_share)
        expected_put_payoff = np.maximum(plan.strike - simulated_prices, 0).mean()
        rows.append({
            **asdict(plan),
            **{f"{name} (com put)": value for name, value in protected.items()},
            "P(exercer a put)": float((simulated_prices < plan.strike).mean()),
            "custo líquido esperado %": (plan.put_premium - expected_put_payoff) / stock_price,
            "redução do ES (p.p.)": unprotected["ES 97,5%"] - protected["ES 97,5%"],
        })
    menu = pd.DataFrame(rows)
    for name, value in unprotected.items():
        menu[f"{name} (sem put)"] = value
    return menu


def recommend_strike(menu: pd.DataFrame, settings: AnalysisSettings) -> tuple:
    within_limit = menu[menu["max_loss_pct"] <= settings.max_acceptable_loss_pct]
    if len(within_limit):
        return within_limit.loc[within_limit["custo líquido esperado %"].idxmin()], True
    return menu.loc[menu["max_loss_pct"].idxmin()], False


def analyze_ticker(ticker: str, provider, risk_free_rate: float, settings: AnalysisSettings, today: pd.Timestamp | None = None) -> dict:
    """Pipeline completo de ação + put de proteção para um ticker (requer opções no provider).

    `provider` é qualquer objeto que implemente `MarketDataProvider` (ver data/base.py).
    Levanta `MarketDataError` se o ticker não tiver histórico ou cadeia de opções — isso
    é o estado esperado para tickers sem derivativo (ver plan.md, seção 0), não um bug;
    quem chama decide se roda só a análise de ação (ver rl/entry_dqn.py) nesse caso.
    """
    today = today or pd.Timestamp.today().normalize()
    closes = provider.download_price_history(ticker, settings.history_start_date)
    stock_price = provider.get_current_stock_price(ticker, float(closes.iloc[-1]))
    expiration = choose_expiration(provider.list_option_expirations(ticker), today, settings)
    shares = shares_for_ticker(ticker, settings)
    if shares < settings.shares_per_option_contract:
        raise MarketDataError(f"quantidade definida menor que 1 contrato ({settings.shares_per_option_contract} ações)")
    required_capital = shares * stock_price
    if settings.max_capital_per_ticker and required_capital > settings.max_capital_per_ticker:
        raise MarketDataError(f"exige {required_capital:,.0f} para {shares} ações, acima do limite de capital")
    liquid_puts = prepare_liquid_puts(provider.download_put_chain(ticker, expiration), stock_price, settings)
    plans = [
        build_protective_put_plan(ticker, row, stock_price, expiration, today, shares, settings)
        for _, row in liquid_puts.iterrows()
    ]

    daily_log_returns = np.log(closes).diff().dropna().values
    garch = fit_garch_t(daily_log_returns)
    dividend_yield = provider.get_trailing_dividend_yield(ticker, stock_price)
    trading_days = max(int(round(plans[0].days_to_expiration * TRADING_DAYS_PER_YEAR / 365)), 1)
    rng = np.random.default_rng(settings.random_seed)
    simulated_prices = simulate_prices_at_expiration(
        garch, stock_price, trading_days, risk_free_rate - dividend_yield, settings.monte_carlo_paths, rng
    )
    menu = build_strike_menu(plans, simulated_prices, settings)
    recommended, respects_limit = recommend_strike(menu, settings)
    return {
        "closes": closes, "stock_price": stock_price, "shares": shares, "expiration": expiration,
        "garch": garch, "dividend_yield": dividend_yield, "simulated_prices": simulated_prices,
        "liquid_puts": liquid_puts, "menu": menu, "recommended": recommended,
        "recommended_respects_limit": respects_limit,
        "uses_last_trade_prices": bool((liquid_puts["quote_source"] == "último negócio").any()),
    }
