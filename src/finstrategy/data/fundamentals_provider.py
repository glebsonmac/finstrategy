"""Dados fundamentalistas normalizados, a partir do Yahoo Finance.

Primeira fonte (decisão da seção 0 do plan.md: usar o Yahoo para todos os tipos de
ação neste momento). Para tickers B3 onde o Yahoo vier incompleto, uma fonte
complementar (Fundamentus, B3/CVM) fica registrada como trabalho futuro — não bloqueia
o MVP, só reduz a cobertura de algumas colunas para alguns tickers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import yfinance as yf

from .errors import MarketDataError

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@dataclass
class FundamentalSnapshot:
    ticker: str
    name: str
    sector: Optional[str]
    industry: Optional[str]
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    price_to_book: Optional[float]
    roe: Optional[float]
    debt_to_equity: Optional[float]
    dividend_yield_ttm: Optional[float]
    revenue_growth_yoy: Optional[float]
    free_cash_flow: Optional[float]
    market_cap: Optional[float]
    profit_margin: Optional[float]


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ZeroDivisionError):
        return None


def get_fundamental_snapshot(ticker: str) -> FundamentalSnapshot:
    """Indicadores fundamentalistas normalizados para um ticker qualquer do Yahoo.

    Campos vêm de `Ticker.get_info()` quando disponível; nem todo ticker (sobretudo
    B3 `.SA` fora das blue chips) tem todos os campos — nesse caso o indicador fica
    `None` em vez de quebrar a coleta de todos os outros.
    """
    handle = yf.Ticker(ticker)
    try:
        info = handle.get_info() or {}
    except Exception as unexpected:
        raise MarketDataError(f"sem dados fundamentalistas no Yahoo: {unexpected}") from unexpected
    if not info:
        raise MarketDataError("sem dados fundamentalistas no Yahoo")

    free_cash_flow = None
    try:
        cashflow = handle.cashflow
        if cashflow is not None and not cashflow.empty and "Free Cash Flow" in cashflow.index:
            free_cash_flow = float(cashflow.loc["Free Cash Flow"].dropna().iloc[0])
    except Exception:
        free_cash_flow = None

    # yfinance devolve `dividendYield` já em pontos percentuais (ex.: 3.69 = 3,69%),
    # diferente de `returnOnEquity`/`revenueGrowth`/`profitMargins`, que vêm como
    # fração (ex.: 0.0986 = 9,86%). Normalizamos aqui para fração, para que todo
    # FundamentalSnapshot seja consistente e a formatação "{:.1%}" no frontend funcione
    # igual para todos os campos de percentual (detectado ao vivo: sem isso, o
    # frontend mostrava "Dividend yield: 369%" em vez de "3.69%").
    raw_dividend_yield = info.get("dividendYield")
    dividend_yield_ttm = raw_dividend_yield / 100 if raw_dividend_yield is not None else None

    return FundamentalSnapshot(
        ticker=ticker,
        name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector"),
        industry=info.get("industry"),
        pe_ratio=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        price_to_book=info.get("priceToBook"),
        roe=info.get("returnOnEquity"),
        debt_to_equity=info.get("debtToEquity"),
        dividend_yield_ttm=dividend_yield_ttm,
        revenue_growth_yoy=info.get("revenueGrowth"),
        free_cash_flow=free_cash_flow,
        market_cap=info.get("marketCap"),
        profit_margin=info.get("profitMargins"),
    )
