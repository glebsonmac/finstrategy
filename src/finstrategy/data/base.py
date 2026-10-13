"""Interface comum de provedor de dados de mercado (hoje: Yahoo; depois: MetaTrader5).

Qualquer novo provedor (MT5Provider, etc.) implementa este protocolo para que o resto
do pipeline (pricing, strategy, rl, collector) não dependa da fonte concreta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class SymbolMatch:
    """Um resultado de busca por nome/código (ver symbol_search.search_symbols)."""

    symbol: str
    name: str
    exchange: str
    quote_type: str


@dataclass
class SymbolInfo:
    """Resultado de validate_symbol: ticker confirmado + metadados mínimos."""

    symbol: str
    name: str
    current_price: float
    currency: str
    exchange: str
    has_options: bool


class MarketDataProvider(Protocol):
    def download_price_history(self, ticker: str, start_date: str) -> pd.Series: ...

    def get_current_stock_price(self, ticker: str, fallback_price: float) -> float: ...

    def get_trailing_dividend_yield(self, ticker: str, current_price: float) -> float: ...

    def get_risk_free_rate(self) -> float: ...

    def list_option_expirations(self, ticker: str) -> list: ...

    def download_put_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame: ...

    def download_call_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame: ...
