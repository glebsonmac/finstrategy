"""Provedor de dados via MetaTrader5 — placeholder (ver plan.md, seção 0).

Ainda não decidido se MT5 entra no MVP da dissertação (depende de corretora, conta,
terminal MT5 aberto, e o pacote `MetaTrader5` só funciona em Windows nativo). Por isso
os métodos lançam `NotImplementedError` com uma mensagem clara em vez de silenciosamente
devolver dado incorreto. Implementar seguindo a mesma assinatura de `YahooProvider`
(MarketDataProvider, em base.py) quando a decisão for tomada.
"""
from __future__ import annotations

import pandas as pd


class MT5Provider:
    _NOT_READY = (
        "MT5Provider ainda não foi implementado — ver plan.md seção 0 "
        "(decisão pendente: MT5 entra no MVP ou fica no backlog)."
    )

    def download_price_history(self, ticker: str, start_date: str) -> pd.Series:
        raise NotImplementedError(self._NOT_READY)

    def get_current_stock_price(self, ticker: str, fallback_price: float) -> float:
        raise NotImplementedError(self._NOT_READY)

    def get_trailing_dividend_yield(self, ticker: str, current_price: float) -> float:
        raise NotImplementedError(self._NOT_READY)

    def get_risk_free_rate(self) -> float:
        raise NotImplementedError(self._NOT_READY)

    def list_option_expirations(self, ticker: str) -> list:
        raise NotImplementedError(self._NOT_READY)

    def download_put_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError(self._NOT_READY)

    def download_call_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError(self._NOT_READY)
