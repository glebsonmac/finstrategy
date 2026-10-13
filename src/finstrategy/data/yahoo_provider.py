"""Provedor de dados via Yahoo Finance.

Extraído da seção 3 do notebook `put_de_protecao_DQN_montecarlos.ipynb`, sem mudar o
comportamento original — só organizado em classe e com `TODAY` recalculado a cada
chamada (o notebook calculava uma vez por execução; aqui o coletor roda por dias, então
precisa da data corrente, não da data em que o processo começou).

Fora do pregão o Yahoo costuma devolver bid/ask zerados; isso é esperado e tratado pelo
chamador (`prepare_liquid_puts`, em strategy/protective_put.py), não aqui.
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd
import yfinance as yf

from .errors import MarketDataError

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _today() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


class YahooProvider:
    """Implementa MarketDataProvider usando yfinance."""

    def download_price_history(self, ticker: str, start_date: str) -> pd.Series:
        history = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
        if history is None or history.empty:
            raise MarketDataError("sem histórico de preços no Yahoo")
        closes = history["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes[closes > 0]

    def get_current_stock_price(self, ticker: str, fallback_price: float) -> float:
        try:
            last_price = float(yf.Ticker(ticker).fast_info["lastPrice"])
            return last_price if last_price > 0 else fallback_price
        except Exception:
            return fallback_price

    def get_trailing_dividend_yield(self, ticker: str, current_price: float) -> float:
        dividends = yf.Ticker(ticker).dividends
        if dividends is None or dividends.empty:
            return 0.0
        dividends.index = pd.to_datetime(dividends.index).tz_localize(None)
        today = _today()
        last_year_total = dividends[dividends.index >= today - pd.Timedelta(days=365)].sum()
        return float(last_year_total / current_price)

    def get_risk_free_rate(self) -> float:
        """Taxa do Treasury de 13 semanas (^IRX), em fração ao ano."""
        history = yf.Ticker("^IRX").history(period="1mo")
        if history is None or history.empty:
            raise MarketDataError("sem taxa ^IRX no Yahoo")
        return float(history["Close"].dropna().iloc[-1]) / 100

    def list_option_expirations(self, ticker: str) -> list:
        expirations = yf.Ticker(ticker).options
        if not expirations:
            raise MarketDataError("o Yahoo não tem opções para este ativo")
        return [pd.Timestamp(date) for date in expirations]

    def download_put_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame:
        chain = yf.Ticker(ticker).option_chain(expiration.strftime("%Y-%m-%d"))
        puts = chain.puts.copy()
        if puts.empty:
            raise MarketDataError(f"sem puts para o vencimento {expiration.date()}")
        return puts

    def download_call_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame:
        chain = yf.Ticker(ticker).option_chain(expiration.strftime("%Y-%m-%d"))
        calls = chain.calls.copy()
        if calls.empty:
            raise MarketDataError(f"sem calls para o vencimento {expiration.date()}")
        return calls

    def has_option_chain(self, ticker: str) -> bool:
        try:
            return bool(yf.Ticker(ticker).options)
        except Exception:
            return False


# Alias de módulo para quem prefere chamar sem instanciar a classe (compatibilidade
# com o estilo funcional do notebook original).
_default_provider = YahooProvider()
download_price_history = _default_provider.download_price_history
get_current_stock_price = _default_provider.get_current_stock_price
get_trailing_dividend_yield = _default_provider.get_trailing_dividend_yield
get_us_risk_free_rate = _default_provider.get_risk_free_rate
list_option_expirations = _default_provider.list_option_expirations
download_put_chain = _default_provider.download_put_chain
download_call_chain = _default_provider.download_call_chain
