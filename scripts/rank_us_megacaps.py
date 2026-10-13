#!/usr/bin/env python3
"""Reavalia as 20 maiores ações dos EUA por valor de mercado, AO VIVO no Yahoo Finance.

`config/us_top20_megacaps.yaml` foi gerado rodando este script (não é uma lista de
memória) — ranking de mercado muda, então rode de novo de vez em quando e atualize o
YAML manualmente com o resultado se a ordem/composição mudar muito.

Uso:
    python scripts/rank_us_megacaps.py
"""
from __future__ import annotations

import logging

import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Pool de candidatas conhecidas por serem historicamente mega-cap nos EUA. O ranking
# real sai do market cap AO VIVO, não da ordem desta lista — ela só precisa ser ampla
# o suficiente para conter as 20 maiores de verdade. Alphabet aparece como GOOGL e
# GOOG (duas classes de ação da mesma empresa); o ranking final usa só GOOGL.
CANDIDATES = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "BRK-B", "AVGO", "TSLA",
    "LLY", "JPM", "V", "XOM", "UNH", "MA", "WMT", "PG", "JNJ", "COST",
    "NFLX", "HD", "ORCL", "BAC", "CVX", "MRK", "ABBV", "KO", "PEP", "TMUS",
    "ADBE", "CRM", "AMD", "CSCO", "MCD", "TMO", "ABT", "NKE", "INTC", "IBM",
    "QCOM", "TXN", "LIN", "DHR", "VZ", "T", "PM", "HON", "UNP", "GS",
    "MS", "CAT", "BA", "MMM", "DIS", "PLTR", "ACN", "WFC", "AXP", "SPGI",
]

DUPLICATE_SHARE_CLASSES = {"GOOG"}  # mesma empresa que GOOGL; não conta duas vezes


def rank_market_caps(candidates: list[str]) -> list[tuple[str, float]]:
    results = []
    for ticker in candidates:
        if ticker in DUPLICATE_SHARE_CLASSES:
            continue
        try:
            market_cap = yf.Ticker(ticker).fast_info["marketCap"]
        except Exception:
            market_cap = None
        if market_cap:
            results.append((ticker, float(market_cap)))
    return sorted(results, key=lambda item: -item[1])


if __name__ == "__main__":
    ranked = rank_market_caps(CANDIDATES)
    print(f"{'Ticker':8s} {'Valor de mercado':>18s}")
    for ticker, market_cap in ranked[:20]:
        print(f"{ticker:8s} {market_cap / 1e9:>15,.0f} bi")
