"""Calendário de pregão — decide se o coletor deve buscar dados agora.

A watchlist (seção 5/6 do plano) mistura tickers B3 (`.SA`) e EUA/ADR; cada grupo
segue um calendário de pregão diferente (BVMF vs. NYSE/NASDAQ). `exchange_for_ticker`
é uma heurística simples baseada no sufixo do ticker — suficiente para decidir qual
calendário consultar, não para nada além disso.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

CALENDAR_B3 = "BVMF"
CALENDAR_US = "NYSE"


def exchange_calendar_for_ticker(ticker: str) -> str:
    return CALENDAR_B3 if ticker.upper().endswith(".SA") else CALENDAR_US


@lru_cache(maxsize=8)
def _get_calendar(calendar_name: str):
    return mcal.get_calendar(calendar_name)


def is_market_open(now: pd.Timestamp, calendar_name: str) -> bool:
    """True se `now` cair dentro do horário de pregão de `calendar_name` no dia de `now`.

    `pandas_market_calendars` devolve `market_open`/`market_close` em UTC; um `now`
    naive (sem timezone) é assumido como **UTC** (nunca como o fuso local do
    calendário, que seria ambíguo) — passe um `now` tz-aware se tiver o fuso certo."""
    calendar = _get_calendar(calendar_name)
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    schedule = calendar.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        return False  # feriado ou fim de semana
    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]
    return market_open <= now <= market_close


def next_market_open(now: pd.Timestamp, calendar_name: str, lookahead_days: int = 10) -> pd.Timestamp | None:
    """Próximo horário de abertura do pregão a partir de `now` (None se não achar em `lookahead_days`)."""
    calendar = _get_calendar(calendar_name)
    now = pd.Timestamp(now)
    schedule = calendar.schedule(start_date=now.date(), end_date=(now + pd.Timedelta(days=lookahead_days)).date())
    if schedule.empty:
        return None
    opens = schedule["market_open"]
    now_utc = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    future_opens = opens[opens > now_utc]
    return future_opens.iloc[0] if not future_opens.empty else None


def any_market_open(now: pd.Timestamp, calendar_names: list[str] | None = None) -> bool:
    """True se QUALQUER um dos calendários (padrão: B3 e EUA) estiver aberto agora —
    o coletor roda enquanto pelo menos um mercado da watchlist estiver no pregão."""
    calendar_names = calendar_names or [CALENDAR_B3, CALENDAR_US]
    return any(is_market_open(now, name) for name in calendar_names)
