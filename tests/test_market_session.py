import pandas as pd

from finstrategy.collector.market_session import (
    any_market_open,
    exchange_calendar_for_ticker,
    is_market_open,
    next_market_open,
)


def test_exchange_calendar_for_ticker_b3_suffix():
    assert exchange_calendar_for_ticker("TAEE11.SA") == "BVMF"
    assert exchange_calendar_for_ticker("taee11.sa") == "BVMF"


def test_exchange_calendar_for_ticker_defaults_to_nyse():
    assert exchange_calendar_for_ticker("AAPL") == "NYSE"
    assert exchange_calendar_for_ticker("XLU") == "NYSE"


def test_nyse_open_during_trading_hours_on_a_weekday():
    # terça-feira, 13/10/2026, 11h horário de Nova York (pregão 9h30-16h)
    moment = pd.Timestamp("2026-10-13 11:00:00", tz="America/New_York")
    assert is_market_open(moment, "NYSE") is True


def test_nyse_closed_after_hours_on_a_weekday():
    moment = pd.Timestamp("2026-10-13 20:00:00", tz="America/New_York")
    assert is_market_open(moment, "NYSE") is False


def test_nyse_closed_on_sunday():
    sunday = pd.Timestamp("2026-10-11 11:00:00", tz="America/New_York")
    assert is_market_open(sunday, "NYSE") is False


def test_bvmf_open_during_trading_hours_on_a_weekday():
    moment = pd.Timestamp("2026-10-13 11:00:00", tz="America/Sao_Paulo")
    assert is_market_open(moment, "BVMF") is True


def test_naive_timestamp_is_assumed_to_be_utc():
    # NYSE pregão em UTC nesse dia: 13:30-20:00 (ver pandas_market_calendars); 15h UTC está dentro
    naive_moment = pd.Timestamp("2026-10-13 15:00:00")
    assert is_market_open(naive_moment, "NYSE") is True
    naive_moment_closed = pd.Timestamp("2026-10-13 11:00:00")
    assert is_market_open(naive_moment_closed, "NYSE") is False


def test_next_market_open_after_a_closed_period():
    sunday = pd.Timestamp("2026-10-11 11:00:00", tz="America/New_York")
    opening = next_market_open(sunday, "NYSE")
    assert opening is not None
    assert opening > sunday


def test_any_market_open_true_when_at_least_one_calendar_is_open():
    moment = pd.Timestamp("2026-10-13 11:00:00", tz="America/New_York")  # NYSE aberta
    assert any_market_open(moment, ["BVMF", "NYSE"]) is True


def test_any_market_open_false_when_all_calendars_closed():
    sunday = pd.Timestamp("2026-10-11 11:00:00", tz="America/New_York")
    assert any_market_open(sunday, ["BVMF", "NYSE"]) is False
