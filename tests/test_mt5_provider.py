"""Testa MT5Provider com um dublê do pacote `MetaTrader5` (que só instala no
Windows) — `mt5_module` é injetável de propósito em `MT5Provider` exatamente para
isso (ver data/mt5_provider.py)."""
import fnmatch
from collections import namedtuple

import pandas as pd
import pytest

from finstrategy.data.errors import MarketDataError
from finstrategy.data.mt5_provider import MT5Provider, _b3_root_ticker, _load_dotenv_if_available

FakeSymbol = namedtuple("FakeSymbol", ["name", "option_right", "option_strike", "expiration_time", "session_open_interest"])
FakeTick = namedtuple("FakeTick", ["bid", "ask", "last", "volume"])


def _fake_symbol(name, option_right=-1, option_strike=0.0, expiration_time=0, open_interest=0):
    return FakeSymbol(name=name, option_right=option_right, option_strike=option_strike,
                       expiration_time=expiration_time, session_open_interest=open_interest)


class FakeMT5:
    TIMEFRAME_D1 = "D1"

    def __init__(self, initialize_ok=True, symbols=(), ticks=None, rates=None):
        self.initialize_ok = initialize_ok
        self.initialize_calls = []
        self.symbols = list(symbols)
        self.ticks = ticks or {}
        self.rates = rates or {}
        self.selected = []

    def initialize(self, **kwargs):
        self.initialize_calls.append(kwargs)
        return self.initialize_ok

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (1, "erro falso")

    def symbol_select(self, symbol, enable):
        self.selected.append(symbol)
        return symbol in self.ticks or any(s.name == symbol for s in self.symbols) or symbol in self.rates

    def symbol_info(self, symbol):
        for s in self.symbols:
            if s.name == symbol:
                return s
        return None

    def symbol_info_tick(self, symbol):
        return self.ticks.get(symbol)

    def symbols_get(self, group=None):
        # Filtra de verdade por glob (igual o MT5 real) — se a fake devolvesse tudo
        # sempre, ela mascararia o próprio bug de padrão de busca errado que os
        # testes abaixo existem pra pegar.
        if group is None:
            return tuple(self.symbols)
        return tuple(s for s in self.symbols if fnmatch.fnmatch(s.name, group))

    def copy_rates_range(self, symbol, timeframe, start, end):
        return self.rates.get(symbol)


# --- .env (python-dotenv é opcional: não pode quebrar quando não está instalado) ---

def test_load_dotenv_if_available_does_nothing_without_python_dotenv(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "dotenv", None)  # simula pacote ausente
    _load_dotenv_if_available()  # não deve lançar


def test_load_dotenv_if_available_calls_load_dotenv_when_installed(monkeypatch):
    import sys
    import types

    calls = []
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: calls.append((a, k))
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    _load_dotenv_if_available()
    assert len(calls) == 1


def test_init_calls_load_dotenv_before_reading_env(monkeypatch):
    import sys
    import types

    monkeypatch.delenv("MT5_LOGIN", raising=False)
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: monkeypatch.setenv("MT5_LOGIN", "555")
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake = FakeMT5()
    MT5Provider(mt5_module=fake)
    assert fake.initialize_calls == [{"login": 555, "password": None, "server": None}]


def test_init_raises_market_data_error_when_initialize_fails():
    fake = FakeMT5(initialize_ok=False)
    with pytest.raises(MarketDataError):
        MT5Provider(mt5_module=fake)


def test_init_without_credentials_calls_initialize_with_no_kwargs():
    fake = FakeMT5()
    MT5Provider(mt5_module=fake)
    assert fake.initialize_calls == [{}]


def test_init_with_login_passes_credentials(monkeypatch):
    monkeypatch.delenv("MT5_LOGIN", raising=False)
    fake = FakeMT5()
    MT5Provider(login=12345, password="secret", server="Demo-Server", mt5_module=fake)
    assert fake.initialize_calls == [{"login": 12345, "password": "secret", "server": "Demo-Server"}]


def test_init_reads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("MT5_LOGIN", "999")
    monkeypatch.setenv("MT5_PASSWORD", "envpass")
    monkeypatch.setenv("MT5_SERVER", "EnvServer")
    fake = FakeMT5()
    MT5Provider(mt5_module=fake)
    assert fake.initialize_calls == [{"login": 999, "password": "envpass", "server": "EnvServer"}]


def test_context_manager_calls_shutdown():
    fake = FakeMT5()
    with MT5Provider(mt5_module=fake) as provider:
        assert isinstance(provider, MT5Provider)
    assert fake.shutdown_called is True


# --- download_price_history ---

def test_download_price_history_returns_close_series():
    rates = [
        {"time": 1_700_000_000, "open": 10, "high": 11, "low": 9, "close": 10.5, "tick_volume": 100},
        {"time": 1_700_086_400, "open": 10.5, "high": 11.5, "low": 10, "close": 11.0, "tick_volume": 120},
    ]
    fake = FakeMT5(rates={"DUK": rates})
    provider = MT5Provider(mt5_module=fake)
    closes = provider.download_price_history("DUK", "2023-11-01")
    assert list(closes.values) == [10.5, 11.0]
    assert isinstance(closes.index, pd.DatetimeIndex)


def test_download_price_history_raises_when_no_rates():
    fake = FakeMT5(rates={"DUK": []})
    provider = MT5Provider(mt5_module=fake)
    with pytest.raises(MarketDataError):
        provider.download_price_history("DUK", "2023-11-01")


def test_download_price_history_raises_when_symbol_not_selectable():
    fake = FakeMT5()  # DUK não está em nenhum dicionário -> symbol_select devolve False
    provider = MT5Provider(mt5_module=fake)
    with pytest.raises(MarketDataError):
        provider.download_price_history("DUK", "2023-11-01")


# --- get_current_stock_price ---

def test_get_current_stock_price_uses_last_when_available():
    fake = FakeMT5(ticks={"DUK": FakeTick(bid=99.0, ask=99.2, last=99.1, volume=10)})
    provider = MT5Provider(mt5_module=fake)
    assert provider.get_current_stock_price("DUK", fallback_price=1.0) == 99.1


def test_get_current_stock_price_falls_back_to_bid_when_last_is_zero():
    fake = FakeMT5(ticks={"DUK": FakeTick(bid=99.0, ask=99.2, last=0.0, volume=10)})
    provider = MT5Provider(mt5_module=fake)
    assert provider.get_current_stock_price("DUK", fallback_price=1.0) == 99.0


def test_get_current_stock_price_falls_back_when_symbol_missing():
    fake = FakeMT5()
    provider = MT5Provider(mt5_module=fake)
    assert provider.get_current_stock_price("NOPE", fallback_price=42.0) == 42.0


# --- dividend / risk-free ---

def test_get_trailing_dividend_yield_is_always_zero():
    provider = MT5Provider(mt5_module=FakeMT5())
    assert provider.get_trailing_dividend_yield("DUK", 100.0) == 0.0


def test_get_risk_free_rate_without_symbol_raises():
    provider = MT5Provider(mt5_module=FakeMT5())
    with pytest.raises(MarketDataError):
        provider.get_risk_free_rate()


def test_get_risk_free_rate_with_symbol():
    fake = FakeMT5(ticks={"US10Y": FakeTick(bid=4.0, ask=4.1, last=4.05, volume=0)})
    provider = MT5Provider(risk_free_symbol="US10Y", mt5_module=fake)
    assert provider.get_risk_free_rate() == pytest.approx(0.0405)


# --- _b3_root_ticker ---

@pytest.mark.parametrize("ticker, expected_root", [
    ("CMIG4.SA", "CMIG"),
    ("CMIG4", "CMIG"),
    ("TAEE11.SA", "TAEE"),
    ("TAEE11", "TAEE"),
    ("CPFE3.SA", "CPFE"),
    ("EGIE3.SA", "EGIE"),
    ("CPLE3.SA", "CPLE"),
    ("DUK", "DUK"),  # sem dígito final (EUA) -> inalterado
    ("SPY", "SPY"),
])
def test_b3_root_ticker(ticker, expected_root):
    assert _b3_root_ticker(ticker) == expected_root


# --- opções ---

def test_option_search_by_full_ticker_misses_b3_option_code():
    # Regressão do bug real: buscar pelo ticker completo (CMIG4) nunca bate com o
    # código de opção da B3 (CMIGO115), porque "CMIG4" não é um pedaço de
    # "CMIGO115" — falta o "4". É por isso que _option_symbols usa a raiz.
    assert not fnmatch.fnmatch("CMIGO115", "*CMIG4*")
    assert fnmatch.fnmatch("CMIGO115", "*CMIG*")


def test_list_option_expirations_finds_b3_style_option_by_root_ticker():
    # CMIGO115 = raiz CMIG + O (venda, março) + 115 (strike) — exemplo real do usuário.
    expiration_march = int(pd.Timestamp("2026-03-20").timestamp())
    symbols = [
        _fake_symbol("CMIG4", option_right=-1),  # a própria ação, não é opção
        _fake_symbol("CMIGO115", option_right=1, option_strike=11.5, expiration_time=expiration_march),
    ]
    fake = FakeMT5(symbols=symbols)
    provider = MT5Provider(mt5_module=fake)

    expirations = provider.list_option_expirations("CMIG4.SA")  # ticker como vem do Yahoo/watchlist
    assert len(expirations) == 1
    puts = provider.download_put_chain("CMIG4.SA", expirations[0])
    assert list(puts["contractSymbol"]) == ["CMIGO115"]


def test_list_option_expirations_filters_and_sorts():
    symbols = [
        _fake_symbol("DUK", option_right=-1),  # não é opção (ação/subjacente)
        _fake_symbol("DUK_C100_1220", option_right=0, option_strike=100.0, expiration_time=1_703_030_400),
        _fake_symbol("DUK_P090_1115", option_right=1, option_strike=90.0, expiration_time=1_699_920_000),
    ]
    fake = FakeMT5(symbols=symbols)
    provider = MT5Provider(mt5_module=fake)
    expirations = provider.list_option_expirations("DUK")
    assert expirations == sorted(expirations)
    assert len(expirations) == 2


def test_list_option_expirations_raises_when_no_option_symbols():
    fake = FakeMT5(symbols=[_fake_symbol("DUK", option_right=-1)])
    provider = MT5Provider(mt5_module=fake)
    with pytest.raises(MarketDataError):
        provider.list_option_expirations("DUK")


def test_download_put_chain_filters_by_right_and_expiration():
    expiration = pd.Timestamp("2026-12-20")
    exp_epoch = int(expiration.timestamp())
    symbols = [
        _fake_symbol("DUK_C100_1220", option_right=0, option_strike=100.0, expiration_time=exp_epoch),
        _fake_symbol("DUK_P090_1220", option_right=1, option_strike=90.0, expiration_time=exp_epoch, open_interest=50),
        _fake_symbol("DUK_P095_1115", option_right=1, option_strike=95.0, expiration_time=int(pd.Timestamp("2026-11-15").timestamp())),
    ]
    ticks = {"DUK_P090_1220": FakeTick(bid=1.0, ask=1.2, last=1.1, volume=5)}
    fake = FakeMT5(symbols=symbols, ticks=ticks)
    provider = MT5Provider(mt5_module=fake)

    puts = provider.download_put_chain("DUK", expiration)
    assert list(puts["contractSymbol"]) == ["DUK_P090_1220"]
    assert puts.iloc[0]["strike"] == 90.0
    assert puts.iloc[0]["ask"] == 1.2
    assert puts.iloc[0]["openInterest"] == 50


def test_download_call_chain_raises_when_nothing_matches():
    expiration = pd.Timestamp("2026-12-20")
    symbols = [_fake_symbol("DUK_P090_1220", option_right=1, option_strike=90.0, expiration_time=int(expiration.timestamp()))]
    fake = FakeMT5(symbols=symbols)
    provider = MT5Provider(mt5_module=fake)
    with pytest.raises(MarketDataError):
        provider.download_call_chain("DUK", expiration)


def test_inspect_symbol_returns_dict():
    fake = FakeMT5(symbols=[_fake_symbol("DUK", option_right=-1)])
    provider = MT5Provider(mt5_module=fake)
    info = provider.inspect_symbol("DUK")
    assert info["name"] == "DUK"


def test_inspect_symbol_raises_when_missing():
    provider = MT5Provider(mt5_module=FakeMT5())
    with pytest.raises(MarketDataError):
        provider.inspect_symbol("NOPE")
