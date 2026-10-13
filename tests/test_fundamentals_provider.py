import pandas as pd
import pytest

from finstrategy.data import fundamentals_provider as fp
from finstrategy.data.errors import MarketDataError


class _FakeTicker:
    def __init__(self, info=None, cashflow=None, raise_on_info=False):
        self._info = info or {}
        self._cashflow = cashflow
        self._raise_on_info = raise_on_info

    def get_info(self):
        if self._raise_on_info:
            raise RuntimeError("Yahoo fora do ar")
        return self._info

    @property
    def cashflow(self):
        return self._cashflow


def _full_info(**overrides) -> dict:
    base = {
        "longName": "Duke Energy Corporation", "sector": "Utilities", "industry": "Utilities - Regulated Electric",
        "trailingPE": 17.89, "forwardPE": 16.45, "priceToBook": 1.71, "returnOnEquity": 0.0986,
        "debtToEquity": 162.15, "dividendYield": 3.69, "revenueGrowth": 0.011, "profitMargins": 0.16,
        "marketCap": 91_760_000_000,
    }
    base.update(overrides)
    return base


def test_dividend_yield_is_normalized_from_percent_number_to_fraction(monkeypatch):
    # regressão: yfinance devolve dividendYield já em pontos percentuais (3.69 = 3,69%),
    # diferente de roe/revenueGrowth/profitMargins (fração) — sem normalizar, o
    # frontend mostrava "369%" em vez de "3.69%" (visto ao vivo com o mercado aberto).
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info=_full_info(dividendYield=3.69)))
    snapshot = fp.get_fundamental_snapshot("DUK")
    assert snapshot.dividend_yield_ttm == pytest.approx(0.0369)


def test_dividend_yield_none_stays_none(monkeypatch):
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info=_full_info(dividendYield=None)))
    snapshot = fp.get_fundamental_snapshot("XLU")
    assert snapshot.dividend_yield_ttm is None


def test_other_percent_fields_pass_through_unchanged_as_fractions(monkeypatch):
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info=_full_info(returnOnEquity=0.0986, revenueGrowth=0.011, profitMargins=0.16)))
    snapshot = fp.get_fundamental_snapshot("DUK")
    assert snapshot.roe == pytest.approx(0.0986)
    assert snapshot.revenue_growth_yoy == pytest.approx(0.011)
    assert snapshot.profit_margin == pytest.approx(0.16)


def test_free_cash_flow_extracted_from_cashflow_statement(monkeypatch):
    cashflow = pd.DataFrame({pd.Timestamp("2025-12-31"): [1_234_000_000.0]}, index=["Free Cash Flow"])
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info=_full_info(), cashflow=cashflow))
    snapshot = fp.get_fundamental_snapshot("DUK")
    assert snapshot.free_cash_flow == pytest.approx(1_234_000_000.0)


def test_free_cash_flow_missing_row_falls_back_to_none(monkeypatch):
    cashflow = pd.DataFrame({pd.Timestamp("2025-12-31"): [1.0]}, index=["Operating Cash Flow"])
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info=_full_info(), cashflow=cashflow))
    snapshot = fp.get_fundamental_snapshot("DUK")
    assert snapshot.free_cash_flow is None


def test_raises_market_data_error_when_info_is_empty(monkeypatch):
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info={}))
    with pytest.raises(MarketDataError):
        fp.get_fundamental_snapshot("NOPE")


def test_raises_market_data_error_when_info_call_fails(monkeypatch):
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(raise_on_info=True))
    with pytest.raises(MarketDataError):
        fp.get_fundamental_snapshot("NOPE")


def test_name_falls_back_to_shortname_then_ticker(monkeypatch):
    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info={"shortName": "Duke Energy"}))
    snapshot = fp.get_fundamental_snapshot("DUK")
    assert snapshot.name == "Duke Energy"

    monkeypatch.setattr(fp.yf, "Ticker", lambda t: _FakeTicker(info={"trailingPE": 10.0}))
    snapshot_no_name = fp.get_fundamental_snapshot("DUK")
    assert snapshot_no_name.name == "DUK"
