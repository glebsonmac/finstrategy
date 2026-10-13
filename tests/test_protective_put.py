import numpy as np
import pandas as pd
import pytest

from finstrategy.config_model import AnalysisSettings
from finstrategy.data.errors import MarketDataError
from finstrategy.strategy.protective_put import (
    analyze_ticker,
    build_protective_put_plan,
    choose_expiration,
    prepare_liquid_puts,
    recommend_strike,
    shares_for_ticker,
)


def _settings(**overrides) -> AnalysisSettings:
    return AnalysisSettings(monte_carlo_paths=2_000, dqn_training_rounds=5, **overrides)


def test_shares_for_ticker_default_lot():
    settings = _settings(contracts_per_ticker=2)
    assert shares_for_ticker("AAPL", settings) == 200


def test_shares_for_ticker_custom_rounds_down_to_lot():
    settings = _settings(shares_per_ticker={"AAPL": 250})
    assert shares_for_ticker("AAPL", settings) == 200


def test_choose_expiration_picks_closest_to_target_above_minimum():
    today = pd.Timestamp("2026-01-01")
    settings = _settings(target_protection_days=45, minimum_protection_days=14)
    expirations = [today + pd.Timedelta(days=d) for d in (5, 20, 40, 90)]
    chosen = choose_expiration(expirations, today, settings)
    assert chosen == today + pd.Timedelta(days=40)


def test_choose_expiration_raises_when_all_too_short():
    today = pd.Timestamp("2026-01-01")
    settings = _settings(minimum_protection_days=30)
    expirations = [today + pd.Timedelta(days=5), today + pd.Timedelta(days=10)]
    with pytest.raises(MarketDataError):
        choose_expiration(expirations, today, settings)


def _raw_puts_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "contractSymbol": ["P80", "P90", "P95", "P60"],
        "strike": [80.0, 90.0, 95.0, 60.0],
        "bid": [1.0, 2.0, 3.0, 0.0],
        "ask": [1.2, 2.2, 6.0, 0.0],  # P95 tem spread largo; P60 sem cotação viva
        "lastPrice": [1.1, 2.1, 4.0, 0.5],
        "openInterest": [500, 50, 500, 500],  # P90 ilíquida (abaixo do mínimo)
        "volume": [10, 10, 10, 10],
        "impliedVolatility": [0.3, 0.3, 0.3, 0.3],
    })


def test_prepare_liquid_puts_filters_strike_range_liquidity_and_spread():
    settings = _settings(strike_range_vs_price=(0.80, 1.02), minimum_open_interest=100, maximum_bid_ask_spread_pct=0.20)
    stock_price = 100.0
    result = prepare_liquid_puts(_raw_puts_frame(), stock_price, settings)
    # P60 fica fora da faixa de strike (60 < 80% de 100); P90 é ilíquida (open interest 50);
    # P95 tem spread (6-3)/4.5=67% > 20% então cai fora
    assert list(result["contractSymbol"]) == ["P80"]
    assert result.iloc[0]["quote_source"] == "ask ao vivo"


def test_prepare_liquid_puts_uses_last_trade_when_market_closed():
    # mercado fechado: bid/ask zerados, cai para lastPrice (tratamento já existente no notebook)
    settings = _settings(strike_range_vs_price=(0.5, 1.5), minimum_open_interest=100, maximum_bid_ask_spread_pct=0.20)
    raw = pd.DataFrame({
        "contractSymbol": ["P90"], "strike": [90.0], "bid": [0.0], "ask": [0.0],
        "lastPrice": [2.5], "openInterest": [500], "volume": [0], "impliedVolatility": [0.3],
    })
    result = prepare_liquid_puts(raw, 100.0, settings)
    assert result.iloc[0]["quote_source"] == "último negócio"
    assert result.iloc[0]["premium"] == 2.5


def test_prepare_liquid_puts_raises_when_nothing_survives():
    settings = _settings(strike_range_vs_price=(0.99, 1.01))
    with pytest.raises(MarketDataError):
        prepare_liquid_puts(_raw_puts_frame(), 100.0, settings)


def test_recommend_strike_prefers_lowest_net_cost_within_loss_limit():
    menu = pd.DataFrame({
        "strike": [80.0, 90.0, 95.0],
        "max_loss_pct": [0.05, 0.03, 0.15],
        "custo líquido esperado %": [0.02, 0.01, 0.005],
    })
    settings = _settings(max_acceptable_loss_pct=0.08)
    chosen, respects_limit = recommend_strike(menu, settings)
    assert respects_limit is True
    assert chosen["strike"] == 90.0  # dentro do limite (0.03) e menor custo líquido (0.01)


def test_recommend_strike_falls_back_to_lowest_loss_when_none_within_limit():
    menu = pd.DataFrame({
        "strike": [80.0, 90.0],
        "max_loss_pct": [0.20, 0.30],
        "custo líquido esperado %": [0.02, 0.01],
    })
    settings = _settings(max_acceptable_loss_pct=0.08)
    chosen, respects_limit = recommend_strike(menu, settings)
    assert respects_limit is False
    assert chosen["strike"] == 80.0


class FakeProvider:
    """Provider determinístico para testar analyze_ticker sem rede."""

    def __init__(self, closes: pd.Series, current_price: float, expiration: pd.Timestamp, put_chain: pd.DataFrame):
        self._closes = closes
        self._current_price = current_price
        self._expiration = expiration
        self._put_chain = put_chain

    def download_price_history(self, ticker, start_date):
        return self._closes

    def get_current_stock_price(self, ticker, fallback_price):
        return self._current_price

    def get_trailing_dividend_yield(self, ticker, current_price):
        return 0.01

    def list_option_expirations(self, ticker):
        return [self._expiration]

    def download_put_chain(self, ticker, expiration):
        return self._put_chain

    def download_call_chain(self, ticker, expiration):
        raise MarketDataError("não usado neste teste")


def _synthetic_closes(days: int = 400, start_price: float = 100.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.015, days)
    prices = start_price * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=days)
    return pd.Series(prices, index=dates)


def test_analyze_ticker_end_to_end_with_fake_provider():
    today = pd.Timestamp.today().normalize()
    closes = _synthetic_closes()
    current_price = float(closes.iloc[-1])
    expiration = today + pd.Timedelta(days=45)
    put_chain = pd.DataFrame({
        "contractSymbol": ["PUT_LOW", "PUT_HIGH"],
        "strike": [current_price * 0.85, current_price * 0.95],
        "bid": [1.0, 2.5], "ask": [1.2, 2.8], "lastPrice": [1.1, 2.6],
        "openInterest": [500, 500], "volume": [50, 50], "impliedVolatility": [0.3, 0.3],
    })
    provider = FakeProvider(closes, current_price, expiration, put_chain)
    settings = _settings(minimum_open_interest=100)

    result = analyze_ticker("FAKE", provider, risk_free_rate=0.05, settings=settings, today=today)

    assert result["stock_price"] == current_price
    assert set(result["menu"]["strike"]) == {current_price * 0.85, current_price * 0.95}
    assert result["recommended"]["strike"] in {current_price * 0.85, current_price * 0.95}
    assert result["shares"] == 100
    assert not result["uses_last_trade_prices"]
