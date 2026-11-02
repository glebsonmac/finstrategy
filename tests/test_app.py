"""Run the real page in-process with Streamlit's AppTest and a simulated data source."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from riskbt import config, data
from riskbt.i18n import t

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fake_report(missing=()):
    return data.DataReport(
        list(missing), 0, 0, {"PETR4.SA": 2}, {"PETR4.SA": 3}, {}, {"VALE3.SA": 4}, [],
        {}, [{"ticker": "VALE3.SA", "date": "2020-03-09", "log_return": -0.45}],
    )


def fake_load_prices(tickers, start, end, as_of, *args, **kwargs):
    rng = np.random.default_rng(51)
    days = 1200
    columns = [*tickers, config.IBOVESPA_TICKER]
    volatility = np.where(np.arange(days) % 300 < 60, 0.03, 0.01)[:, None]
    returns = rng.standard_t(5, (days, len(columns))) * volatility
    index = pd.bdate_range("2021-01-04", periods=days)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=columns)
    missing = [ticker for ticker in tickers if ticker.startswith("NOPE")]
    return prices, fake_report(missing)


@pytest.fixture(autouse=True)
def fresh_app(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(data, "load_prices", fake_load_prices)


def start(lang="en"):
    app = AppTest.from_file(APP, default_timeout=300).run()
    assert not app.exception
    if lang != "en":
        app.selectbox[0].select(lang).run()
    return app


def press_run(app):
    app.button[0].click().run()
    assert not app.exception, app.exception
    return app


def all_text(app):
    pieces = [element.value for kind in ("markdown", "caption", "subheader", "title", "info", "warning", "error") for element in getattr(app, kind)]
    return "\n".join(str(piece) for piece in pieces)


def test_starts_in_english_with_notice_and_no_result():
    app = start()
    assert app.title[0].value == t("app.title", "en")
    assert t("app.notice", "en") in all_text(app)
    assert t("input.start_hint", "en") in all_text(app)
    assert len(app.subheader) == 1


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_full_page_renders_all_blocks_in_order(lang):
    app = press_run(start(lang))
    headers = [element.value for element in app.subheader]
    expected = ["input.header", "main.header", "compare.header", "backtest.header", "data.header", "download.header"]
    assert headers == [t(key, lang) for key in expected]
    assert t("app.notice", lang) in all_text(app)
    assert len(app.metric) == 4
    assert len(app.expander) >= 9  # main + six estimators + tests + data
    assert len(app.dataframe) >= 2
    assert t("recommend.why", lang) in all_text(app)


def test_portuguese_page_uses_portuguese_numbers():
    app = press_run(start("pt"))
    assert "Em 99 de cada 100 dias" in all_text(app)
    value = app.metric[0].value
    assert value.startswith("R$ ") and "," in value[-3:]


def test_language_switch_keeps_the_result():
    app = press_run(start("en"))
    app.selectbox[0].select("pt").run()
    assert not app.exception
    assert app.subheader[1].value == t("main.header", "pt")


def test_estimator_selector_changes_backtest_without_failing():
    app = press_run(start())
    for name in ("hs", "caviar"):
        app.selectbox[1].select(name).run()
        assert not app.exception
    app.selectbox[2].select(config.ALPHA_VAR).run()
    assert not app.exception


def test_weights_must_sum_to_one():
    app = start()
    frame = pd.DataFrame({"ticker": ["PETR4", "VALE3"], "weight": [0.5, 0.2]})
    parsed = app_module_parse(frame, 1000.0, "en")
    assert parsed[2] == t("error.weight_sum", "en", total="0.7000")


def app_module_parse(frame, amount, lang):
    """Load parse_portfolio without running the page."""
    import ast
    source = Path(APP).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_portfolio")
    namespace = {"t": t, "pd": pd, "format_number": __import__("riskbt.i18n", fromlist=["format_number"]).format_number}
    exec(compile(ast.Module([function], []), APP, "exec"), namespace)
    return namespace["parse_portfolio"](frame, amount, lang)


def test_parse_normalizes_tickers_and_rejects_bad_input():
    frame = pd.DataFrame({"ticker": [" petr4 ", "VALE3.SA"], "weight": [0.5, 0.5]})
    assert app_module_parse(frame, 1000.0, "en")[:2] == (("PETR4.SA", "VALE3.SA"), (0.5, 0.5))
    duplicate = pd.DataFrame({"ticker": ["PETR4", "PETR4.SA"], "weight": [0.5, 0.5]})
    assert app_module_parse(duplicate, 1000.0, "pt")[2] == t("error.duplicate", "pt")
    negative = pd.DataFrame({"ticker": ["A", "B"], "weight": [1.5, -0.5]})
    assert app_module_parse(negative, 1000.0, "en")[2] == t("error.negative", "en")
    assert app_module_parse(pd.DataFrame({"ticker": [None], "weight": [None]}), 1000.0, "en")[2] == t("error.no_assets", "en")
    assert app_module_parse(frame, 0.0, "en")[2] == t("error.amount", "en")


def test_unknown_ticker_shows_a_translated_error(monkeypatch):
    def with_missing(tickers, *args, **kwargs):
        prices, _ = fake_load_prices(tickers, *args, **kwargs)
        return prices.drop(columns="PETR4.SA"), fake_report(["PETR4.SA"])

    monkeypatch.setattr(data, "load_prices", with_missing)
    app = press_run(start("pt"))
    assert app.error[0].value == t("error.unknown_tickers", "pt", tickers="PETR4.SA")
    assert len(app.subheader) == 1


def test_short_history_shows_a_translated_error(monkeypatch):
    def short(tickers, *args, **kwargs):
        prices, report = fake_load_prices(tickers, *args, **kwargs)
        return prices.iloc[:300], report

    monkeypatch.setattr(data, "load_prices", short)
    app = press_run(start())
    assert app.error[0].value == t("error.too_short", "en", days=299, needed=1001)


def test_download_failure_shows_a_translated_error(monkeypatch):
    def failing(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr(data, "load_prices", failing)
    app = press_run(start())
    assert app.error[0].value == t("error.download", "en")
