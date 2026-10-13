import pandas as pd

from finstrategy.reporting.panel import build_panel_dataframe, build_plain_language_summary, describe_entry_signal


def _chosen_row() -> dict:
    return {
        "shares": 100, "contracts": 1, "stock_price": 100.0, "contract_symbol": "FAKE_PUT",
        "strike": 90.0, "expiration": pd.Timestamp("2026-12-31"), "put_premium": 3.0,
        "quote_source": "ask ao vivo", "total_cost": 10300.0, "guaranteed_sale_value": 9000.0,
        "max_loss_total": 1300.0, "max_loss_pct": 0.13, "breakeven_price": 103.0, "breakeven_change_pct": 0.03,
        "ES 97,5% (sem put)": 0.20, "ES 97,5% (com put)": 0.08, "P(perda) (com put)": 0.3,
    }


def _result(respects_limit=True) -> dict:
    return {"recommended": _chosen_row(), "expiration": pd.Timestamp("2026-12-31"), "recommended_respects_limit": respects_limit}


def _evaluation(action="comprar hoje", beats=True, gain=0.5) -> dict:
    return {"today_action": action, "beats_benchmarks": beats, "today_expected_gain_from_waiting_pct": gain}


def test_describe_entry_signal_weak_signal_when_not_beating_benchmarks():
    text = describe_entry_signal(_evaluation(action="esperar", beats=False))
    assert "sinal fraco" in text


def test_describe_entry_signal_wait_with_gain_estimate():
    text = describe_entry_signal(_evaluation(action="esperar", beats=True, gain=1.23))
    assert "esperar" in text and "1.23" in text


def test_describe_entry_signal_buy_today():
    assert describe_entry_signal(_evaluation(action="comprar hoje", beats=True)) == "comprar hoje"


def test_describe_entry_signal_handles_missing_evaluation():
    assert describe_entry_signal(None) == "sem avaliação de ponto de entrada"


def test_build_panel_dataframe_includes_eligible_and_not_eligible_tickers():
    results = {"AAPL": _result()}
    not_eligible = {"TAEE11.SA": "o Yahoo não tem opções para este ativo"}
    evaluations = {"AAPL": _evaluation(), "TAEE11.SA": _evaluation(action="esperar", beats=False)}
    panel = build_panel_dataframe(results, not_eligible, evaluations)
    assert set(panel.index) == {"AAPL", "TAEE11.SA"}
    assert "90.00" in panel.loc["AAPL", "strike / vencimento"]
    assert "sinal fraco" in panel.loc["TAEE11.SA", "entrada (DQN)"]


def test_build_panel_dataframe_empty_inputs_returns_empty_frame():
    panel = build_panel_dataframe({}, {}, {})
    assert panel.empty


def test_plain_language_summary_for_eligible_ticker_mentions_max_loss_and_breakeven():
    summary = build_plain_language_summary("AAPL", _result(), _evaluation())
    assert summary.is_eligible_for_protection is True
    joined = " ".join(summary.body_lines)
    assert "1.300" in joined.replace(",", ".") or "1300" in joined
    assert "103" in joined


def test_plain_language_summary_warns_when_loss_limit_not_respected():
    summary = build_plain_language_summary("AAPL", _result(respects_limit=False), _evaluation())
    joined = " ".join(summary.body_lines)
    assert "Atenção" in joined


def test_plain_language_summary_for_not_eligible_ticker():
    summary = build_plain_language_summary("TAEE11.SA", None, _evaluation(action="esperar", beats=False), not_eligible_reason="sem opções no Yahoo")
    assert summary.is_eligible_for_protection is False
    assert "sem opção de venda" in summary.body_lines[0] or "não tem opção" in summary.body_lines[0]
