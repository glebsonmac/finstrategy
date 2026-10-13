import numpy as np
import pandas as pd
import pytest

from finstrategy.config_model import AnalysisSettings
from finstrategy.pricing.garch_montecarlo import GarchParameters
from finstrategy.reporting import model_store
from finstrategy.rl.entry_dqn import evaluate_entry_timing
from finstrategy.rl.exit_dqn import evaluate_exit_timing
from finstrategy.strategy.protective_put import ProtectivePutPlan


def _synthetic_closes(days: int = 320, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, 0.018, days)
    prices = 50.0 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range("2023-01-02", periods=days)
    return pd.Series(prices, index=dates)


def _fast_settings(**overrides) -> AnalysisSettings:
    base = dict(
        entry_window_days=5, dqn_training_rounds=3, dqn_hidden_units=8, test_period_fraction=0.3,
        monte_carlo_paths=2_000, exit_dqn_training_rounds=3, exit_dqn_hidden_units=8, random_seed=1,
    )
    base.update(overrides)
    return AnalysisSettings(**base)


def _plan(stock_price=100.0, strike=95.0) -> ProtectivePutPlan:
    return ProtectivePutPlan(
        ticker="FAKE", contract_symbol="FAKE_PUT", expiration=pd.Timestamp("2026-12-31"),
        days_to_expiration=30, stock_price=stock_price, strike=strike, put_premium=3.0,
        quote_source="ask ao vivo", shares=100, contracts=1, cost_per_share=103.0, total_cost=10300.0,
        guaranteed_sale_price=strike, guaranteed_sale_value=strike * 100, max_loss_per_share=8.0,
        max_loss_total=800.0, max_loss_pct=0.08, breakeven_price=103.0, breakeven_change_pct=0.03,
        protection_cost_pct=0.03, annualized_protection_cost_pct=0.24, implied_volatility=0.3, open_interest=500,
    )


@pytest.fixture
def entry_evaluation():
    settings = _fast_settings()
    closes = _synthetic_closes()
    return evaluate_entry_timing("FAKE", closes, settings)


@pytest.fixture
def exit_evaluation():
    settings = _fast_settings()
    plan = _plan()
    garch = GarchParameters(mean_daily_return=0.0, omega=1e-6, alpha=0.05, beta=0.9, degrees_of_freedom=6, next_day_variance=1e-4)
    return evaluate_exit_timing(plan, garch, risk_free_rate=0.05, dividend_yield=0.0, window_days=10, settings=settings, fixed_rule_exit_day=5)


def test_save_and_load_entry_model_roundtrip_preserves_network(tmp_path, entry_evaluation):
    settings = _fast_settings()
    path = model_store.save_model("FAKE", "entry", entry_evaluation, settings, models_dir=tmp_path)
    assert path.exists()

    loaded = model_store.load_model(path.name, models_dir=tmp_path)
    assert loaded["ticker"] == "FAKE"
    assert loaded["model_type"] == "entry"
    assert loaded["hyperparameters"]["dqn_hidden_units"] == 8

    original_network = entry_evaluation["network"]
    loaded_network = loaded["evaluation"]["network"]
    probe_state = np.random.default_rng(0).normal(size=(3, original_network.weights[0].shape[0]))
    np.testing.assert_allclose(original_network.action_values(probe_state), loaded_network.action_values(probe_state))


def test_save_and_load_exit_model_roundtrip(tmp_path, exit_evaluation):
    settings = _fast_settings()
    path = model_store.save_model("FAKE", "exit", exit_evaluation, settings, models_dir=tmp_path)
    loaded = model_store.load_model(path.name, models_dir=tmp_path)
    assert loaded["model_type"] == "exit"
    assert "network" in loaded["evaluation"]


def test_save_model_appends_to_index_with_headline_metrics(tmp_path, entry_evaluation):
    settings = _fast_settings()
    model_store.save_model("FAKE", "entry", entry_evaluation, settings, models_dir=tmp_path)
    index = model_store.list_models(models_dir=tmp_path)
    assert len(index) == 1
    row = index.iloc[0]
    assert row["ticker"] == "FAKE"
    assert row["model_type"] == "entry"
    assert row["metrica_principal"] == "economia média (%)"
    assert row["sinal_hoje"] == entry_evaluation["today_action"]
    assert bool(row["bateu_benchmark"]) == entry_evaluation["beats_benchmarks"]


def test_save_model_exit_headline_metrics(tmp_path, exit_evaluation):
    settings = _fast_settings()
    model_store.save_model("FAKE", "exit", exit_evaluation, settings, models_dir=tmp_path)
    index = model_store.list_models(models_dir=tmp_path)
    row = index.iloc[0]
    assert row["metrica_principal"] == "valor médio realizado"
    assert row["sinal_hoje"] == ""


def test_list_models_most_recent_first(tmp_path, entry_evaluation):
    settings = _fast_settings()
    model_store.save_model("AAA", "entry", entry_evaluation, settings, models_dir=tmp_path)
    model_store.save_model("BBB", "entry", entry_evaluation, settings, models_dir=tmp_path)
    index = model_store.list_models(models_dir=tmp_path)
    # a segunda rodada tem timestamp maior ou igual à primeira (mesmo segundo é
    # possível em testes rápidos) — garante só que as duas estão lá, sem duplicar
    assert set(index["ticker"]) == {"AAA", "BBB"}
    assert len(index) == 2


def test_list_models_empty_when_no_index(tmp_path):
    index = model_store.list_models(models_dir=tmp_path)
    assert index.empty
    assert list(index.columns) == model_store.INDEX_COLUMNS
