import numpy as np
import pandas as pd
import pytest

from finstrategy.config_model import AnalysisSettings
from finstrategy.rl.entry_dqn import (
    ACTION_BUY,
    ACTION_WAIT,
    EntryTimingEnvironment,
    QNetwork,
    bootstrap_mean_interval,
    compute_entry_signals,
    evaluate_entry_timing,
    greedy_purchase_days,
    train_entry_dqn,
)


def _synthetic_closes(days: int = 320, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, 0.018, days)
    prices = 50.0 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range("2023-01-02", periods=days)
    return pd.Series(prices, index=dates)


def test_compute_entry_signals_has_expected_columns_and_no_nan():
    signals = compute_entry_signals(_synthetic_closes())
    expected_columns = {
        "retorno 5 dias", "retorno 20 dias", "distância da média 20", "distância da média 50",
        "RSI 14", "vol. 20d / vol. 6 meses", "queda desde a máxima 60d",
    }
    assert set(signals.columns) == expected_columns
    assert not signals.isna().any().any()
    assert len(signals) > 0


def test_qnetwork_forward_and_train_step_reduce_error_on_toy_batch():
    rng = np.random.default_rng(0)
    network = QNetwork(input_size=4, output_size=2, hidden_units=8, learning_rate=1e-2, seed=0)
    states = rng.normal(size=(32, 4))
    actions = rng.integers(0, 2, size=32)
    targets = rng.normal(size=32)
    first_error = network.train_step(states, actions, targets)
    for _ in range(50):
        last_error = network.train_step(states, actions, targets)
    assert last_error < first_error


def test_environment_savings_pct_positive_when_price_drops():
    prices = np.array([100.0, 95.0, 90.0, 110.0])
    signals = np.zeros((4, 1))
    env = EntryTimingEnvironment(prices, signals, window_days=3)
    savings = env.savings_pct(np.array([0]), np.array([2]))
    assert savings[0] == pytest.approx(10.0)  # (100-90)/100 * 100


def test_greedy_purchase_days_defaults_to_last_day_when_network_prefers_waiting():
    class AlwaysWaitNetwork:
        def action_values(self, states):
            values = np.zeros((len(states), 2))
            values[:, ACTION_WAIT] = 1.0
            return values

    env = EntryTimingEnvironment(np.linspace(100, 90, 10), np.zeros((10, 1)), window_days=5)
    start_days = np.array([0])
    purchase_days = greedy_purchase_days(AlwaysWaitNetwork(), env, start_days)
    assert purchase_days[0] == start_days[0] + env.window_days - 1


def test_bootstrap_mean_interval_brackets_the_sample_mean():
    rng = np.random.default_rng(5)
    values = rng.normal(2.0, 1.0, 200)
    lower, upper = bootstrap_mean_interval(values, rng, draws=500)
    assert lower < values.mean() < upper


def test_evaluate_entry_timing_runs_end_to_end_and_returns_expected_shape():
    settings = AnalysisSettings(
        entry_window_days=5, dqn_training_rounds=3, dqn_hidden_units=8, test_period_fraction=0.3, random_seed=1,
    )
    closes = _synthetic_closes(days=320)
    evaluation = evaluate_entry_timing("FAKE", closes, settings)

    assert set(evaluation["summary"].index) == {"DQN", "Comprar no 1º dia", "Preço médio da janela", "Dia aleatório"}
    assert evaluation["test_windows"] > 0
    assert evaluation["today_action"] in {"comprar hoje", "esperar"}
    assert isinstance(evaluation["beats_benchmarks"], bool)
    assert 0 <= evaluation["p_value_vs_first_day"] <= 1
