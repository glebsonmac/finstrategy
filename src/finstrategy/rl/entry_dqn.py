"""DQN de entrada: vale comprar a ação hoje ou esperar até `entry_window_days` dias úteis?

Extraído das seções 9 e 10 do notebook `put_de_protecao_DQN_montecarlos.ipynb`, sem
mudar a lógica original (MLP em NumPy, Double DQN, replay buffer, avaliação honesta
treino/teste sem sobreposição com bootstrap e p-valor contra benchmarks).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import stats

from ..config_model import AnalysisSettings

ENTRY_SIGNAL_NAMES = [
    "retorno 5 dias", "retorno 20 dias", "distância da média 20", "distância da média 50",
    "RSI 14", "vol. 20d / vol. 6 meses", "queda desde a máxima 60d",
]
ACTION_WAIT, ACTION_BUY = 0, 1


def compute_entry_signals(closes: pd.Series) -> pd.DataFrame:
    log_returns = np.log(closes).diff()
    price_changes = closes.diff()
    average_gain = price_changes.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    average_loss = (-price_changes.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    signals = pd.DataFrame({
        "retorno 5 dias": closes / closes.shift(5) - 1,
        "retorno 20 dias": closes / closes.shift(20) - 1,
        "distância da média 20": closes / closes.rolling(20).mean() - 1,
        "distância da média 50": closes / closes.rolling(50).mean() - 1,
        "RSI 14": 100 - 100 / (1 + average_gain / average_loss.replace(0, np.nan)),
        "vol. 20d / vol. 6 meses": log_returns.rolling(20).std() / log_returns.rolling(126).std(),
        "queda desde a máxima 60d": closes / closes.rolling(60).max() - 1,
    })
    return signals.dropna()


class QNetwork:
    """MLP de 2 camadas ocultas ReLU, estimando o valor de cada ação; treino com Adam."""

    def __init__(self, input_size: int, output_size: int, hidden_units: int, learning_rate: float, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_units)), np.zeros(hidden_units),
            rng.normal(0, np.sqrt(2 / hidden_units), (hidden_units, hidden_units)), np.zeros(hidden_units),
            rng.normal(0, 0.01, (hidden_units, output_size)), np.zeros(output_size),
        ]
        self.first_moments = [np.zeros_like(w) for w in self.weights]
        self.second_moments = [np.zeros_like(w) for w in self.weights]
        self.learning_rate, self.update_count = learning_rate, 0

    def _forward(self, states):
        w1, b1, w2, b2, w3, b3 = self.weights
        hidden1_pre = states @ w1 + b1
        hidden1 = np.maximum(hidden1_pre, 0)
        hidden2_pre = hidden1 @ w2 + b2
        hidden2 = np.maximum(hidden2_pre, 0)
        return hidden2 @ w3 + b3, (states, hidden1_pre, hidden1, hidden2_pre, hidden2)

    def action_values(self, states: np.ndarray) -> np.ndarray:
        return self._forward(states)[0]

    def train_step(self, states, actions, targets) -> float:
        predictions, (inputs, hidden1_pre, hidden1, hidden2_pre, hidden2) = self._forward(states)
        batch = np.arange(len(states))
        errors = np.clip(predictions[batch, actions] - targets, -1, 1)
        output_grad = np.zeros_like(predictions)
        output_grad[batch, actions] = errors / len(states)
        hidden2_grad = (output_grad @ self.weights[4].T) * (hidden2_pre > 0)
        hidden1_grad = (hidden2_grad @ self.weights[2].T) * (hidden1_pre > 0)
        gradients = [
            inputs.T @ hidden1_grad, hidden1_grad.sum(0), hidden1.T @ hidden2_grad, hidden2_grad.sum(0),
            hidden2.T @ output_grad, output_grad.sum(0),
        ]
        self.update_count += 1
        for index, gradient in enumerate(gradients):
            self.first_moments[index] = 0.9 * self.first_moments[index] + 0.1 * gradient
            self.second_moments[index] = 0.999 * self.second_moments[index] + 0.001 * gradient ** 2
            corrected_first = self.first_moments[index] / (1 - 0.9 ** self.update_count)
            corrected_second = self.second_moments[index] / (1 - 0.999 ** self.update_count)
            self.weights[index] -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        return float((errors ** 2).mean())


class EntryTimingEnvironment:
    def __init__(self, prices: np.ndarray, normalized_signals: np.ndarray, window_days: int):
        self.prices, self.signals, self.window_days = prices, normalized_signals, window_days

    def states(self, start_days: np.ndarray, days_elapsed: int) -> np.ndarray:
        time_left = np.full((len(start_days), 1), 1 - days_elapsed / self.window_days)
        return np.hstack([self.signals[start_days + days_elapsed], time_left])

    def savings_pct(self, start_days: np.ndarray, purchase_days: np.ndarray) -> np.ndarray:
        first_day_price = self.prices[start_days]
        return 100 * (first_day_price - self.prices[purchase_days]) / first_day_price


def train_entry_dqn(environment: EntryTimingEnvironment, training_start_days: np.ndarray, settings: AnalysisSettings, seed: int) -> QNetwork:
    rng = np.random.default_rng(seed)
    state_size = environment.signals.shape[1] + 1
    online_network = QNetwork(state_size, 2, settings.dqn_hidden_units, 1e-3, seed)
    target_network = copy.deepcopy(online_network)
    capacity, parallel_episodes, batch_size = 100_000, 64, 256
    memory_states = np.zeros((capacity, state_size))
    memory_next_states = np.zeros((capacity, state_size))
    memory_actions = np.zeros(capacity, int)
    memory_rewards = np.zeros(capacity)
    memory_done = np.zeros(capacity)
    memory_next_is_last_day = np.zeros(capacity, bool)
    write_position, stored = 0, 0

    for training_round in range(settings.dqn_training_rounds):
        exploration_rate = max(0.05, 1 - training_round / (0.7 * settings.dqn_training_rounds))
        start_days = rng.choice(training_start_days, parallel_episodes)
        still_waiting = np.ones(parallel_episodes, bool)
        for days_elapsed in range(environment.window_days):
            active = np.where(still_waiting)[0]
            if not len(active):
                break
            states = environment.states(start_days[active], days_elapsed)
            greedy = online_network.action_values(states).argmax(1)
            explore = rng.random(len(active)) < exploration_rate
            actions = np.where(explore, rng.integers(2, size=len(active)), greedy)
            last_day = days_elapsed == environment.window_days - 1
            if last_day:
                actions[:] = ACTION_BUY
            bought = actions == ACTION_BUY
            rewards = np.where(bought, environment.savings_pct(start_days[active], start_days[active] + days_elapsed), 0.0)
            next_states = states if last_day else environment.states(start_days[active], days_elapsed + 1)
            slots = (write_position + np.arange(len(active))) % capacity
            memory_states[slots], memory_actions[slots], memory_rewards[slots] = states, actions, rewards
            memory_next_states[slots], memory_done[slots] = next_states, bought.astype(float)
            memory_next_is_last_day[slots] = days_elapsed + 1 == environment.window_days - 1
            write_position = (write_position + len(active)) % capacity
            stored = min(stored + len(active), capacity)
            still_waiting[active[bought]] = False

            if stored >= 1_000:
                for _ in range(4):
                    sample = rng.integers(stored, size=batch_size)
                    next_states_sample = memory_next_states[sample]
                    best_next_actions = online_network.action_values(next_states_sample).argmax(1)
                    best_next_actions[memory_next_is_last_day[sample]] = ACTION_BUY
                    next_values = target_network.action_values(next_states_sample)[np.arange(batch_size), best_next_actions]
                    targets = memory_rewards[sample] + (1 - memory_done[sample]) * next_values
                    online_network.train_step(memory_states[sample], memory_actions[sample], targets)
                    if online_network.update_count % 200 == 0:
                        target_network = copy.deepcopy(online_network)
    return online_network


def greedy_purchase_days(network: QNetwork, environment: EntryTimingEnvironment, start_days: np.ndarray) -> np.ndarray:
    purchase_days = start_days + environment.window_days - 1
    decided = np.zeros(len(start_days), bool)
    for days_elapsed in range(environment.window_days - 1):
        undecided = np.where(~decided)[0]
        values = network.action_values(environment.states(start_days[undecided], days_elapsed))
        buys_now = undecided[values[:, ACTION_BUY] >= values[:, ACTION_WAIT]]
        purchase_days[buys_now] = start_days[buys_now] + days_elapsed
        decided[buys_now] = True
    return purchase_days


def bootstrap_mean_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 5_000) -> tuple:
    means = rng.choice(values, (draws, len(values))).mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def evaluate_entry_timing(ticker: str, closes: pd.Series, settings: AnalysisSettings) -> dict:
    signals = compute_entry_signals(closes)
    prices = closes.reindex(signals.index).values
    window = settings.entry_window_days
    total_days = len(prices)
    test_start = int(total_days * (1 - settings.test_period_fraction))
    training_end = test_start - window

    signal_mean = signals.iloc[:training_end].mean()
    signal_std = signals.iloc[:training_end].std().replace(0, 1)
    normalized = ((signals - signal_mean) / signal_std).clip(-5, 5).values
    environment = EntryTimingEnvironment(prices, normalized, window)

    network = train_entry_dqn(environment, np.arange(0, training_end - window), settings, settings.random_seed)

    rng = np.random.default_rng(settings.random_seed + 1)
    test_start_days = np.arange(test_start, total_days - window, window)
    dqn_days = greedy_purchase_days(network, environment, test_start_days)
    random_days = test_start_days + rng.integers(window, size=len(test_start_days))
    window_average = np.array([prices[day:day + window].mean() for day in test_start_days])
    savings = {
        "DQN": environment.savings_pct(test_start_days, dqn_days),
        "Comprar no 1º dia": np.zeros(len(test_start_days)),
        "Preço médio da janela": 100 * (prices[test_start_days] - window_average) / prices[test_start_days],
        "Dia aleatório": environment.savings_pct(test_start_days, random_days),
    }
    summary_rows = {}
    for name, values in savings.items():
        lower, upper = bootstrap_mean_interval(values, rng)
        summary_rows[name] = {
            "economia média (%)": values.mean(), "IC 95% inferior": lower,
            "IC 95% superior": upper, "% janelas melhores que o 1º dia": (values > 0).mean(),
        }
    summary = pd.DataFrame(summary_rows).T
    p_value_vs_first_day = stats.ttest_1samp(savings["DQN"], 0).pvalue
    p_value_vs_average = stats.ttest_1samp(savings["DQN"] - savings["Preço médio da janela"], 0).pvalue
    beats_benchmarks = (
        savings["DQN"].mean() > 0 and p_value_vs_first_day < 0.05
        and savings["DQN"].mean() > savings["Preço médio da janela"].mean() and p_value_vs_average < 0.05
    )

    today_state = np.hstack([normalized[-1], [1.0]])[None, :]
    wait_value, buy_value = network.action_values(today_state)[0]
    return {
        "summary": summary, "test_windows": len(test_start_days), "average_purchase_day": float((dqn_days - test_start_days).mean()),
        "p_value_vs_first_day": p_value_vs_first_day, "p_value_vs_average": p_value_vs_average,
        "beats_benchmarks": bool(beats_benchmarks), "today_action": "comprar hoje" if buy_value >= wait_value else "esperar",
        "today_expected_gain_from_waiting_pct": float(wait_value - buy_value),
        "test_period_start": signals.index[test_start].date(), "network": network,
    }
