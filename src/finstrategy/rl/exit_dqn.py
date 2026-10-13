"""DQN de saída: já tenho a put, mantenho ou vendo agora? (plan.md, seções 3 e 4).

Reaproveita a arquitetura (MLP NumPy + Double DQN + replay buffer) de `rl/entry_dqn.py`
— só o ambiente, o estado e a recompensa são novos. Treina sobre trajetórias
simuladas pelo GARCH Monte Carlo já calibrado para o ticker (mesma vol/alpha/beta/dof
usada no menu de proteção), não sobre histórico real de preços de opção (que o Yahoo
não oferece dia a dia) — ver limitação registrada em plan.md, seção 3.

**Recompensa:** ao vender no dia `t`, o agente ganha
`valor_de_venda_hoje − valor_de_manter_até_o_vencimento_descontado`. Em trajetórias em
que a ação cai (a put protege), manter vale mais do que vender — a recompensa de
vender fica negativa, o que ensina o agente a **não abrir mão da proteção quando ela
é útil**. Em trajetórias em que a ação sobe ou fica estável (a put só perde valor-tempo),
vender antes vale mais — a recompensa de vender fica positiva. Isso é a tradução
operacional de "redução máxima de risco financeiro": o agente não está otimizando
retorno médio, está aprendendo a soltar a proteção só quando ela não é mais necessária.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..config_model import AnalysisSettings
from ..pricing.black_scholes import price_and_greeks_array
from ..pricing.garch_montecarlo import TRADING_DAYS_PER_YEAR, GarchParameters, simulate_price_paths
from ..strategy.protective_put import ProtectivePutPlan
from .entry_dqn import QNetwork, bootstrap_mean_interval

EXIT_STATE_NAMES = [
    "moneyness", "delta", "theta por dia", "vega", "P&L não realizado %", "gap IV vs. GARCH", "tempo restante (frac.)",
]
ACTION_HOLD, ACTION_SELL = 0, 1


def build_exit_states(
    price_paths: np.ndarray,
    plan: ProtectivePutPlan,
    risk_free_rate: float,
    dividend_yield: float,
    garch_annual_volatility: float,
    window_days: int,
) -> np.ndarray:
    """Estado em cada (caminho, dia): sinais de Black-Scholes + P&L não realizado.

    `price_paths`: shape (num_paths, window_days + 1), de `simulate_price_paths`.
    Devolve shape (num_paths, window_days + 1, 7).
    """
    volatility = max(plan.implied_volatility, 1e-4)
    num_paths = price_paths.shape[0]
    states = np.zeros((num_paths, window_days + 1, len(EXIT_STATE_NAMES)))
    for day in range(window_days + 1):
        spot = price_paths[:, day]
        days_remaining = window_days - day
        time_years = days_remaining / TRADING_DAYS_PER_YEAR
        greeks = price_and_greeks_array("put", spot, plan.strike, time_years, risk_free_rate, volatility, dividend_yield)
        states[:, day, 0] = spot / plan.strike - 1
        states[:, day, 1] = greeks["delta"]
        states[:, day, 2] = greeks["theta"] / TRADING_DAYS_PER_YEAR
        states[:, day, 3] = greeks["vega"] / 100
        states[:, day, 4] = (greeks["price"] - plan.put_premium) / plan.put_premium
        states[:, day, 5] = volatility - garch_annual_volatility
        states[:, day, 6] = (window_days - day) / window_days
    return states


@dataclass
class ExitTrainingResult:
    network: QNetwork
    price_paths: np.ndarray
    states: np.ndarray


def train_exit_dqn(
    plan: ProtectivePutPlan,
    garch: GarchParameters,
    risk_free_rate: float,
    dividend_yield: float,
    window_days: int,
    settings: AnalysisSettings,
    seed: int,
    transaction_cost_pct: float = 0.0,
) -> ExitTrainingResult:
    rng = np.random.default_rng(seed)
    training_paths = simulate_price_paths(
        garch, plan.stock_price, window_days, risk_free_rate - dividend_yield, settings.monte_carlo_paths, rng
    )
    states = build_exit_states(training_paths, plan, risk_free_rate, dividend_yield, garch.current_annual_volatility, window_days)

    state_size = states.shape[2]
    online_network = QNetwork(state_size, 2, settings.exit_dqn_hidden_units, 1e-3, seed)
    target_network = copy.deepcopy(online_network)
    capacity, parallel_episodes, batch_size = 100_000, 64, 256
    memory_states = np.zeros((capacity, state_size))
    memory_next_states = np.zeros((capacity, state_size))
    memory_actions = np.zeros(capacity, int)
    memory_rewards = np.zeros(capacity)
    memory_done = np.zeros(capacity)
    memory_next_is_last_day = np.zeros(capacity, bool)
    write_position, stored = 0, 0
    num_paths = training_paths.shape[0]

    def discounted_hold_value(path_index: np.ndarray, day: int) -> np.ndarray:
        terminal_price = training_paths[path_index, -1]
        intrinsic = np.maximum(plan.strike - terminal_price, 0.0)
        days_remaining = window_days - day
        return intrinsic * np.exp(-risk_free_rate * days_remaining / TRADING_DAYS_PER_YEAR)

    for training_round in range(settings.exit_dqn_training_rounds):
        exploration_rate = max(0.05, 1 - training_round / (0.7 * settings.exit_dqn_training_rounds))
        path_index = rng.choice(num_paths, parallel_episodes)
        still_holding = np.ones(parallel_episodes, bool)
        for day in range(window_days):
            active = np.where(still_holding)[0]
            if not len(active):
                break
            active_paths = path_index[active]
            current_states = states[active_paths, day]
            greedy = online_network.action_values(current_states).argmax(1)
            explore = rng.random(len(active)) < exploration_rate
            actions = np.where(explore, rng.integers(2, size=len(active)), greedy)
            last_day = day == window_days - 1
            if last_day:
                actions[:] = ACTION_SELL
            sold = actions == ACTION_SELL

            sell_value = states[active_paths, day, 4] * plan.put_premium + plan.put_premium  # bs_price reconstruído do P&L%
            sell_value = sell_value * (1 - transaction_cost_pct)
            hold_value = discounted_hold_value(active_paths, day)
            rewards = np.where(sold, sell_value - hold_value, 0.0)

            next_day = day if last_day else day + 1
            next_states = states[active_paths, next_day]
            slots = (write_position + np.arange(len(active))) % capacity
            memory_states[slots], memory_actions[slots], memory_rewards[slots] = current_states, actions, rewards
            memory_next_states[slots], memory_done[slots] = next_states, sold.astype(float)
            memory_next_is_last_day[slots] = (day + 1 == window_days - 1)
            write_position = (write_position + len(active)) % capacity
            stored = min(stored + len(active), capacity)
            still_holding[active[sold]] = False

            if stored >= 1_000:
                for _ in range(4):
                    sample = rng.integers(stored, size=batch_size)
                    next_states_sample = memory_next_states[sample]
                    best_next_actions = online_network.action_values(next_states_sample).argmax(1)
                    best_next_actions[memory_next_is_last_day[sample]] = ACTION_SELL
                    next_values = target_network.action_values(next_states_sample)[np.arange(batch_size), best_next_actions]
                    targets = memory_rewards[sample] + (1 - memory_done[sample]) * next_values
                    online_network.train_step(memory_states[sample], memory_actions[sample], targets)
                    if online_network.update_count % 200 == 0:
                        target_network = copy.deepcopy(online_network)

    return ExitTrainingResult(network=online_network, price_paths=training_paths, states=states)


def greedy_exit_day(network: QNetwork, states: np.ndarray, window_days: int) -> np.ndarray:
    """Para cada caminho simulado, o dia em que a rede treinada decide vender."""
    num_paths = states.shape[0]
    exit_day = np.full(num_paths, window_days)
    decided = np.zeros(num_paths, bool)
    for day in range(window_days):
        undecided = np.where(~decided)[0]
        if not len(undecided):
            break
        values = network.action_values(states[undecided, day])
        sell_now = undecided[values[:, ACTION_SELL] >= values[:, ACTION_HOLD]]
        exit_day[sell_now] = day
        decided[sell_now] = True
    return exit_day


def evaluate_exit_timing(
    plan: ProtectivePutPlan,
    garch: GarchParameters,
    risk_free_rate: float,
    dividend_yield: float,
    window_days: int,
    settings: AnalysisSettings,
    fixed_rule_exit_day: int,
) -> dict:
    """Avalia a DQN de saída em trajetórias NUNCA usadas no treino, contra dois
    benchmarks: manter até o vencimento e a regra fixa de Black-Scholes (seção 3).
    Mesmo padrão de avaliação honesta do DQN de entrada (treino/teste separados,
    bootstrap de IC, p-valor)."""
    training = train_exit_dqn(plan, garch, risk_free_rate, dividend_yield, window_days, settings, settings.random_seed)

    test_rng = np.random.default_rng(settings.random_seed + 1)
    test_paths = simulate_price_paths(
        garch, plan.stock_price, window_days, risk_free_rate - dividend_yield, settings.monte_carlo_paths, test_rng
    )
    test_states = build_exit_states(test_paths, plan, risk_free_rate, dividend_yield, garch.current_annual_volatility, window_days)

    dqn_exit_days = greedy_exit_day(training.network, test_states, window_days)

    def realized_value(exit_days: np.ndarray) -> np.ndarray:
        # agrupa por dia de saída (no máximo window_days+1 valores distintos) para
        # repreçar com price_and_greeks_array em lote, em vez de um laço por caminho
        sell_prices = np.empty(len(exit_days), dtype=float)
        for day in np.unique(exit_days):
            mask = exit_days == day
            time_years = max(window_days - day, 0) / TRADING_DAYS_PER_YEAR
            greeks = price_and_greeks_array(
                "put", test_paths[mask, day], plan.strike, time_years, risk_free_rate,
                max(plan.implied_volatility, 1e-4), dividend_yield,
            )
            sell_prices[mask] = greeks["price"]
        return sell_prices

    dqn_value = realized_value(dqn_exit_days)
    hold_to_expiration_value = np.maximum(plan.strike - test_paths[:, -1], 0.0)
    fixed_rule_value = realized_value(np.full(len(test_paths), fixed_rule_exit_day))

    comparisons = {
        "DQN": dqn_value, "Manter até o vencimento": hold_to_expiration_value, "Regra fixa (Black-Scholes)": fixed_rule_value,
    }
    summary_rows = {}
    for name, values in comparisons.items():
        lower, upper = bootstrap_mean_interval(values, test_rng)
        summary_rows[name] = {"valor médio realizado": values.mean(), "IC 95% inferior": lower, "IC 95% superior": upper}

    p_value_vs_hold = stats.ttest_1samp(dqn_value - hold_to_expiration_value, 0).pvalue
    p_value_vs_fixed_rule = stats.ttest_1samp(dqn_value - fixed_rule_value, 0).pvalue
    beats_benchmarks = (
        dqn_value.mean() > hold_to_expiration_value.mean() and dqn_value.mean() > fixed_rule_value.mean()
        and p_value_vs_fixed_rule < 0.05
    )
    return {
        "summary": summary_rows, "average_exit_day": float(dqn_exit_days.mean()),
        "p_value_vs_hold": p_value_vs_hold, "p_value_vs_fixed_rule": p_value_vs_fixed_rule,
        "beats_benchmarks": bool(beats_benchmarks), "network": training.network,
    }
