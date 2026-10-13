"""AnalysisSettings: parâmetros únicos do pipeline, carregáveis/salváveis em YAML.

Extraído e estendido a partir da célula de configuração do notebook
`put_de_protecao_DQN_montecarlos.ipynb` (seção 1). Os campos originais do notebook
não mudaram de nome nem de significado; os campos novos (prefixo `exit_`) suportam a
timing de saída do derivativo via Black-Scholes/DQN (ver plan.md, seções 3 e 4).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
SEED_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.example.yaml"


@dataclass
class AnalysisSettings:
    history_start_date: str = "2012-01-01"

    contracts_per_ticker: int = 1
    shares_per_ticker: dict = field(default_factory=dict)
    max_capital_per_ticker: Optional[float] = None
    shares_per_option_contract: int = 100

    target_protection_days: int = 45
    minimum_protection_days: int = 14
    strike_range_vs_price: tuple = (0.80, 1.02)
    max_acceptable_loss_pct: float = 0.08

    minimum_open_interest: int = 100
    maximum_bid_ask_spread_pct: float = 0.20

    stock_commission_pct: float = 0.0005
    option_commission_per_contract: float = 0.65

    monte_carlo_paths: int = 20_000

    entry_window_days: int = 10
    dqn_training_rounds: int = 120
    dqn_hidden_units: int = 64
    test_period_fraction: float = 0.30

    # Timing de saída do derivativo (novo, não existia no notebook original)
    exit_time_value_floor_pct: float = 0.25
    exit_dqn_training_rounds: int = 120
    exit_dqn_hidden_units: int = 64

    random_seed: int = 42

    def to_dict(self) -> dict:
        data = asdict(self)
        data["strike_range_vs_price"] = list(self.strike_range_vs_price)
        return data


def _coerce_tuples(data: dict) -> dict:
    data = dict(data)
    if "strike_range_vs_price" in data and data["strike_range_vs_price"] is not None:
        data["strike_range_vs_price"] = tuple(data["strike_range_vs_price"])
    return data


def load_settings(path: Path | str = DEFAULT_SETTINGS_PATH) -> AnalysisSettings:
    """Carrega settings.yaml; se não existir, parte da semente versionada (settings.example.yaml)."""
    path = Path(path)
    source = path if path.exists() else SEED_SETTINGS_PATH
    with open(source, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known_fields = {f.name for f in fields(AnalysisSettings)}
    raw = {key: value for key, value in raw.items() if key in known_fields}
    return AnalysisSettings(**_coerce_tuples(raw))


def save_settings(settings: AnalysisSettings, path: Path | str = DEFAULT_SETTINGS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(settings.to_dict(), handle, sort_keys=False, allow_unicode=True)
