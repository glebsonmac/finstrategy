"""Persistência de modelos DQN treinados — pickle + índice CSV (plan.md, seção 10.3).

Hoje a DQN treina do zero toda vez que a tela "Modelos" é usada, e tudo se perde ao
trocar de tela. Este módulo salva o modelo treinado (pesos, hiperparâmetros, métricas
de avaliação completas) para poder **carregar depois sem retreinar** e comparar
execuções ao longo do tempo (mesmo ticker, dias diferentes, configurações diferentes).

`evaluation` é o dict devolvido por `rl.entry_dqn.evaluate_entry_timing` ou
`rl.exit_dqn.evaluate_exit_timing` — já contém a rede treinada (`evaluation["network"]`,
um `QNetwork`), por isso não precisamos de um parâmetro de rede separado: pickle lida
bem com `QNetwork` (só arrays NumPy e escalares Python, nada de lambda/arquivo aberto).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import pandas as pd

from ..config_model import AnalysisSettings
from ..data.watchlist import DEFAULT_DATA_STORE_ROOT

ModelType = Literal["entry", "exit"]

DEFAULT_MODELS_DIR = DEFAULT_DATA_STORE_ROOT / "models"

INDEX_COLUMNS = ["ticker", "model_type", "trained_at", "filename", "metrica_principal", "valor_principal", "bateu_benchmark", "sinal_hoje"]


def _models_dir(models_dir: Path | str | None) -> Path:
    path = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(models_dir: Path | str | None) -> Path:
    return _models_dir(models_dir) / "index.csv"


def _summary_row(summary, name: str) -> dict:
    """`summary` pode ser um DataFrame (DQN de entrada) ou um dict de dicts (DQN de
    saída) — normaliza pra um dict só, pela linha `name`."""
    if isinstance(summary, pd.DataFrame):
        return summary.loc[name].to_dict()
    return dict(summary[name])


def _headline_metrics(model_type: ModelType, evaluation: dict) -> dict:
    dqn_row = _summary_row(evaluation["summary"], "DQN")
    if model_type == "entry":
        return {
            "metrica_principal": "economia média (%)",
            "valor_principal": float(dqn_row["economia média (%)"]),
            "bateu_benchmark": bool(evaluation["beats_benchmarks"]),
            "sinal_hoje": evaluation.get("today_action", ""),
        }
    return {
        "metrica_principal": "valor médio realizado",
        "valor_principal": float(dqn_row["valor médio realizado"]),
        "bateu_benchmark": bool(evaluation["beats_benchmarks"]),
        "sinal_hoje": "",
    }


def save_model(ticker: str, model_type: ModelType, evaluation: dict, settings: AnalysisSettings, models_dir: Path | str | None = None) -> Path:
    """Salva o modelo treinado (`evaluation["network"]` incluso) + metadados num
    `.pkl`, e acrescenta um resumo em `index.csv` — a tela "Modelos" lista a partir
    do índice, sem precisar desempacotar cada arquivo pra mostrar a tabela."""
    trained_at = pd.Timestamp.now()
    timestamp = trained_at.strftime("%Y%m%dT%H%M%S")
    filename = f"{ticker.replace('/', '_')}_{model_type}_{timestamp}.pkl"
    path = _models_dir(models_dir) / filename

    payload = {
        "ticker": ticker, "model_type": model_type, "trained_at": trained_at.isoformat(timespec="seconds"),
        "hyperparameters": settings.to_dict(), "evaluation": evaluation,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    headline = _headline_metrics(model_type, evaluation)
    index_row = pd.DataFrame([{"ticker": ticker, "model_type": model_type, "trained_at": payload["trained_at"], "filename": filename, **headline}])
    index_path = _index_path(models_dir)
    index_row.to_csv(index_path, mode="a", header=not index_path.exists(), index=False)
    return path


def list_models(models_dir: Path | str | None = None) -> pd.DataFrame:
    """Índice dos modelos salvos, mais recente primeiro. Nunca abre os `.pkl`."""
    index_path = _index_path(models_dir)
    if not index_path.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    # keep_default_na=False: toda linha sempre preenche todas as colunas (inclusive
    # "sinal_hoje", vazio de propósito pra modelos de saída) — sem isso, pandas lê o
    # campo vazio de volta como NaN em vez de "", quebrando comparação de string.
    index = pd.read_csv(index_path, keep_default_na=False)
    return index.sort_values("trained_at", ascending=False).reset_index(drop=True)


def load_model(filename: str, models_dir: Path | str | None = None) -> dict:
    """Carrega de volta o payload salvo por `save_model` — inclui a rede treinada
    pronta pra usar (`payload["evaluation"]["network"].action_values(...)`), sem
    precisar retreinar nada."""
    path = _models_dir(models_dir) / filename
    with open(path, "rb") as handle:
        return pickle.load(handle)
