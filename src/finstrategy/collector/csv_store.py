"""Escrita append-safe em CSV, particionada por diretório/arquivo.

Usado tanto pela coleta contínua do pregão (seção 6 do plano) quanto por qualquer
outro ponto do sistema que precise registrar uma linha de dado em CSV sem reescrever
o arquivo inteiro (watchlist ao adicionar ticker, decisões dos DQNs, metadados de
relatório). Um `fcntl`/lock de SO não é usado de propósito: cada processo deste
projeto escreve em arquivos próprios (um arquivo por ticker/dia), então a contenção
real é baixa; o lock aqui é só para proteger contra duas threads do mesmo processo
escrevendo ao mesmo tempo.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

_write_lock = threading.Lock()


def append_row(csv_path: Path | str, row: dict) -> None:
    append_rows(csv_path, [row])


def append_rows(csv_path: Path | str, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    with _write_lock:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        frame.to_csv(csv_path, mode="a", header=not file_exists, index=False)


def read_csv(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def sort_csv_by(csv_path: Path | str, by: str) -> None:
    """Reordena um CSV existente por uma coluna (ex.: `by="date"`, crescente por
    padrão com `sort_values`). Usado pelos logs "sempre anexa embaixo" (ver
    `collector/mt5_daily_log.py`) para ficarem em ordem cronológica mesmo coletando
    fora de ordem em dias diferentes. Não faz nada se o arquivo não existir, estiver
    vazio ou não tiver a coluna — seguro de chamar sempre, mesmo na primeira rodada."""
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    frame = pd.read_csv(csv_path)
    if frame.empty or by not in frame.columns:
        return
    frame = frame.sort_values(by, kind="stable").reset_index(drop=True)
    with _write_lock:
        frame.to_csv(csv_path, index=False)


def market_csv_path(data_store_root: Path | str, ticker: str, as_of_date) -> Path:
    date_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    safe_ticker = ticker.replace("/", "_")
    return Path(data_store_root) / "market" / safe_ticker / f"{date_str}.csv"


def model_runs_csv_path(data_store_root: Path | str, as_of_date) -> Path:
    date_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    return Path(data_store_root) / "model_runs" / f"{date_str}.csv"


def fundamentals_csv_path(data_store_root: Path | str, ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "_")
    return Path(data_store_root) / "fundamentals" / f"{safe_ticker}.csv"


def reports_csv_path(data_store_root: Path | str, as_of_date) -> Path:
    date_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    return Path(data_store_root) / "reports" / f"{date_str}.csv"


def report_pdf_path(data_store_root: Path | str, generated_at) -> Path:
    """Um arquivo por geração (data + hora), não por dia — gerar duas vezes no mesmo
    dia não deve sobrescrever o relatório anterior."""
    timestamp_str = pd.Timestamp(generated_at).strftime("%Y-%m-%d_%H-%M-%S")
    return Path(data_store_root) / "reports" / f"{timestamp_str}.pdf"
