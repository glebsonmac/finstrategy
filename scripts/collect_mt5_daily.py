#!/usr/bin/env python3
"""Wrapper de linha de comando para a coleta diária via MT5 — lógica real em
`src/finstrategy/collector/mt5_daily_log.py`.

Uso (precisa do terminal MT5 aberto e logado no simulador/demo):
    python scripts/collect_mt5_daily.py --tickers AAPL,DUK,NEE

Pensado para rodar uma vez por dia (ex. via agendador do Windows), por vários dias —
cada rodada só anexa linhas nos dois CSVs de log (data_store/mt5/), nunca sobrescreve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finstrategy.collector.mt5_daily_log import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
