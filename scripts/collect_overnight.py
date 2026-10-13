#!/usr/bin/env python3
"""Wrapper de linha de comando para a coleta noturna — lógica real em
`src/finstrategy/collector/overnight.py`.

Uso:
    python scripts/collect_overnight.py
    python scripts/collect_overnight.py --tickers AAPL,MSFT,DUK
    python scripts/collect_overnight.py --output-dir /caminho/alternativo

Pensado para rodar via cron/agendador, sem precisar do Streamlit aberto — só precisa
do venv do projeto ativado (ver README.md, seção "Coleta noturna (cron)").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finstrategy.collector.overnight import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
