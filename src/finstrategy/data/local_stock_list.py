"""Lista local de ações (S&P 500) para navegar/escolher sem bater na internet.

`symbol_search.search_symbols` busca ao vivo no Yahoo — útil, mas lento e repetitivo
quando o usuário só quer folhear/marcar ações conhecidas (pedido: "estou tendo que
buscar na internet toda vez"). Este módulo lê `config/us_stock_list.csv`, um arquivo
estático versionado (≈500 ações do S&P 500: ticker, nome, setor), e filtra em memória
— instantâneo, sem rede. A tela "Ações" do frontend oferece as duas formas lado a
lado: busca ao vivo (qualquer ação do Yahoo) e esta lista local (rápida, mas limitada
ao que está no CSV).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .base import SymbolMatch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIST_PATH = PROJECT_ROOT / "config" / "us_stock_list.csv"


@lru_cache(maxsize=4)
def _load(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name", "sector"])
    return pd.read_csv(path)


def load_all(path: Path | str = DEFAULT_LIST_PATH) -> pd.DataFrame:
    """Devolve a lista inteira (symbol, name, sector) — para a tabela de navegação."""
    return _load(path).copy()


def search_local_stocks(query: str, path: Path | str = DEFAULT_LIST_PATH, max_results: int = 50) -> list[SymbolMatch]:
    """Filtra a lista local por ticker ou nome (substring, sem diferenciar maiúsculas).
    Nunca toca na rede — devolve na hora, mesmo offline."""
    query = (query or "").strip().lower()
    table = _load(path)
    if table.empty:
        return []
    if query:
        mask = table["symbol"].str.lower().str.contains(query, na=False) | table["name"].str.lower().str.contains(query, na=False)
        table = table[mask]
    matches = []
    for _, row in table.head(max_results).iterrows():
        # reaproveita SymbolMatch (mesmo tipo da busca ao vivo) para o frontend poder
        # desenhar os dois resultados com o mesmo componente; `exchange` carrega o
        # setor aqui (não há bolsa no CSV local, e setor é mais útil para filtrar)
        matches.append(SymbolMatch(symbol=row["symbol"], name=row["name"], exchange=row.get("sector", "?"), quote_type="Equity"))
    return matches
