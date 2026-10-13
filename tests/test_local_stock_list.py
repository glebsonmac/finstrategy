import pandas as pd
import pytest

from finstrategy.data.local_stock_list import load_all, search_local_stocks


@pytest.fixture
def sample_list_path(tmp_path):
    path = tmp_path / "stocks.csv"
    pd.DataFrame({
        "symbol": ["AAPL", "DUK", "MSFT"],
        "name": ["Apple Inc.", "Duke Energy", "Microsoft Corp."],
        "sector": ["Information Technology", "Utilities", "Information Technology"],
    }).to_csv(path, index=False)
    # limpa o cache do lru_cache entre testes, já que cada teste usa um path diferente
    from finstrategy.data import local_stock_list
    local_stock_list._load.cache_clear()
    return path


def test_search_local_stocks_matches_by_symbol(sample_list_path):
    matches = search_local_stocks("AAPL", path=sample_list_path)
    assert [m.symbol for m in matches] == ["AAPL"]


def test_search_local_stocks_matches_by_name_case_insensitive(sample_list_path):
    matches = search_local_stocks("duke", path=sample_list_path)
    assert [m.symbol for m in matches] == ["DUK"]
    assert matches[0].name == "Duke Energy"
    assert matches[0].exchange == "Utilities"  # carrega o setor, não a bolsa


def test_search_local_stocks_partial_match(sample_list_path):
    matches = search_local_stocks("corp", path=sample_list_path)
    assert [m.symbol for m in matches] == ["MSFT"]


def test_search_local_stocks_empty_query_returns_nothing_filtered_but_capped(sample_list_path):
    matches = search_local_stocks("", path=sample_list_path, max_results=2)
    assert len(matches) == 2  # sem filtro, mas respeita max_results


def test_search_local_stocks_no_match_returns_empty_list(sample_list_path):
    assert search_local_stocks("nope-nothing-like-this", path=sample_list_path) == []


def test_search_local_stocks_missing_file_returns_empty_list(tmp_path):
    from finstrategy.data import local_stock_list
    local_stock_list._load.cache_clear()
    assert search_local_stocks("AAPL", path=tmp_path / "nope.csv") == []


def test_load_all_returns_full_table(sample_list_path):
    table = load_all(sample_list_path)
    assert len(table) == 3
    assert set(table.columns) == {"symbol", "name", "sector"}


def test_real_config_file_loads_and_has_expected_columns():
    from finstrategy.data.local_stock_list import DEFAULT_LIST_PATH
    table = load_all(DEFAULT_LIST_PATH)
    assert len(table) > 400  # S&P 500 completo
    assert {"symbol", "name", "sector"}.issubset(table.columns)
    assert "AAPL" in set(table["symbol"])
