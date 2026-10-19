import pandas as pd
import pytest

from finstrategy.collector.csv_store import (
    append_row,
    append_rows,
    fundamentals_csv_path,
    market_csv_path,
    model_runs_csv_path,
    read_csv,
    reports_csv_path,
    sort_csv_by,
)


def test_append_row_creates_file_with_header(tmp_path):
    csv_path = tmp_path / "out.csv"
    append_row(csv_path, {"a": 1, "b": "x"})
    frame = read_csv(csv_path)
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 1


def test_append_row_twice_appends_without_duplicating_header(tmp_path):
    csv_path = tmp_path / "out.csv"
    append_row(csv_path, {"a": 1})
    append_row(csv_path, {"a": 2})
    frame = read_csv(csv_path)
    assert list(frame["a"]) == [1, 2]
    with open(csv_path) as handle:
        lines = handle.readlines()
    assert lines[0].strip() == "a"  # só um cabeçalho, não um por linha adicionada


def test_append_rows_batch(tmp_path):
    csv_path = tmp_path / "out.csv"
    append_rows(csv_path, [{"a": 1}, {"a": 2}, {"a": 3}])
    frame = read_csv(csv_path)
    assert len(frame) == 3


def test_append_rows_empty_list_does_not_create_file(tmp_path):
    csv_path = tmp_path / "out.csv"
    append_rows(csv_path, [])
    assert not csv_path.exists()


def test_read_csv_missing_file_returns_empty_dataframe(tmp_path):
    frame = read_csv(tmp_path / "nope.csv")
    assert frame.empty


def test_market_csv_path_partitions_by_ticker_and_day(tmp_path):
    path = market_csv_path(tmp_path, "TAEE11.SA", pd.Timestamp("2026-10-13"))
    assert path == tmp_path / "market" / "TAEE11.SA" / "2026-10-13.csv"


def test_market_csv_path_sanitizes_slash_in_ticker(tmp_path):
    path = market_csv_path(tmp_path, "BRK/B", pd.Timestamp("2026-10-13"))
    assert "/" not in path.name and "/" not in path.parent.name


def test_model_runs_and_reports_csv_path_by_day(tmp_path):
    day = pd.Timestamp("2026-10-13")
    assert model_runs_csv_path(tmp_path, day) == tmp_path / "model_runs" / "2026-10-13.csv"
    assert reports_csv_path(tmp_path, day) == tmp_path / "reports" / "2026-10-13.csv"


def test_fundamentals_csv_path_by_ticker(tmp_path):
    assert fundamentals_csv_path(tmp_path, "AAPL") == tmp_path / "fundamentals" / "AAPL.csv"


def test_sort_csv_by_reorders_rows_ascending(tmp_path):
    csv_path = tmp_path / "log.csv"
    append_rows(csv_path, [{"date": "2026-10-15", "ticker": "DUK"}])
    append_rows(csv_path, [{"date": "2026-10-13", "ticker": "NEE"}])
    append_rows(csv_path, [{"date": "2026-10-14", "ticker": "AEP"}])
    sort_csv_by(csv_path, "date")
    frame = read_csv(csv_path)
    assert list(frame["date"]) == ["2026-10-13", "2026-10-14", "2026-10-15"]
    assert list(frame["ticker"]) == ["NEE", "AEP", "DUK"]


def test_sort_csv_by_keeps_relative_order_within_same_date(tmp_path):
    csv_path = tmp_path / "log.csv"
    append_rows(csv_path, [{"date": "2026-10-13", "ticker": "DUK"}, {"date": "2026-10-13", "ticker": "NEE"}])
    sort_csv_by(csv_path, "date")
    frame = read_csv(csv_path)
    assert list(frame["ticker"]) == ["DUK", "NEE"]


def test_sort_csv_by_does_nothing_when_file_missing(tmp_path):
    sort_csv_by(tmp_path / "nope.csv", "date")  # não deve lançar


def test_sort_csv_by_does_nothing_when_column_absent(tmp_path):
    csv_path = tmp_path / "log.csv"
    append_rows(csv_path, [{"ticker": "DUK"}])
    sort_csv_by(csv_path, "date")  # coluna "date" não existe
    frame = read_csv(csv_path)
    assert list(frame["ticker"]) == ["DUK"]
