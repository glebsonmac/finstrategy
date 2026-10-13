"""Coleta noturna de dados essenciais — um CSV por rodada, pensado para cron/agendador.

Diferente de `scheduler.run_forever` (que fica rodando o dia inteiro, só durante o
pregão, controlado pelo Streamlit), este módulo faz **uma rodada só** e sai: preço +
fundamentalista de cada ticker, tudo numa linha por ticker, num CSV só — fácil de
levar para outro computador ou abrir direto numa planilha. Não depende do app
Streamlit estar aberto; funciona com o mercado aberto ou fechado (fora do pregão,
pega o último preço negociado, igual ao resto do sistema).

Ordem de decisão de quais tickers coletar (sem passar `--tickers`):
1. `config/watchlist.yaml`, se já existir e tiver tickers;
2. senão, a semente das 20 maiores ações dos EUA (`config/us_top20_megacaps.yaml`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data import fundamentals_provider, symbol_search
from ..data.errors import MarketDataError
from ..data.watchlist import DEFAULT_WATCHLIST_PATH, load_watchlist, seed_tickers

PROJECT_ROOT = Path(__file__).resolve().parents[3]
US_TOP20_SEED_PATH = PROJECT_ROOT / "config" / "us_top20_megacaps.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_store" / "overnight"

ESSENTIAL_COLUMNS = [
    "ticker", "status", "name", "sector", "industry", "price", "currency", "exchange",
    "elegivel_derivativo", "pe_ratio", "forward_pe", "price_to_book", "roe", "debt_to_equity",
    "dividend_yield_ttm", "revenue_growth_yoy", "free_cash_flow", "market_cap", "profit_margin",
    "collected_at",
]


def resolve_tickers(explicit: list[str] | None, watchlist_path: Path | str = DEFAULT_WATCHLIST_PATH, seed_path: Path | str = US_TOP20_SEED_PATH) -> list[str]:
    """Decide quais tickers coletar: lista explícita > watchlist.yaml > semente top-20 EUA."""
    if explicit:
        return explicit
    watchlist = load_watchlist(watchlist_path)
    if watchlist:
        return [entry.symbol for entry in watchlist]
    seed = seed_tickers(seed_path)
    return [item["symbol"] for item in seed]


def collect_essential_row(ticker: str) -> dict:
    """Uma linha com o essencial de um ticker: cotação + indicadores fundamentalistas,
    numa linha só. Nunca lança exceção — falha vira `status`, para a rodada inteira
    não ser derrubada por um ticker problemático."""
    row = {"ticker": ticker, "collected_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    try:
        info = symbol_search.validate_symbol(ticker)
        row.update({
            "name": info.name, "price": info.current_price, "currency": info.currency,
            "exchange": info.exchange, "elegivel_derivativo": info.has_options, "status": "ok",
        })
    except MarketDataError as reason:
        row["status"] = f"sem cotação: {reason}"
        return row
    except Exception as unexpected:
        row["status"] = f"erro inesperado na cotação: {unexpected}"
        return row

    try:
        snapshot = fundamentals_provider.get_fundamental_snapshot(ticker)
        row.update({
            "sector": snapshot.sector, "industry": snapshot.industry, "pe_ratio": snapshot.pe_ratio,
            "forward_pe": snapshot.forward_pe, "price_to_book": snapshot.price_to_book, "roe": snapshot.roe,
            "debt_to_equity": snapshot.debt_to_equity, "dividend_yield_ttm": snapshot.dividend_yield_ttm,
            "revenue_growth_yoy": snapshot.revenue_growth_yoy, "free_cash_flow": snapshot.free_cash_flow,
            "market_cap": snapshot.market_cap, "profit_margin": snapshot.profit_margin,
        })
    except Exception as unexpected:
        # fundamentalista ausente não derruba a linha — a cotação já foi gravada
        row["status"] = f"ok (sem fundamentalista: {unexpected})"
    return row


def run_overnight_collection(tickers: list[str], output_dir: Path | str = DEFAULT_OUTPUT_DIR, log=print) -> Path:
    """Coleta todos os tickers e grava dois CSVs: um com a data de hoje (histórico de
    rodadas) e `latest.csv` (sempre sobrescrito — caminho estável para outro processo
    ou computador sempre achar a rodada mais recente)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, ticker in enumerate(tickers, start=1):
        log(f"[{index}/{len(tickers)}] coletando {ticker}...")
        try:
            rows.append(collect_essential_row(ticker))
        except Exception as unexpected:  # defesa extra: nunca deixa um ticker matar a rodada
            rows.append({"ticker": ticker, "collected_at": pd.Timestamp.now().isoformat(timespec="seconds"), "status": f"erro inesperado: {unexpected}"})
            log(f"  -> falhou: {unexpected}")

    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=ESSENTIAL_COLUMNS)
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    dated_path = output_dir / f"{today_str}_essencial.csv"
    frame.to_csv(dated_path, index=False)
    latest_path = output_dir / "latest.csv"
    frame.to_csv(latest_path, index=False)

    ok_count = int(frame["status"].str.startswith("ok").sum()) if "status" in frame and len(frame) else 0
    log(f"Salvo: {dated_path}")
    log(f"Também salvo (sempre sobrescrito): {latest_path}")
    log(f"{ok_count}/{len(frame)} tickers coletados com sucesso.")
    return dated_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coleta noturna de dados essenciais (preço + fundamentalista) em um CSV só, sem precisar do Streamlit aberto."
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Tickers separados por vírgula (ex.: AAPL,MSFT,DUK). Se omitido, usa config/watchlist.yaml "
             "se existir, senão a semente das 20 maiores ações dos EUA.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Diretório onde salvar os CSVs (padrão: data_store/overnight/).")
    args = parser.parse_args(argv)

    explicit = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    # passa os paths explicitamente (em vez de confiar nos defaults de resolve_tickers,
    # que ficam presos ao valor de import-time) para que testes consigam fazer
    # monkeypatch de DEFAULT_WATCHLIST_PATH/US_TOP20_SEED_PATH neste módulo
    tickers = resolve_tickers(explicit, watchlist_path=DEFAULT_WATCHLIST_PATH, seed_path=US_TOP20_SEED_PATH)
    if not tickers:
        print("Nenhum ticker para coletar (sem --tickers, watchlist vazia e semente ausente).")
        return 1

    print(f"Coletando dados essenciais de {len(tickers)} ticker(s): {', '.join(tickers)}")
    run_overnight_collection(tickers, output_dir=Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
