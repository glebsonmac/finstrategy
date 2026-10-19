"""Coleta diária via MT5 (simulador/demo), em dois CSVs no estilo log: cada rodada só
**anexa** linhas embaixo das anteriores (nunca sobrescreve), pensado para rodar uma
vez por dia, por vários dias, até acumular histórico suficiente para o GARCH Monte
Carlo (`pricing/garch_montecarlo.py`) e para o Black-Scholes (`pricing/black_scholes.py`).

Cada linha é o **fechamento do dia** — não um preço intradiário. Dois arquivos:

- `acoes_diario.csv`: ticker + preço de fechamento (MT5) + indicadores fundamentalistas
  (Yahoo — o MT5 não tem fundamentalista, ver `data/mt5_provider.py`).
- `opcoes_diario.csv`: cada call/put líquida do vencimento mais próximo disponível no
  MT5, com prêmio/strike/vencimento/preço do ativo-objeto naquele dia já na mesma
  linha — dá para rodar Black-Scholes direto, sem precisar cruzar os dois CSVs.

Depois de cada rodada, os dois CSVs são reordenados por `date` (crescente) via
`csv_store.sort_csv_by` — ficam sempre legíveis mesmo se a coleta rodar fora de ordem
em dias diferentes. É puro "append" (sem deduplicar): rodar duas vezes no mesmo dia
grava duas linhas para aquele dia — rode no máximo uma vez por dia por ticker.

Por padrão, cada rodada grava só o fechamento **mais recente** (o uso diário normal).
Para popular o log de uma vez com vários anos de histórico (`run_daily_collection(...,
years_back=2)` ou `--years-back 2` na CLI), use `collect_stock_history_rows` — o MT5
guarda histórico de preço em cache no terminal, então isso não depende do mercado
estar aberto nem de rodar dia após dia.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import fundamentals_provider
from ..data.errors import MarketDataError
from . import csv_store

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_STORE_ROOT = PROJECT_ROOT / "data_store"

STOCK_LOG_COLUMNS = [
    "date", "ticker", "close", "sector", "industry", "pe_ratio", "forward_pe",
    "price_to_book", "roe", "debt_to_equity", "dividend_yield_ttm", "revenue_growth_yoy",
    "free_cash_flow", "market_cap", "profit_margin", "status", "collected_at",
]
OPTION_LOG_COLUMNS = [
    "date", "ticker", "option_type", "contract_symbol", "strike", "expiration",
    "underlying_price", "premium", "quote_source", "bid", "ask", "last_price",
    "implied_volatility", "open_interest", "volume", "collected_at",
]


def stock_log_path(data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT) -> Path:
    return Path(data_store_root) / "mt5" / "acoes_diario.csv"


def option_log_path(data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT) -> Path:
    return Path(data_store_root) / "mt5" / "opcoes_diario.csv"


def collect_stock_row(ticker: str, provider, lookback_days: int = 7) -> dict:
    """Uma linha: fechamento mais recente disponível no MT5 + fundamentalista do
    Yahoo. Nunca lança — falha vira `status`, para um ticker ruim não derrubar a
    rodada inteira (mesmo padrão defensivo do resto do coletor, ver `overnight.py`).

    `lookback_days` busca uma pequena janela (não só "hoje") para não falhar em
    fins de semana/feriados — o fechamento usado é sempre o último da janela.
    """
    collected_at = pd.Timestamp.now().isoformat(timespec="seconds")
    row = {"ticker": ticker, "collected_at": collected_at}
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        closes = provider.download_price_history(ticker, start_date=start)
        if closes.empty:
            raise MarketDataError("sem fechamento recente")
        row["date"] = pd.Timestamp(closes.index[-1]).strftime("%Y-%m-%d")
        row["close"] = float(closes.iloc[-1])
        row["status"] = "ok"
    except MarketDataError as reason:
        row["date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        row["status"] = f"sem preço: {reason}"
        return row
    except Exception as unexpected:
        row["date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        row["status"] = f"erro inesperado no preço: {unexpected}"
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
        # fundamentalista ausente não derruba a linha — o preço (o essencial) já foi gravado
        row["status"] = f"ok (sem fundamentalista: {unexpected})"
    return row


def collect_stock_history_rows(ticker: str, provider, years_back: float) -> list[dict]:
    """Backfill de preço: uma linha por fechamento diário dos últimos `years_back`
    anos (ex. `years_back=2` pega ~2 anos de histórico numa rodada só), em vez de
    esperar o coletor incremental rodar um dia de cada vez. Útil para popular o log
    de uma vez quando o MT5/corretor tem histórico de sobra em cache.

    Fundamentalista (Yahoo) só é anexado na **linha mais recente** — não existe
    fundamentalista histórico diário nesta pipeline (P/L, ROE etc. são um retrato de
    hoje, não variam dia a dia); preencher todas as linhas repetiria o mesmo valor e
    daria a entender, de forma enganosa, que o fundamentalista de anos atrás era
    igual ao de hoje. Nas linhas antigas, essas colunas ficam em branco.

    Nunca lança — falha no preço vira uma única linha de status (mesmo padrão de
    `collect_stock_row`)."""
    collected_at = pd.Timestamp.now().isoformat(timespec="seconds")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d")
    try:
        closes = provider.download_price_history(ticker, start_date=start)
        if closes.empty:
            raise MarketDataError("sem histórico de preços no período pedido")
    except MarketDataError as reason:
        return [{"date": pd.Timestamp.today().strftime("%Y-%m-%d"), "ticker": ticker,
                  "status": f"sem preço: {reason}", "collected_at": collected_at}]
    except Exception as unexpected:
        return [{"date": pd.Timestamp.today().strftime("%Y-%m-%d"), "ticker": ticker,
                  "status": f"erro inesperado no preço: {unexpected}", "collected_at": collected_at}]

    rows = [
        {"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "ticker": ticker, "close": float(close),
         "status": "ok", "collected_at": collected_at}
        for date, close in closes.items()
    ]
    try:
        snapshot = fundamentals_provider.get_fundamental_snapshot(ticker)
        rows[-1].update({
            "sector": snapshot.sector, "industry": snapshot.industry, "pe_ratio": snapshot.pe_ratio,
            "forward_pe": snapshot.forward_pe, "price_to_book": snapshot.price_to_book, "roe": snapshot.roe,
            "debt_to_equity": snapshot.debt_to_equity, "dividend_yield_ttm": snapshot.dividend_yield_ttm,
            "revenue_growth_yoy": snapshot.revenue_growth_yoy, "free_cash_flow": snapshot.free_cash_flow,
            "market_cap": snapshot.market_cap, "profit_margin": snapshot.profit_margin,
        })
    except Exception as unexpected:
        rows[-1]["status"] = f"ok (sem fundamentalista: {unexpected})"
    return rows


def collect_option_rows(ticker: str, provider, underlying_price: float, date_str: str) -> list[dict]:
    """Uma linha por call/put líquida do vencimento mais próximo disponível no MT5
    para esse ticker. Nunca lança — ticker sem opções no MT5 só vira lista vazia
    (mesma decisão já tomada para o Yahoo em `strategy/protective_put.py`: "não
    elegível" não é um bug)."""
    collected_at = pd.Timestamp.now().isoformat(timespec="seconds")
    try:
        expirations = provider.list_option_expirations(ticker)
    except MarketDataError:
        return []
    if not expirations:
        return []
    expiration = min(expirations)  # vencimento mais próximo disponível no MT5

    rows = []
    for option_type, downloader in (("CALL", provider.download_call_chain), ("PUT", provider.download_put_chain)):
        try:
            chain = downloader(ticker, expiration)
        except MarketDataError:
            continue
        for _, contract in chain.iterrows():
            bid, ask = float(contract.get("bid", 0) or 0), float(contract.get("ask", 0) or 0)
            last_price = float(contract.get("lastPrice", 0) or 0)
            has_live_quote = bid > 0 and ask > 0
            premium = ask if has_live_quote else last_price
            rows.append({
                "date": date_str, "ticker": ticker, "option_type": option_type,
                "contract_symbol": contract.get("contractSymbol", ""),
                "strike": float(contract["strike"]), "expiration": expiration.strftime("%Y-%m-%d"),
                "underlying_price": underlying_price, "premium": premium,
                "quote_source": "ask ao vivo" if has_live_quote else "último negócio",
                "bid": bid, "ask": ask, "last_price": last_price,
                "implied_volatility": contract.get("impliedVolatility"),
                "open_interest": int(contract.get("openInterest", 0) or 0),
                "volume": int(contract.get("volume", 0) or 0),
                "collected_at": collected_at,
            })
    return rows


def run_daily_collection(
    tickers: list[str], provider, data_store_root: Path | str = DEFAULT_DATA_STORE_ROOT,
    log=print, years_back: float | None = None,
) -> tuple[Path, Path]:
    """Coleta todos os tickers e **anexa** (nunca sobrescreve) nos dois CSVs de log;
    reordena os dois por data ao final. Devolve os caminhos dos dois CSVs.

    Por padrão (`years_back=None`) grava só o fechamento mais recente de cada
    ticker — o uso do dia a dia. Com `years_back` definido (ex. `2.0`), faz um
    backfill de preço dos últimos N anos numa rodada só (ver
    `collect_stock_history_rows`); as opções continuam sendo só a cadeia atual
    (não existe cadeia de opções histórica no MT5 — vira a última linha de preço).
    """
    stock_rows, option_rows = [], []
    for index, ticker in enumerate(tickers, start=1):
        log(f"[{index}/{len(tickers)}] {ticker}...")
        ticker_rows = collect_stock_history_rows(ticker, provider, years_back) if years_back else [collect_stock_row(ticker, provider)]
        stock_rows.extend(ticker_rows)
        latest_row = ticker_rows[-1]
        if years_back:
            log(f"  -> {len(ticker_rows)} dia(s) de histórico")
        if latest_row["status"].startswith("ok"):
            option_rows.extend(collect_option_rows(ticker, provider, latest_row["close"], latest_row["date"]))
        else:
            log(f"  -> sem preço: {latest_row['status']}")

    stocks_path = stock_log_path(data_store_root)
    options_path = option_log_path(data_store_root)
    csv_store.append_rows(stocks_path, [{column: row.get(column, "") for column in STOCK_LOG_COLUMNS} for row in stock_rows])
    if option_rows:
        csv_store.append_rows(options_path, [{column: row.get(column, "") for column in OPTION_LOG_COLUMNS} for row in option_rows])
    csv_store.sort_csv_by(stocks_path, "date")
    csv_store.sort_csv_by(options_path, "date")

    ok_count = sum(1 for row in stock_rows if row["status"].startswith("ok"))
    log(f"{ok_count}/{len(stock_rows)} ações coletadas, {len(option_rows)} linhas de opção.")
    log(f"Log de ações (append): {stocks_path}")
    log(f"Log de opções (append): {options_path}")
    return stocks_path, options_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Coleta diária via MT5 (simulador/demo): anexa fechamento+fundamentalista e "
                     "cadeia de opções líquida de cada ticker em dois CSVs de log (nunca sobrescreve)."
    )
    parser.add_argument("--tickers", type=str, required=True, help="Tickers separados por vírgula, no formato de símbolo do seu MT5 (ex.: AAPL,DUK,NEE).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_DATA_STORE_ROOT), help="Diretório raiz do data_store (padrão: data_store/).")
    parser.add_argument("--login", type=str, default=None, help="Login da conta MT5 (opcional — por padrão usa o terminal já aberto e logado).")
    parser.add_argument("--password", type=str, default=None, help="Senha da conta MT5 (opcional).")
    parser.add_argument("--server", type=str, default=None, help="Servidor da corretora no MT5 (opcional).")
    parser.add_argument("--risk-free-symbol", type=str, default=None, help="Símbolo MT5 usado como taxa livre de risco (opcional).")
    parser.add_argument(
        "--years-back", type=float, default=None,
        help="Backfill: busca N anos de histórico de preço numa rodada só (ex. 2 = últimos 2 anos), "
             "em vez de só o fechamento mais recente. Use uma vez para popular o log; nas rodadas "
             "diárias seguintes, omita essa opção.",
    )
    args = parser.parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("Nenhum ticker em --tickers.")
        return 1

    from ..data.mt5_provider import MT5Provider

    with MT5Provider(login=args.login, password=args.password, server=args.server, risk_free_symbol=args.risk_free_symbol) as provider:
        print(f"Coletando {len(tickers)} ticker(s) via MT5: {', '.join(tickers)}"
              + (f" (backfill de {args.years_back} ano(s))" if args.years_back else ""))
        run_daily_collection(tickers, provider, data_store_root=Path(args.output_dir), years_back=args.years_back)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
