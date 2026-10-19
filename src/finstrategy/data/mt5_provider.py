"""Provedor de dados via MetaTrader5 (terminal/simulador aberto localmente).

Implementa `MarketDataProvider` (ver `data/base.py`) usando o pacote `MetaTrader5`,
que só instala em Python nativo do Windows e só funciona com o terminal MT5 aberto
(conta real, demo ou simulador). Instale com `pip install -e ".[mt5]"` — não é
dependência padrão do projeto porque não instala fora do Windows (ver `pyproject.toml`).

**Conexão.** Por padrão, `MT5Provider()` só se conecta ao terminal MT5 já aberto e
logado (`mt5.initialize()` sem credenciais) — é o jeito mais simples para um
simulador/demo que você abre manualmente antes de rodar o script. Para login
automático, passe `login`/`password`/`server`/`terminal_path` no construtor, ou
defina as variáveis de ambiente `MT5_LOGIN`/`MT5_PASSWORD`/`MT5_SERVER`/`MT5_PATH`.

**O que NÃO dá para confiar igual ao Yahoo** (ao contrário de `yahoo_provider.py`,
que foi testado ponta a ponta com dados reais, isto aqui não pôde ser testado contra
um terminal MT5 de verdade neste ambiente — valide no seu simulador antes de confiar):
- **Sem dado fundamentalista.** O MT5 não expõe P/L, ROE, dívida/patrimônio etc. —
  `get_trailing_dividend_yield` aqui devolve sempre `0.0`. O dado fundamentalista
  completo do coletor diário (`collector/mt5_daily_log.py`) continua vindo do Yahoo
  (`data/fundamentals_provider.py`), que já é testado — é o único jeito razoável hoje.
- **Sem taxa livre de risco padrão.** Cada corretora expõe (ou não) um instrumento
  diferente para isso. `get_risk_free_rate` lê um símbolo configurável
  (`risk_free_symbol`, ex. um CFD de título do tesouro); se não tiver um no seu
  simulador, passe a taxa manualmente para quem precisa dela.
- **Opções dependem 100% do corretor.** Nem todo corretor/demo MT5 lista opções, e a
  forma de nomear os símbolos varia muito entre eles. `list_option_expirations`/
  `download_put_chain`/`download_call_chain` procuram símbolos cujo nome contém o
  ticker (`option_group_pattern`, ajustável) e filtram pelos campos `option_right`/
  `option_strike`/`expiration_time` que o MT5 define para instrumentos de opção — use
  `inspect_symbol(...)` para conferir, num símbolo real do seu simulador, se esses
  campos vêm preenchidos antes de confiar no resultado.
"""
from __future__ import annotations

import os

import pandas as pd

from .errors import MarketDataError


class MT5Provider:
    """Implementa `MarketDataProvider` via o pacote `MetaTrader5`.

    `mt5_module` é injetável de propósito (em vez de `import MetaTrader5` fixo no
    topo do arquivo): permite testar toda a lógica com um dublê em qualquer sistema
    operacional, já que o pacote real só instala no Windows. Em uso normal, deixe
    `mt5_module=None` — o pacote de verdade é importado automaticamente.
    """

    def __init__(
        self,
        login: int | str | None = None,
        password: str | None = None,
        server: str | None = None,
        terminal_path: str | None = None,
        risk_free_symbol: str | None = None,
        option_group_pattern: str = "*{ticker}*",
        mt5_module=None,
    ):
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "Pacote 'MetaTrader5' não instalado (pip install -e \".[mt5]\") — só "
                    "funciona em Python nativo do Windows, com o terminal MT5 instalado."
                ) from exc
        self._mt5 = mt5_module

        login = login if login is not None else os.environ.get("MT5_LOGIN")
        password = password or os.environ.get("MT5_PASSWORD")
        server = server or os.environ.get("MT5_SERVER")
        terminal_path = terminal_path or os.environ.get("MT5_PATH")

        kwargs: dict = {}
        if terminal_path:
            kwargs["path"] = terminal_path
        if login:
            kwargs.update(login=int(login), password=password, server=server)
        initialized = self._mt5.initialize(**kwargs) if kwargs else self._mt5.initialize()
        if not initialized:
            raise MarketDataError(
                f"falha ao conectar no terminal MT5 ({self._last_error()}) — abra o MT5 "
                "e faça login na conta/simulador antes de rodar."
            )

        self.risk_free_symbol = risk_free_symbol
        self.option_group_pattern = option_group_pattern

    def __enter__(self) -> "MT5Provider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self._mt5.shutdown()

    def _last_error(self) -> str:
        try:
            return str(self._mt5.last_error())
        except Exception:
            return "sem detalhe do erro"

    def _ensure_symbol(self, symbol: str) -> None:
        if not self._mt5.symbol_select(symbol, True):
            raise MarketDataError(f"símbolo '{symbol}' não encontrado/selecionável no MT5 ({self._last_error()})")

    def inspect_symbol(self, symbol: str) -> dict:
        """Dump cru de `symbol_info(symbol)` — use para conferir, contra um símbolo real
        do seu simulador, quais campos (`option_right`, `option_strike`,
        `expiration_time`, open interest...) vêm preenchidos antes de confiar na cadeia
        de opções. Não é usado pelo resto do pipeline, é uma ferramenta de depuração."""
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise MarketDataError(f"símbolo '{symbol}' não encontrado no MT5 ({self._last_error()})")
        return info._asdict()

    # --- MarketDataProvider ---

    def download_price_history(self, ticker: str, start_date: str) -> pd.Series:
        self._ensure_symbol(ticker)
        start = pd.Timestamp(start_date).to_pydatetime()
        end = pd.Timestamp.now().to_pydatetime()
        rates = self._mt5.copy_rates_range(ticker, self._mt5.TIMEFRAME_D1, start, end)
        if rates is None or len(rates) == 0:
            raise MarketDataError(f"sem histórico de preços no MT5 para '{ticker}' ({self._last_error()})")
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s")
        closes = frame.set_index("time")["close"].astype(float)
        return closes[closes > 0]

    def get_current_stock_price(self, ticker: str, fallback_price: float) -> float:
        try:
            self._ensure_symbol(ticker)
            tick = self._mt5.symbol_info_tick(ticker)
        except MarketDataError:
            return fallback_price
        if tick is None:
            return fallback_price
        price = tick.last if tick.last > 0 else tick.bid
        return float(price) if price and price > 0 else fallback_price

    def get_trailing_dividend_yield(self, ticker: str, current_price: float) -> float:
        # O MT5 não expõe dividendos de forma padronizada entre corretoras — não dá
        # para calcular isso de forma confiável pela API. O dado fundamentalista
        # completo (inclusive dividend yield) vem do Yahoo (ver fundamentals_provider.py),
        # que é o que o coletor diário (collector/mt5_daily_log.py) realmente usa.
        return 0.0

    def get_risk_free_rate(self) -> float:
        if not self.risk_free_symbol:
            raise MarketDataError(
                "MT5Provider sem 'risk_free_symbol' configurado — o MT5 não tem uma taxa "
                "livre de risco padrão entre corretoras. Passe risk_free_symbol=... (um "
                "instrumento do seu corretor, ex. CFD de título do tesouro) ou informe a "
                "taxa manualmente para quem precisa dela."
            )
        self._ensure_symbol(self.risk_free_symbol)
        tick = self._mt5.symbol_info_tick(self.risk_free_symbol)
        if tick is None or tick.last <= 0:
            raise MarketDataError(f"sem cotação para '{self.risk_free_symbol}' no MT5")
        return float(tick.last) / 100

    def _option_symbols(self, ticker: str) -> list:
        pattern = self.option_group_pattern.format(ticker=ticker)
        symbols = self._mt5.symbols_get(group=pattern) or ()
        options = [
            symbol for symbol in symbols
            if getattr(symbol, "option_right", -1) in (0, 1) and getattr(symbol, "option_strike", 0) > 0
        ]
        if not options:
            raise MarketDataError(
                f"nenhum símbolo de opção encontrado no MT5 para '{pattern}' ({self._last_error()}) — "
                "confira o padrão de nome dos símbolos de opção do seu corretor com inspect_symbol(...) "
                "e ajuste option_group_pattern se precisar."
            )
        return options

    def list_option_expirations(self, ticker: str) -> list:
        options = self._option_symbols(ticker)
        expirations = sorted({
            pd.Timestamp(symbol.expiration_time, unit="s")
            for symbol in options if getattr(symbol, "expiration_time", 0)
        })
        if not expirations:
            raise MarketDataError(
                f"símbolos de opção de '{ticker}' no MT5 sem 'expiration_time' preenchido — "
                "confira com inspect_symbol(...) se o seu corretor expõe esse campo."
            )
        return expirations

    def _download_chain(self, ticker: str, expiration: pd.Timestamp, right: int) -> pd.DataFrame:
        rows = []
        for symbol in self._option_symbols(ticker):
            if getattr(symbol, "option_right", None) != right:
                continue
            expiration_time = getattr(symbol, "expiration_time", 0)
            if not expiration_time or pd.Timestamp(expiration_time, unit="s").date() != expiration.date():
                continue
            self._mt5.symbol_select(symbol.name, True)
            tick = self._mt5.symbol_info_tick(symbol.name)
            bid = float(tick.bid) if tick else 0.0
            ask = float(tick.ask) if tick else 0.0
            last_price = float(tick.last) if tick else 0.0
            rows.append({
                "contractSymbol": symbol.name,
                "strike": float(symbol.option_strike),
                "bid": bid, "ask": ask, "lastPrice": last_price,
                "impliedVolatility": float("nan"),  # o MT5 não calcula/expõe IV — ver pricing/black_scholes.implied_volatility
                "openInterest": int(getattr(symbol, "session_open_interest", 0) or 0),
                "volume": int(tick.volume) if tick else 0,
            })
        if not rows:
            kind = "calls" if right == 0 else "puts"
            raise MarketDataError(f"sem {kind} de '{ticker}' no MT5 para o vencimento {expiration.date()}")
        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    def download_call_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame:
        return self._download_chain(ticker, expiration, right=0)  # SYMBOL_OPTION_RIGHT_CALL

    def download_put_chain(self, ticker: str, expiration: pd.Timestamp) -> pd.DataFrame:
        return self._download_chain(ticker, expiration, right=1)  # SYMBOL_OPTION_RIGHT_PUT
