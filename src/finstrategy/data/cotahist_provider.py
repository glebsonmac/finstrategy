"""Leitor da COTAHIST da B3 — série histórica oficial de cotações (ações + opções).

Por que este módulo existe
---------------------------
Tanto o Yahoo quanto o MT5 falharam para o caso de uso da dissertação (cotações de
ações **e opções** do setor elétrico/energia da B3):

- O Yahoo não cobre a cadeia de opções da B3 e alguns papéis nem resolvem (ver a nota
  em `config/tickers_setor_eletrico.yaml`: `ELET3.SA`, `TRPL4.SA` etc. não resolveram).
- O MT5 (ver `mt5_provider.py`) só enxerga as opções que o corretor/demo lista *agora*
  — contrato vencido some do servidor, então não dá histórico de anos passados.

A **COTAHIST** resolve os dois problemas: é o arquivo oficial da B3 com **todos** os
instrumentos negociados no ano — ações, opções de compra (TPMERC 070) e de venda
(080), inclusive séries já vencidas. É um arquivo estático, público e gratuito,
disponibilizado pela própria B3 justamente para download. Baixa-se **um arquivo por
ano**; aqui ele é cacheado em disco (`data_store/cotahist/`) e relido localmente, então
o download acontece uma vez só por ano.

Sobre "randomizar IP/MAC para não ser derrubado pela B3"
--------------------------------------------------------
Não é feito aqui, de propósito. A COTAHIST não é uma API com rate-limit agressivo: é um
arquivo estático cujo uso pretendido é exatamente ser baixado. Como baixamos 1 arquivo
por ano e cacheamos em disco, há no máximo um punhado de requisições na vida do projeto
— não há o que "evadir". (E randomizar MAC não teria efeito nenhum: o endereço MAC não
trafega além da sua rede local, o servidor da B3 nunca o vê.) O módulo usa um
User-Agent honesto e um timeout razoável, como qualquer cliente HTTP bem-comportado.

Layout do arquivo
-----------------
Texto de largura fixa, 245 colunas por linha, uma linha por instrumento/pregão. O
registro tipo 00 é cabeçalho, 01 é cotação, 99 é trailer. As posições (1-indexadas,
inclusivas) e os decimais implícitos (V99 = 2 casas) seguem o layout oficial da B3
("Séries Históricas — layout do arquivo COTAHIST").
"""
from __future__ import annotations

import io
import logging
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from .errors import MarketDataError

logger = logging.getLogger(__name__)

# URL estável da série histórica anual (arquivo COTAHIST_A<ano>.ZIP).
_COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"

# User-Agent honesto (não é spoofing: identifica um cliente HTTP comum, não finge ser
# outra máquina). A B3 serve o arquivo normalmente com um UA de navegador.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; finstrategy/1.0; pesquisa academica; "
        "+https://github.com/)"
    )
}

# Diretório de cache (um .ZIP por ano). Fica fora do git (ver .gitignore / data_store).
_CACHE_DIR = Path("data_store") / "cotahist"

# Tipos de mercado (campo TPMERC) que nos interessam.
TPMERC_VISTA = 10  # ação à vista (lote padrão)
TPMERC_OPCAO_COMPRA = 70  # call
TPMERC_OPCAO_VENDA = 80  # put

# Layout de largura fixa do registro tipo 01 (col_início 0-indexado, col_fim exclusivo).
# Nome -> (início, fim, conversor). Decimais implícitos tratados no conversor.
def _preco(x: str) -> float:
    """Campo de preço (11)V99: inteiro com 2 casas decimais implícitas."""
    return int(x) / 100.0


def _data(x: str) -> pd.Timestamp | None:
    x = x.strip()
    if not x or x == "0" * len(x):
        return None
    return pd.to_datetime(x, format="%Y%m%d")


_COLUNAS = {
    "data_pregao": (2, 10, _data),
    "codbdi": (10, 12, str.strip),
    "codneg": (12, 24, str.strip),  # ticker / código de negociação
    "tpmerc": (24, 27, int),  # tipo de mercado
    "nomres": (27, 39, str.strip),  # nome resumido da empresa
    "especi": (39, 49, str.strip),  # especificação (ON, PN, etc.)
    "preabe": (56, 69, _preco),  # abertura
    "premax": (69, 82, _preco),  # máximo
    "premin": (82, 95, _preco),  # mínimo
    "premed": (95, 108, _preco),  # médio
    "preult": (108, 121, _preco),  # último (fechamento)
    "totneg": (147, 152, int),  # nº de negócios
    "quatot": (152, 170, int),  # quantidade total negociada
    "voltot": (170, 188, _preco),  # volume financeiro total (16)V99
    "preexe": (188, 201, _preco),  # preço de exercício (opções)
    "datven": (202, 210, _data),  # data de vencimento (opções)
    "codisi": (230, 242, str.strip),  # código ISIN
}


@dataclass
class CotahistProvider:
    """Baixa, cacheia e lê a COTAHIST anual da B3.

    Parameters
    ----------
    cache_dir:
        Onde guardar os .ZIP baixados. Default: ``data_store/cotahist``.
    timeout:
        Timeout (s) de cada requisição HTTP.
    """

    cache_dir: Path = _CACHE_DIR
    timeout: float = 120.0

    # ---- download / cache -------------------------------------------------

    def _zip_path(self, year: int) -> Path:
        return Path(self.cache_dir) / f"COTAHIST_A{year}.ZIP"

    def ensure_year(self, year: int, *, force: bool = False) -> Path:
        """Garante o .ZIP do ano em disco (baixa só se faltar). Retorna o caminho."""
        dest = self._zip_path(year)
        if dest.exists() and not force:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _COTAHIST_URL.format(year=year)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:  # rede, 404 de ano futuro, etc.
            raise MarketDataError(
                f"Falha ao baixar a COTAHIST de {year} ({url}): {exc}"
            ) from exc
        # Resposta precisa ser um ZIP de verdade (a B3 às vezes devolve HTML de erro).
        if not resp.content[:2] == b"PK":
            raise MarketDataError(
                f"Resposta da B3 para {year} não é um ZIP válido "
                f"(ano ainda não publicado ou URL mudou): {url}"
            )
        dest.write_bytes(resp.content)
        return dest

    # ---- parsing ----------------------------------------------------------

    def _parse_zip(self, zip_path: Path) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path) as zf:
            # O ZIP tem um único .TXT com o mesmo nome-base.
            nome_txt = next(n for n in zf.namelist() if n.upper().endswith(".TXT"))
            with zf.open(nome_txt) as fh:
                linhas = io.TextIOWrapper(fh, encoding="latin-1")
                registros = []
                for linha in linhas:
                    if not linha.startswith("01"):  # só registros de cotação
                        continue
                    registros.append(_parse_linha(linha))
        if not registros:
            raise MarketDataError(f"Nenhum registro de cotação em {zip_path}")
        return pd.DataFrame.from_records(registros)

    def load_year(self, year: int, *, force_download: bool = False) -> pd.DataFrame:
        """DataFrame com TODAS as cotações do ano (ações + opções + outros mercados)."""
        zip_path = self.ensure_year(year, force=force_download)
        return self._parse_zip(zip_path)

    # ---- filtros de alto nível -------------------------------------------

    def load_setor(
        self,
        roots: list[str],
        years: list[int],
        *,
        classes: list[str] | None = None,
        force_download: bool = False,
        skip_missing_years: bool = True,
        sleep_max: float = 0.0,
    ) -> pd.DataFrame:
        """Cotações de um setor (ações à vista e/ou opções) para vários anos.

        `roots` é a lista de raízes dos tickers do setor (ex. ``["ELET", "CMIG",
        "EGIE", "TAEE", "CPLE", "CPFE"]``). Uma ação casa quando o ``codneg`` começa
        pela raiz e é mercado à vista; uma opção casa quando o ``codneg`` começa pela
        raiz e o mercado é 070 (call) ou 080 (put) — assim pegamos toda a cadeia de
        opções daquele ativo-objeto, inclusive séries já vencidas no ano.

        `classes` restringe o resultado a um subconjunto de ``{"acao", "call",
        "put"}`` (default: as três). Passar ``["acao"]`` é bem mais rápido num
        intervalo longo de anos, porque nem chega a filtrar/concatenar as opções (que
        são a imensa maioria das linhas).

        `skip_missing_years`: se ``True`` (default), um ano que não baixa/parseia
        (ex. ano ainda não publicado, 404, queda de rede) vira um aviso no log e é
        **pulado** — os demais anos continuam. Se ``False``, o primeiro ano com
        problema interrompe tudo (comportamento antigo). Isso é o que te deixa pedir
        ``2010-2026`` sem um único ano ruim derrubar a coleta inteira.

        `sleep_max`: se > 0, dorme um intervalo **aleatório** em ``[0, sleep_max]``
        segundos *entre downloads de verdade* — rate-limiting educado, para espaçar o
        tráfego quando se baixa vários anos de uma vez. Não dorme depois do último ano
        nem quando o ano veio do cache (leitura local não toca a B3).

        Acrescenta a coluna ``classe`` com os valores ``acao`` / ``call`` / ``put``.
        """
        roots_norm = [r.strip().upper() for r in roots]
        classes_sel = set(classes) if classes else {"acao", "call", "put"}
        tpmerc_por_classe = {
            "acao": TPMERC_VISTA,
            "call": TPMERC_OPCAO_COMPRA,
            "put": TPMERC_OPCAO_VENDA,
        }
        mercados = {tpmerc_por_classe[c] for c in classes_sel}
        rotulo_mercado = {v: k for k, v in tpmerc_por_classe.items()}

        partes = []
        anos_ok, anos_falha = [], []
        for i, year in enumerate(years):
            # Cache-hit? Então não houve requisição à B3 → não precisa pausar.
            baixou = force_download or not self._zip_path(year).exists()
            try:
                df = self.load_year(year, force_download=force_download)
            except MarketDataError as exc:
                if not skip_missing_years:
                    raise
                logger.warning("Pulando %s: %s", year, exc)
                anos_falha.append(year)
                continue
            # Pausa educada só depois de um download real e nunca após o último ano.
            if sleep_max > 0 and baixou and i < len(years) - 1:
                espera = random.uniform(0, sleep_max)
                logger.info("Pausa de %.1fs antes do próximo ano...", espera)
                time.sleep(espera)
            df = df[df["tpmerc"].isin(mercados)].copy()
            # casa a raiz do código de negociação
            mask = df["codneg"].str.upper().str.startswith(tuple(roots_norm))
            df = df[mask].copy()
            df["classe"] = df["tpmerc"].map(rotulo_mercado)
            partes.append(df)
            anos_ok.append(year)

        if anos_falha:
            logger.warning(
                "Anos obtidos: %s | anos pulados (indisponíveis): %s",
                anos_ok,
                anos_falha,
            )
        if not partes:
            return pd.DataFrame()
        out = pd.concat(partes, ignore_index=True)
        return out.sort_values(["codneg", "data_pregao"]).reset_index(drop=True)


def _parse_linha(linha: str) -> dict:
    """Converte uma linha de 245 colunas num dict segundo `_COLUNAS`."""
    registro = {}
    for nome, (ini, fim, conv) in _COLUNAS.items():
        bruto = linha[ini:fim]
        try:
            registro[nome] = conv(bruto)
        except (ValueError, TypeError):
            registro[nome] = None
    return registro
