#!/usr/bin/env python3
"""Baixa a COTAHIST da B3 e extrai ações + opções do setor de energia/elétrico.

Lógica real em `src/finstrategy/data/cotahist_provider.py`. Este wrapper só baixa os
anos pedidos (cacheando em `data_store/cotahist/`), filtra pelas raízes dos tickers e
grava um CSV por classe (ação / call / put).

Uso:
    # anos 2023 a 2025, raízes padrão do setor elétrico
    python scripts/collect_cotahist.py --years 2023-2025

    # anos avulsos e raízes customizadas (ações à vista + toda a cadeia de opções)
    python scripts/collect_cotahist.py --years 2024,2025 --roots PETR,ELET,EGIE

    python scripts/collect_cotahist.py --years 2025 --output-dir data_store/energia

Não precisa de Yahoo nem de terminal MT5 aberto — é só download de arquivo público.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finstrategy.data.cotahist_provider import CotahistProvider  # noqa: E402

# Raízes do SETOR ELÉTRICO brasileiro na B3 (geração/transmissão/distribuição).
# Raiz = ticker sem dígito de classe; casa todas as classes da ação (ON/PN/UNIT) e,
# se pedido, toda a cadeia de opções do ativo-objeto.
RAIZES_SETOR_ELETRICO = [
    "TAEE",  # Taesa (transmissão)
    "TRPL",  # ISA CTEEP / Transmissão Paulista
    "ALUP",  # Alupar (transmissão/geração)
    "CPFE",  # CPFL Energia (distribuição)
    "EGIE",  # Engie Brasil (geração)
    "CMIG",  # Cemig (integrada)
    "CPLE",  # Copel (integrada)
    "ELET",  # Eletrobras (integrada)
    "ENGI",  # Energisa (distribuição)
    "ENEV",  # Eneva (geração)
    "EQTL",  # Equatorial (distribuição/transmissão)
    "NEOE",  # Neoenergia (integrada)
    "AURE",  # Auren Energia (geração)
    "ENMT",  # Energisa Mato Grosso
    "GEPA",  # Energias do Brasil / Gerdau Pactual Energia (histórico)
    "LIGT",  # Light (distribuição)
    "REDE",  # Rede Energia
    "COCE",  # Coelce (histórico)
    "EKTR",  # Elektro (histórico)
]

# Óleo & gás — NÃO fazem parte do setor elétrico; ficam aqui só para quem quiser
# passar via --roots explicitamente (ex. --roots PETR,PRIO,RRRP).
RAIZES_OLEO_GAS = ["PETR", "PRIO", "RRRP", "RECV", "CSAN", "UGPA", "VBBR"]


def _parse_years(texto: str) -> list[int]:
    anos: set[int] = set()
    for pedaco in texto.split(","):
        pedaco = pedaco.strip()
        if "-" in pedaco:
            ini, fim = pedaco.split("-", 1)
            anos.update(range(int(ini), int(fim) + 1))
        elif pedaco:
            anos.add(int(pedaco))
    return sorted(anos)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        required=True,
        help="Anos a baixar: '2025', '2023-2025' ou '2024,2025'.",
    )
    parser.add_argument(
        "--roots",
        default=",".join(RAIZES_SETOR_ELETRICO),
        help="Raízes dos tickers separadas por vírgula (default: setor elétrico).",
    )
    parser.add_argument(
        "--only",
        default="acao,call,put",
        help=(
            "Classes a coletar, separadas por vírgula: acao, call, put "
            "(default: todas). Use '--only acao' para só ações — bem mais rápido "
            "num intervalo longo de anos, pois nem processa as opções."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data_store/cotahist",
        help="Onde gravar os CSVs de saída (default: data_store/cotahist).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=60.0,
        help=(
            "Pausa ALEATÓRIA de 0 até N segundos entre downloads de anos "
            "(rate-limiting educado; default 60 = até 1 min). Só vale para downloads "
            "de verdade — não pausa em anos lidos do cache. Use 0 para desligar."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Rebaixa os .ZIP mesmo que já estejam no cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    anos = _parse_years(args.years)
    raizes = [r.strip().upper() for r in args.roots.split(",") if r.strip()]
    classes = [c.strip().lower() for c in args.only.split(",") if c.strip()]
    validas = {"acao", "call", "put"}
    invalidas = set(classes) - validas
    if invalidas:
        parser.error(f"--only aceita só {sorted(validas)}; recebi {sorted(invalidas)}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Anos: {anos}")
    print(f"Raízes: {raizes}")
    print(f"Classes: {classes}")

    provider = CotahistProvider()
    df = provider.load_setor(
        raizes,
        anos,
        classes=classes,
        force_download=args.force_download,
        sleep_max=args.sleep,
    )

    if df.empty:
        print(
            "Nenhum registro encontrado. Verifique as raízes/anos — e note que anos "
            "indisponíveis são pulados com aviso (WARNING acima), não derrubam a coleta.",
            file=sys.stderr,
        )
        return 1

    # Prefixo do arquivo: 'eletrico' se for o default, senão 'setor'.
    prefixo = "eletrico" if set(raizes) == set(RAIZES_SETOR_ELETRICO) else "setor"
    sufixo = f"{anos[0]}_{anos[-1]}"

    # Grava um CSV por classe pedida + um consolidado.
    consolidado = out_dir / f"{prefixo}_{sufixo}.csv"
    df.to_csv(consolidado, index=False)
    print(f"Consolidado ({len(df)} linhas): {consolidado}")

    for classe in classes:
        sub = df[df["classe"] == classe]
        if sub.empty:
            continue
        destino = out_dir / f"{prefixo}_{classe}_{sufixo}.csv"
        sub.to_csv(destino, index=False)
        print(f"  {classe}: {len(sub)} linhas, {sub['codneg'].nunique()} séries -> {destino}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
