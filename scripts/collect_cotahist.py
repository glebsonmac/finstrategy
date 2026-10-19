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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finstrategy.data.cotahist_provider import CotahistProvider  # noqa: E402

# Raízes do setor elétrico/energia da B3 (derivadas de config/tickers_setor_eletrico.yaml
# + grandes nomes de energia). Raiz = ticker sem dígito de classe; casa a ação e toda a
# cadeia de opções do ativo-objeto.
RAIZES_ENERGIA_PADRAO = [
    "TAEE",  # Taesa
    "CPFE",  # CPFL Energia
    "EGIE",  # Engie Brasil
    "CMIG",  # Cemig
    "CPLE",  # Copel
    "ELET",  # Eletrobras
    "ENGI",  # Energisa
    "ENEV",  # Eneva
    "EQTL",  # Equatorial
    "NEOE",  # Neoenergia
    "AURE",  # Auren
    "TRPL",  # ISA CTEEP (Transmissão Paulista)
    "PETR",  # Petrobras (energia / óleo e gás)
    "PRIO",  # PRIO (óleo e gás)
]


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
        default=",".join(RAIZES_ENERGIA_PADRAO),
        help="Raízes dos tickers separadas por vírgula (default: setor energia).",
    )
    parser.add_argument(
        "--output-dir",
        default="data_store/cotahist",
        help="Onde gravar os CSVs de saída (default: data_store/cotahist).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Rebaixa os .ZIP mesmo que já estejam no cache.",
    )
    args = parser.parse_args()

    anos = _parse_years(args.years)
    raizes = [r.strip().upper() for r in args.roots.split(",") if r.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Anos: {anos}")
    print(f"Raízes: {raizes}")

    provider = CotahistProvider()
    df = provider.load_setor(raizes, anos, force_download=args.force_download)

    if df.empty:
        print("Nenhum registro encontrado para essas raízes/anos.", file=sys.stderr)
        return 1

    # Grava um CSV por classe + um consolidado.
    consolidado = out_dir / f"energia_{anos[0]}_{anos[-1]}.csv"
    df.to_csv(consolidado, index=False)
    print(f"Consolidado ({len(df)} linhas): {consolidado}")

    for classe in ("acao", "call", "put"):
        sub = df[df["classe"] == classe]
        if sub.empty:
            continue
        destino = out_dir / f"energia_{classe}_{anos[0]}_{anos[-1]}.csv"
        sub.to_csv(destino, index=False)
        print(f"  {classe}: {len(sub)} linhas, {sub['codneg'].nunique()} séries -> {destino}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
