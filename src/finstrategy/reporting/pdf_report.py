"""PDF para o usuário (plan.md, seção 7 + seções 8/9 do notebook original).

Uma página por ativo com o resumo em linguagem direta (vem de `reporting/panel.py`),
seguido do mesmo conteúdo técnico que o notebook mostrava (menu completo de strikes,
avaliação da DQN de entrada). A capa mostra data E hora da geração — cada chamada
grava um arquivo próprio (`collector.csv_store.report_pdf_path`), não sobrescreve o
relatório anterior do mesmo dia.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .panel import (
    PlainLanguageSummary,
    build_entry_evaluation_table,
    build_entry_evaluation_verdict,
    build_strike_menu_header,
    build_strike_menu_table,
)

DISCLAIMER = (
    "Nada aqui garante lucro e não é recomendação de investimento. Confira os preços "
    "na corretora antes de operar. As cotações têm atraso e podem estar fechadas fora "
    "do horário de pregão."
)

_TABLE_WIDTH = 17 * cm  # largura útil da página A4 com margens de 2cm


def _escape_xml(text: str) -> str:
    """`Paragraph` interpreta o texto como um mini-XML — qualquer `&`/`<`/`>` vindo de
    dado real (nome de ativo, texto de erro, etc.) sem isso quebra a geração do PDF
    inteiro, silenciosamente do ponto de vista do usuário (o download some)."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _markdown_bold_to_xml(text: str) -> str:
    escaped = _escape_xml(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TituloRelatorio", parent=base["Title"], fontSize=20, spaceAfter=6),
        "subtitle": ParagraphStyle("Subtitulo", parent=base["Normal"], fontSize=11, textColor=colors.grey, spaceAfter=18),
        "headline": ParagraphStyle("Headline", parent=base["Heading2"], fontSize=15, textColor=colors.HexColor("#0b3d91"), spaceAfter=10),
        "subheading": ParagraphStyle("Subsecao", parent=base["Heading3"], fontSize=12, textColor=colors.HexColor("#0b3d91"), spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("Corpo", parent=base["Normal"], fontSize=12, leading=17, spaceAfter=8),
        "disclaimer": ParagraphStyle("Disclaimer", parent=base["Normal"], fontSize=9, textColor=colors.grey, leading=12),
        "cell": ParagraphStyle("Celula", parent=base["Normal"], fontSize=6.5, leading=8),
        "cell_header": ParagraphStyle("CelulaHeader", parent=base["Normal"], fontSize=6.5, leading=8, textColor=colors.white),
    }


def _dataframe_to_table(table_df: pd.DataFrame, styles: dict, index_label: str = "") -> Table:
    """Converte um DataFrame já formatado (strings) para uma `Table` do reportlab, com
    células em `Paragraph` pra permitir quebra de linha em colunas estreitas."""
    header = [Paragraph(f"<b>{_escape_xml(index_label)}</b>", styles["cell_header"])] + [
        Paragraph(f"<b>{_escape_xml(column)}</b>", styles["cell_header"]) for column in table_df.columns
    ]
    rows = [header]
    for row_label, row in table_df.iterrows():
        rows.append(
            [Paragraph(_escape_xml(row_label), styles["cell"])]
            + [Paragraph(_escape_xml(value), styles["cell"]) for value in row]
        )
    column_count = len(table_df.columns) + 1
    table = Table(rows, colWidths=[_TABLE_WIDTH / column_count] * column_count, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fb")]),
    ]))
    return table


def _strike_menu_flowables(ticker: str, result: dict, styles: dict) -> list:
    """Menu completo de strikes (não só a recomendada) — notebook, seção 8."""
    flowables = [
        Paragraph("Menu completo de strikes disponíveis", styles["subheading"]),
        Paragraph(_escape_xml(build_strike_menu_header(ticker, result)), styles["disclaimer"]),
        Spacer(1, 0.2 * cm),
    ]
    table_df = build_strike_menu_table(result)
    table_df.index = table_df.index.str.replace("►", ">>", regex=False)  # glyph ausente na fonte base do reportlab
    half = len(table_df.columns) // 2
    for part in (table_df.iloc[:, :half], table_df.iloc[:, half:]):
        flowables.append(_dataframe_to_table(part, styles, index_label="strike"))
        flowables.append(Spacer(1, 0.3 * cm))
    if not result["recommended_respects_limit"]:
        flowables.append(Paragraph(
            "Nenhuma strike deixa a perda máxima dentro do limite desejado; a recomendada "
            "é a de menor perda máxima disponível.", styles["disclaimer"],
        ))
    return flowables


def _entry_evaluation_flowables(evaluation: dict, styles: dict) -> list:
    """Avaliação da DQN de entrada (4 estratégias comparadas) — notebook, seção 9."""
    flowables = [
        Paragraph("Avaliação do ponto de entrada (DQN vs. referências)", styles["subheading"]),
        _dataframe_to_table(build_entry_evaluation_table(evaluation), styles, index_label="estratégia"),
        Spacer(1, 0.2 * cm),
        Paragraph(_markdown_bold_to_xml(build_entry_evaluation_verdict(evaluation)), styles["body"]),
    ]
    return flowables


def _ticker_page(summary: PlainLanguageSummary, styles: dict, result: dict | None = None, evaluation: dict | None = None) -> list:
    flowables = [Paragraph(_escape_xml(summary.headline), styles["headline"])]
    for line in summary.body_lines:
        flowables.append(Paragraph(_escape_xml(line), styles["body"]))
    flowables.append(Spacer(1, 0.4 * cm))
    status = "✔ proteção disponível (opção de venda líquida no Yahoo)" if summary.is_eligible_for_protection else "⚠ sem proteção disponível agora"
    flowables.append(Paragraph(status, styles["disclaimer"]))
    if result is not None:
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.extend(_strike_menu_flowables(summary.ticker, result, styles))
    if evaluation is not None:
        flowables.append(Spacer(1, 0.3 * cm))
        flowables.extend(_entry_evaluation_flowables(evaluation, styles))
    return flowables


def generate_pdf_report(
    summaries: list[PlainLanguageSummary],
    output_path: Path | str,
    report_date: pd.Timestamp | None = None,
    results: dict | None = None,
    entry_evaluations: dict | None = None,
) -> Path:
    """Gera um PDF com capa + uma página por ativo (resumo simples + menu completo de
    strikes + avaliação da DQN, quando disponíveis).

    `report_date` é a data/hora de geração mostrada na capa — por padrão, agora. Cada
    chamada deve usar um `output_path` próprio (ver `collector.csv_store.report_pdf_path`)
    pra não sobrescrever um relatório gerado antes no mesmo dia. `results`/`entry_evaluations`
    são opcionais (mesmo formato de `strategy.protective_put.analyze_ticker`/
    `rl.entry_dqn.evaluate_entry_timing`, indexados por ticker) — sem eles, a página de
    cada ativo tem só o resumo em linguagem simples.
    """
    results = results or {}
    entry_evaluations = entry_evaluations or {}
    report_date = report_date or pd.Timestamp.now()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    document = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm)
    story = [
        Paragraph("Relatório de ações + proteção com opções", styles["title"]),
        Paragraph(
            f"Setor elétrico (e outros ativos monitorados) · gerado em {report_date.strftime('%d/%m/%Y às %H:%M:%S')}",
            styles["subtitle"],
        ),
    ]
    if summaries:
        summary_table_rows = [["Ativo", "O que fazer"]] + [[item.ticker, item.headline.split(":", 1)[-1].strip()] for item in summaries]
        table = Table(summary_table_rows, colWidths=[3 * cm, 13 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("Nenhum ativo elegível nesta rodada.", styles["body"]))

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(DISCLAIMER, styles["disclaimer"]))

    for summary in summaries:
        story.append(PageBreak())
        story.extend(_ticker_page(summary, styles, results.get(summary.ticker), entry_evaluations.get(summary.ticker)))
        story.append(Spacer(1, 0.6 * cm))
        story.append(Paragraph(DISCLAIMER, styles["disclaimer"]))

    document.build(story)
    return output_path
