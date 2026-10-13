"""PDF simples para usuário não técnico (plan.md, seção 7).

Uma página por ativo, linguagem direta (vem de `reporting/panel.py`), sem jargão, com
o aviso de que nada aqui é recomendação de investimento (texto já existente no
notebook original, seção 11, reaproveitado aqui).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .panel import PlainLanguageSummary

DISCLAIMER = (
    "Nada aqui garante lucro e não é recomendação de investimento. Confira os preços "
    "na corretora antes de operar. As cotações têm atraso e podem estar fechadas fora "
    "do horário de pregão."
)


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TituloRelatorio", parent=base["Title"], fontSize=20, spaceAfter=6),
        "subtitle": ParagraphStyle("Subtitulo", parent=base["Normal"], fontSize=11, textColor=colors.grey, spaceAfter=18),
        "headline": ParagraphStyle("Headline", parent=base["Heading2"], fontSize=15, textColor=colors.HexColor("#0b3d91"), spaceAfter=10),
        "body": ParagraphStyle("Corpo", parent=base["Normal"], fontSize=12, leading=17, spaceAfter=8),
        "disclaimer": ParagraphStyle("Disclaimer", parent=base["Normal"], fontSize=9, textColor=colors.grey, leading=12),
    }


def _ticker_page(summary: PlainLanguageSummary, styles: dict) -> list:
    flowables = [Paragraph(summary.headline, styles["headline"])]
    for line in summary.body_lines:
        flowables.append(Paragraph(line, styles["body"]))
    flowables.append(Spacer(1, 0.4 * cm))
    status = "✔ proteção disponível (opção de venda líquida no Yahoo)" if summary.is_eligible_for_protection else "⚠ sem proteção disponível agora"
    flowables.append(Paragraph(status, styles["disclaimer"]))
    return flowables


def generate_pdf_report(summaries: list[PlainLanguageSummary], output_path: Path | str, report_date: pd.Timestamp | None = None) -> Path:
    """Gera um PDF com uma página de capa + uma página por ativo. Devolve o caminho gerado."""
    report_date = report_date or pd.Timestamp.today().normalize()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    document = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm)
    story = [
        Paragraph("Relatório de ações + proteção com opções", styles["title"]),
        Paragraph(f"Setor elétrico (e outros ativos monitorados) · {report_date.date()}", styles["subtitle"]),
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
        story.extend(_ticker_page(summary, styles))
        story.append(Spacer(1, 0.6 * cm))
        story.append(Paragraph(DISCLAIMER, styles["disclaimer"]))

    document.build(story)
    return output_path
