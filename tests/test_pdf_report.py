from finstrategy.reporting.panel import PlainLanguageSummary
from finstrategy.reporting.pdf_report import generate_pdf_report


def test_generate_pdf_report_creates_nonempty_file(tmp_path):
    summaries = [
        PlainLanguageSummary(
            ticker="AAPL", headline="AAPL: comprar a ação + proteção (put) com strike 90,00",
            body_lines=["Sinal de entrada: comprar hoje.", "Perda máxima: 1.300 (13%)."],
            is_eligible_for_protection=True,
        ),
        PlainLanguageSummary(
            ticker="TAEE11.SA", headline="TAEE11.SA: sem proteção com opção disponível agora",
            body_lines=["Esta ação não tem opção de venda (put) disponível no Yahoo Finance no momento."],
            is_eligible_for_protection=False,
        ),
    ]
    output_path = tmp_path / "relatorio.pdf"
    result_path = generate_pdf_report(summaries, output_path)
    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000  # PDF real, não um arquivo vazio/truncado
    with open(output_path, "rb") as handle:
        assert handle.read(4) == b"%PDF"


def test_generate_pdf_report_handles_empty_summaries(tmp_path):
    output_path = tmp_path / "vazio.pdf"
    generate_pdf_report([], output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
