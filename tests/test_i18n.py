import json
import re
from pathlib import Path

import numpy as np
import pytest

from riskbt import config, explain
from riskbt.estimators import ESTIMATORS, garch_t
from riskbt.i18n import LOCALES_DIR, format_currency, format_number, format_percent, format_scientific, load_locale, t
from riskbt.recommend import CAVIAR_ONLY, NONE_PASSED, PASSED

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_returns():
    return np.random.default_rng(31).standard_t(5, 1000) * 0.012


def source_files():
    return [ROOT / "app.py", *sorted((ROOT / "riskbt").glob("*.py"))]


def test_locales_have_identical_keys():
    assert set(load_locale("en")) == set(load_locale("pt"))


def test_locales_have_identical_placeholders():
    pattern = re.compile(r"\{(\w+)\}")
    en, pt = load_locale("en"), load_locale("pt")
    for key in en:
        assert set(pattern.findall(en[key])) == set(pattern.findall(pt[key])), key


def test_every_static_key_used_in_code_exists():
    pattern = re.compile(r"""\bt\(\s*["']([\w.]+)["']""")
    used = {key for path in source_files() for key in pattern.findall(path.read_text(encoding="utf-8"))}
    used |= {key for path in source_files() for key in re.findall(r'"((?:tests|backtest|compare)\.[\w.]+)"', path.read_text(encoding="utf-8"))}
    assert used
    assert not used - set(load_locale("en"))


def test_dynamic_keys_exist():
    en = load_locale("en")
    for regime in ("calm", "crisis", "unclassified", "all"):
        assert f"regime.{regime}" in en
    for zone in ("green", "yellow", "red"):
        assert f"zone.{zone}" in en
    for status in (PASSED, CAVIAR_ONLY, NONE_PASSED):
        assert f"recommend.reason.{status}" in en
    for name in ESTIMATORS:
        assert f"estimator.{name}.name" in en


def test_no_key_is_unused():
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_files())
    dynamic = ("main.regime_reading.", "estimator.", "regime.", "zone.", "recommend.reason.", "test.")
    unused = [key for key in load_locale("en") if not key.startswith(dynamic) and f'"{key}"' not in text and f"'{key}'" not in text]
    assert not unused


def test_lookup_fills_placeholders():
    assert "R$ 5" in t("main.var_reading", "en", days=99, amount="R$ 5", percent="1%")
    with pytest.raises(KeyError):
        t("no.such.key", "en")


def test_number_formatting_follows_the_language():
    assert format_currency(3490, "pt") == "R$ 3.490,00"
    assert format_currency(3490, "en") == "R$ 3,490.00"
    assert format_percent(0.0349, "pt") == "3,49%"
    assert format_percent(0.0349, "en") == "3.49%"
    assert format_number(1234567.891, "pt", 1) == "1.234.567,9"
    assert format_scientific(1.051e-6, "pt") == "1,051e-06"


@pytest.mark.parametrize("lang", ["en", "pt"])
@pytest.mark.parametrize("alpha", [config.ALPHA_VAR, config.ALPHA_ES])
@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_every_estimator_explanation_renders_with_actual_numbers(sample_returns, name, alpha, lang):
    spec = ESTIMATORS[name]
    forecast = spec.forecast(sample_returns, spec.fit(sample_returns, alpha), alpha)
    result = explain.explain_estimator(name, forecast, alpha, 100_000, lang)
    assert len(result.steps) == explain.ESTIMATOR_STEP_COUNTS[name]
    assert result.summary and result.formula and result.assumptions and result.limitations and result.reference
    assert format_percent(forecast.var, lang) in " ".join(result.steps)
    assert not re.search(r"\{\w+\}", " ".join(result.steps))
    assert not any(step.endswith(" ()") for step in result.steps)


def test_garch_explanation_shows_the_required_values(sample_returns):
    forecast = garch_t(sample_returns, config.ALPHA_ES)
    text = " ".join(explain.explain_estimator("garch_t", forecast, config.ALPHA_ES, 100_000, "en").steps)
    for symbol in ("ω =", "a =", "b =", "ν =", "σ =", "a + b ="):
        assert symbol in text
    assert format_number(forecast.details["half_life"], "en", 1) in text


@pytest.mark.parametrize("lang", ["en", "pt"])
@pytest.mark.parametrize("test", ["kupiec", "christoffersen", "z2", "fz0", "dm"])
def test_backtest_explanations_render(test, lang):
    result = explain.test_explanation(test, lang)
    assert result.summary and result.formula and result.reference


def test_main_summary_matches_the_project_example():
    lines = explain.main_result_summary(0.0349, 0.042, 0.01, 0.025, 100_000, "en")
    assert lines[0].startswith("On 99 out of 100 days")
    assert "R$ 3,490.00" in lines[0]
    assert "R$ 4.200,00" in explain.main_result_summary(0.0349, 0.042, 0.01, 0.025, 100_000, "pt")[1]


def test_summary_without_es_says_so():
    lines = explain.main_result_summary(0.03, None, 0.01, 0.025, 100_000, "en")
    assert lines[1] == t("main.es_missing", "en")


def test_notice_text_is_the_required_one():
    assert t("app.notice", "en") == "Risk measurement tool. Not investment advice."
    assert t("app.notice", "pt") == "Ferramenta de medição de risco. Não é recomendação de investimento."


def test_locale_files_are_valid_utf8_json():
    for path in LOCALES_DIR.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))
