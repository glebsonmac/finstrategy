"""Translated text lookup and number formatting. Currency stays BRL in both languages."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LANGUAGES = {"en": "English", "pt": "Português"}
DEFAULT_LANGUAGE = "en"
LOCALES_DIR = Path(__file__).parent / "locales"


@lru_cache(maxsize=None)
def load_locale(lang: str) -> dict[str, str]:
    return json.loads((LOCALES_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def t(key: str, lang: str, **values) -> str:
    """Return the translated text for `key`, filling {placeholders} with `values`."""
    return load_locale(lang)[key].format(**values)


def format_number(value: float, lang: str, decimals: int = 2) -> str:
    """Thousands and decimal separators of each language: 3,490.00 (en) or 3.490,00 (pt)."""
    text = f"{value:,.{decimals}f}"
    return text.translate(str.maketrans(",.", ".,")) if lang == "pt" else text


def format_scientific(value: float, lang: str) -> str:
    text = f"{value:.3e}"
    return text.replace(".", ",") if lang == "pt" else text


def format_currency(value: float, lang: str) -> str:
    return f"R$ {format_number(value, lang)}"


def format_percent(fraction: float, lang: str, decimals: int = 2) -> str:
    return f"{format_number(100 * fraction, lang, decimals)}%"
