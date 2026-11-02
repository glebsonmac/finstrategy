"""Layered explanations built from an estimator's `details` and the locale texts.

Nothing is recomputed here: every number comes from `details` or from the backtest tables.
Formulas are the same in both languages, so they live in this module and not in the locales.
"""

from __future__ import annotations

from dataclasses import dataclass

from riskbt.estimators import Forecast
from riskbt.i18n import format_currency, format_number, format_percent, format_scientific, t

FORMULAS = {
    "hs": r"\widehat{VaR}^{\alpha}_{t+1} = -r_{(k)}, \quad \widehat{ES}^{\alpha}_{t+1} = -\frac{1}{k}\sum_{i=1}^{k} r_{(i)}, \quad k = \lfloor n\alpha \rfloor",
    "normal": r"\widehat{VaR}^{\alpha}_{t+1} = -(\hat\mu + \hat\sigma z_\alpha), \quad \widehat{ES}^{\alpha}_{t+1} = -\hat\mu + \hat\sigma \frac{\phi(z_\alpha)}{\alpha}",
    "garch_t": r"\sigma^2_t = \omega + a\,\varepsilon^2_{t-1} + b\,\sigma^2_{t-1}, \quad \widehat{VaR}^{\alpha}_{t+1} = -(\hat\mu + \hat\sigma_{t+1} s_\nu t_{\nu,\alpha}), \quad \widehat{ES}^{\alpha}_{t+1} = -\hat\mu + \hat\sigma_{t+1} s_\nu \frac{g_\nu(t_{\nu,\alpha})}{\alpha} \frac{\nu + t^2_{\nu,\alpha}}{\nu - 1}",
    "fhs": r"\hat z_i = \frac{r_i - \hat\mu}{\hat\sigma_i}, \quad \widehat{VaR}^{\alpha}_{t+1} = -(\hat\mu + \hat\sigma_{t+1} \hat z_{(k)}), \quad \widehat{ES}^{\alpha}_{t+1} = -\left(\hat\mu + \hat\sigma_{t+1} \frac{1}{k}\sum_{i=1}^{k} \hat z_{(i)}\right)",
    "evt_pot": r"y_\alpha = u + \frac{\hat\beta}{\hat\xi}\left[\left(\frac{n}{N_u}\alpha\right)^{-\hat\xi} - 1\right], \quad e_{y} = \frac{y_\alpha}{1-\hat\xi} + \frac{\hat\beta - \hat\xi u}{1-\hat\xi}, \quad \widehat{VaR}^{\alpha}_{t+1} = -\hat\mu + \hat\sigma_{t+1} y_\alpha",
    "caviar": r"q_t = \beta_1 + \beta_2 q_{t-1} + \beta_3 \max(r_{t-1}, 0) + \beta_4 \max(-r_{t-1}, 0), \quad \widehat{VaR}^{\alpha}_{t+1} = -q_{t+1}",
    "var_es": r"VaR^{\alpha}_t = -\inf\{x : P(r_t \le x \mid \mathcal{F}_{t-1}) \ge \alpha\}, \quad ES^{\alpha}_t = -E[r_t \mid r_t \le -VaR^{\alpha}_t, \mathcal{F}_{t-1}]",
    "regime": r"RV_t = \sqrt{\frac{252}{21}\sum_{i=0}^{20} r^2_{t-i}}",
    "kupiec": r"LR_{uc} = -2\ln\frac{(1-\alpha)^{T-x}\alpha^{x}}{(1-\hat\pi)^{T-x}\hat\pi^{x}} \sim \chi^2_1",
    "christoffersen": r"LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2",
    "z2": r"Z_2 = \sum_{t=1}^{T} \frac{r_t I_t}{T \alpha ES^{\alpha}_t} + 1",
    "fz0": r"L_{FZ0}(r_t, v_t, e_t) = -\frac{1}{\alpha e_t} \mathbb{1}\{r_t \le v_t\}(v_t - r_t) + \frac{v_t}{e_t} + \ln(-e_t) - 1",
    "dm": r"DM = \frac{\bar d}{\sqrt{\hat\sigma^2_d / T}} \sim N(0, 1), \quad d_t = L^A_t - L^B_t",
}
ESTIMATOR_STEP_COUNTS = {"hs": 3, "normal": 3, "garch_t": 5, "fhs": 5, "evt_pot": 5, "caviar": 3}


@dataclass(frozen=True)
class Explanation:
    summary: str
    formula: str
    steps: list[str]
    assumptions: str
    limitations: str
    reference: str


def _display_values(forecast: Forecast, alpha: float, amount: float, lang: str) -> dict[str, str]:
    """Format every number of `details` for the step-by-step text."""
    values = {
        "alpha": format_percent(alpha, lang, 1),
        "var_pct": format_percent(forecast.var, lang),
        "var_brl": format_currency(forecast.var * amount, lang),
        "es_pct": format_percent(forecast.es, lang) if forecast.es is not None else "",
        "es_brl": format_currency(forecast.es * amount, lang) if forecast.es is not None else "",
    }
    percent_keys = {"mu", "sigma", "sigma_next", "long_run_volatility", "quantile_pct", "q_next", "q_last", "last_return"}
    for key, value in forecast.details.items():
        if key == "tail_losses":
            continue
        if key in {"n", "k", "n_u"}:
            values[key] = format_number(value, lang, 0)
        elif key == "omega":
            values[key] = format_scientific(value, lang)
        elif key in percent_keys:
            values[key] = format_percent(value, lang, 3)
        elif key == "half_life":
            values[key] = format_number(value, lang, 1)
        else:
            values[key] = format_number(value, lang, 4)
    return values


def explain_estimator(name: str, forecast: Forecast, alpha: float, amount: float, lang: str) -> Explanation:
    """Explain one estimator's forecast with the portfolio's actual numbers."""
    values = _display_values(forecast, alpha, amount, lang)
    steps = [
        t(f"estimator.{name}.step{number}", lang, **values)
        for number in range(1, ESTIMATOR_STEP_COUNTS[name] + 1)
    ]
    return Explanation(
        summary=t(f"estimator.{name}.summary", lang),
        formula=FORMULAS[name],
        steps=steps,
        assumptions=t(f"estimator.{name}.assumptions", lang),
        limitations=t(f"estimator.{name}.limitations", lang),
        reference=t(f"estimator.{name}.reference", lang),
    )


def main_result_summary(var: float, es: float | None, alpha_var: float, alpha_es: float, amount: float, lang: str) -> list[str]:
    """Plain-language reading of the two headline numbers."""
    days = round((1 - alpha_var) * 100)
    lines = [t("main.var_reading", lang, days=days, amount=format_currency(var * amount, lang), percent=format_percent(var, lang))]
    if es is None:
        return lines + [t("main.es_missing", lang)]
    tail = format_number(100 * alpha_es, lang, 1)
    lines.append(t("main.es_reading", lang, tail=tail, amount=format_currency(es * amount, lang), percent=format_percent(es, lang)))
    return lines


def recommendation_reason(estimator_name: str, status: str, regime_name: str, rejected: list[str], lang: str) -> str:
    """Why this estimator was chosen, in one short paragraph."""
    others = ", ".join(rejected) if rejected else t("recommend.nobody", lang)
    return t(f"recommend.reason.{status}", lang, estimator=estimator_name, regime=regime_name, rejected=others)


def test_explanation(test: str, lang: str) -> Explanation:
    """Formula and reading of one backtest. It has no step list: the table holds the numbers."""
    return Explanation(
        summary=t(f"test.{test}.summary", lang),
        formula=FORMULAS[test],
        steps=[],
        assumptions=t(f"test.{test}.assumptions", lang),
        limitations=t(f"test.{test}.limitations", lang),
        reference=t(f"test.{test}.reference", lang),
    )
