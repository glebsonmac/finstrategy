"""Streamlit interface: one page, three layers per block (summary, chart, "see more")."""

from __future__ import annotations

from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from riskbt import backtest, config, data, experiment
from riskbt.estimators import ESTIMATORS
from riskbt.explain import FORMULAS, Explanation, explain_estimator, main_result_summary, recommendation_reason, test_explanation
from riskbt.i18n import DEFAULT_LANGUAGE, LANGUAGES, format_currency, format_number, format_percent, t
from riskbt.recommend import NONE_PASSED
from riskbt.regimes import crisis_episodes

DEFAULT_PORTFOLIO = pd.DataFrame({"ticker": ["PETR4", "VALE3", "ITUB4"], "weight": [0.4, 0.3, 0.3]})
DEFAULT_AMOUNT = 100_000.0
REGIMES = ("all", "calm", "crisis")
LOSS_COLOR, ES_COLOR, MUTED_COLOR = "#2f7fd6", "#e0562a", "#8a8a8a"  # legible on light and dark themes
alt.data_transformers.disable_max_rows()


@st.cache_data(show_spinner=False)
def run_analysis(tickers: tuple[str, ...], weights: tuple[float, ...], as_of: str):
    """Download and analyze; the end date is exclusive, so today's unfinished session stays out."""
    prices, report = data.load_prices(list(tickers), config.START_DATE, as_of, as_of)
    if report.missing_tickers:
        raise data.DataError(",".join(report.missing_tickers))
    return experiment.analyze_portfolio(prices, dict(zip(tickers, weights))), report


def percent_axis(lang: str, title: str) -> alt.Axis:
    """Axis of percent values that follows the language's decimal separator."""
    expression = "replace(format(datum.value, '.1f'), '.', ',')" if lang == "pt" else "format(datum.value, '.1f')"
    return alt.Axis(title=title, labelExpr=expression)


def number_axis(lang: str, title: str) -> alt.Axis:
    """Axis of large values that follows the language's thousands separator."""
    expression = "replace(format(datum.value, ','), ',', '.')" if lang == "pt" else "format(datum.value, ',')"
    return alt.Axis(title=title, labelExpr=expression)


def date_axis(lang: str) -> alt.X:
    """Years only, because Vega-Lite would print month names in English."""
    return alt.X("date:T", axis=alt.Axis(title=t("backtest.chart_x", lang), format="%Y", tickCount="year"))


def show_chart(chart: alt.TopLevelMixin, caption: str) -> None:
    st.altair_chart(chart, width="stretch")
    st.caption(caption)


def show_explanation_parts(explanation: Explanation, lang: str) -> None:
    st.markdown(f"**{t('compare.assumptions', lang)}:** {explanation.assumptions}")
    st.markdown(f"**{t('compare.limitations', lang)}:** {explanation.limitations}")
    st.markdown(f"**{t('compare.reference', lang)}:** {explanation.reference}")


def read_inputs(lang: str) -> tuple[pd.DataFrame, float, bool]:
    st.subheader(t("input.header", lang))
    st.write(t("input.help", lang))
    table = st.data_editor(
        DEFAULT_PORTFOLIO, num_rows="dynamic", hide_index=True, key="portfolio",
        column_config={
            "ticker": st.column_config.TextColumn(t("input.ticker", lang)),
            "weight": st.column_config.NumberColumn(t("input.weight", lang), format="%.4f"),
        },
    )
    amount = st.number_input(t("input.amount", lang), min_value=0.0, value=DEFAULT_AMOUNT, step=1000.0)
    return table, amount, st.button(t("input.run", lang), type="primary")


def parse_portfolio(table: pd.DataFrame, amount: float, lang: str) -> tuple[tuple, tuple, str | None]:
    """Return tickers, weights and a translated error message, if any."""
    rows = table.dropna(subset=["ticker"])
    rows = rows[rows["ticker"].astype(str).str.strip() != ""]
    tickers = [str(code).strip().upper() for code in rows["ticker"]]
    tickers = [code if code.endswith(".SA") or code.startswith("^") else f"{code}.SA" for code in tickers]
    weights = [float(value) for value in rows["weight"].fillna(0.0)]
    if not tickers:
        return (), (), t("error.no_assets", lang)
    if len(set(tickers)) != len(tickers):
        return (), (), t("error.duplicate", lang)
    if min(weights) < 0:
        return (), (), t("error.negative", lang)
    if abs(sum(weights) - 1) > 1e-6:
        return (), (), t("error.weight_sum", lang, total=format_number(sum(weights), lang, 4))
    if amount <= 0:
        return (), (), t("error.amount", lang)
    return tuple(tickers), tuple(weights), None


def rejected_names(recommendation, lang: str) -> list[str]:
    table = recommendation.table
    return [t(f"estimator.{name}.name", lang) for name in ESTIMATORS if name != recommendation.estimator and name in table.index and not table.loc[name, "passed"]]


def returns_histogram(returns: pd.Series, var: float, es: float | None, lang: str) -> alt.LayerChart:
    days = min(len(returns), config.ESTIMATION_WINDOW)
    frame = pd.DataFrame({"value": 100 * returns.iloc[-days:].to_numpy()})
    bars = alt.Chart(frame).mark_bar(color=MUTED_COLOR).encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=60), axis=percent_axis(lang, t("main.chart_x", lang))),
        y=alt.Y("count():Q", title=t("main.chart_y", lang)),
    )
    marks = [{"position": -100 * var, "label": t("main.var_metric", lang), "color": LOSS_COLOR, "row": 4}]
    if es is not None:
        marks.append({"position": -100 * es, "label": t("main.es_metric", lang), "color": ES_COLOR, "row": 20})
    lines = alt.Chart(pd.DataFrame(marks)).mark_rule(strokeWidth=2).encode(
        x="position:Q", color=alt.Color("color:N", scale=None), tooltip=["label:N"],
    )
    labels = alt.Chart(pd.DataFrame(marks)).mark_text(align="right", dx=-4, baseline="top", fontWeight="bold").encode(
        x="position:Q", y=alt.Y("row:Q", scale=None), text="label:N", color=alt.Color("color:N", scale=None),
    )
    return (bars + lines + labels).properties(title=t("main.chart_title", lang, days=days), height=260)


def show_main_result(analysis, amount: float, lang: str) -> None:
    rec = analysis.recommendation
    forecast_var = analysis.latest[(rec.estimator, config.ALPHA_VAR)]
    forecast_es = analysis.latest[(rec.estimator, config.ALPHA_ES)]
    name = t(f"estimator.{rec.estimator}.name", lang)
    regime_name = t(f"regime.{rec.regime}", lang)
    st.subheader(t("main.header", lang))
    columns = st.columns([1, 1, 0.7, 1.7])  # the estimator's name is the longest value
    columns[0].metric(t("main.var_metric", lang), format_currency(forecast_var.var * amount, lang), format_percent(forecast_var.var, lang), delta_color="off", delta_arrow="off")
    es_text = format_currency(forecast_es.es * amount, lang) if forecast_es.es is not None else t("main.not_available", lang)
    es_delta = format_percent(forecast_es.es, lang) if forecast_es.es is not None else None
    columns[1].metric(t("main.es_metric", lang), es_text, es_delta, delta_color="off", delta_arrow="off")
    columns[2].metric(t("main.regime_metric", lang), regime_name)
    columns[3].metric(t("main.estimator_metric", lang), name)
    for line in main_result_summary(forecast_var.var, forecast_es.es, config.ALPHA_VAR, config.ALPHA_ES, amount, lang):
        st.write(line)
    st.write(t("main.horizon", lang))
    show_recommendation(rec, name, regime_name, lang)
    show_chart(returns_histogram(analysis.returns, forecast_var.var, forecast_es.es, lang), t("main.chart_caption", lang))
    with st.expander(t("app.see_more", lang)):
        st.write(t("main.more_var_es", lang))
        st.latex(FORMULAS["var_es"])
        st.write(t("main.more_regime", lang))
        st.latex(FORMULAS["regime"])
        st.write(t("main.more_regime_value", lang, regime=regime_name))
        st.markdown(f"**{t('compare.limitations', lang)}:** {t('main.more_limits', lang)}")
        st.markdown(f"**{t('compare.reference', lang)}:** {t('main.more_reference', lang)}")


def show_recommendation(rec, name: str, regime_name: str, lang: str) -> None:
    reason = recommendation_reason(name, rec.status, regime_name.lower(), rejected_names(rec, lang), lang)
    st.markdown(f"**{t('recommend.why', lang)}:** {reason}")
    if rec.status == NONE_PASSED:
        st.warning(t("recommend.warning_none", lang))
    if rec.low_power:
        st.warning(t("recommend.warning_low_power", lang))


def status_text(rec, name: str, lang: str) -> str:
    row = rec.table.loc[name]
    return t("compare.passed", lang) if row["passed"] else t("compare.failed", lang, count=int(row["tests_failed"]))


def comparison_table(analysis, amount: float, lang: str) -> pd.DataFrame:
    rec, latest = analysis.recommendation, analysis.latest
    money = lambda value: format_currency(value * amount, lang) if value is not None else t("tests.not_applicable", lang)
    rows = []
    for name in ESTIMATORS:
        label = t(f"estimator.{name}.name", lang)
        rows.append({
            t("compare.estimator", lang): t("compare.recommended", lang, name=label) if name == rec.estimator else label,
            t("compare.var99", lang): money(latest[(name, config.ALPHA_VAR)].var),
            t("compare.var975", lang): money(latest[(name, config.ALPHA_ES)].var),
            t("compare.es975", lang): money(latest[(name, config.ALPHA_ES)].es),
            t("compare.status", lang): status_text(rec, name, lang),
        })
    return pd.DataFrame(rows)


def comparison_chart(analysis, amount: float, lang: str) -> alt.Chart:
    rows = []
    for name in ESTIMATORS:
        label = t(f"estimator.{name}.name", lang)
        rows.append({"estimator": label, "measure": t("compare.var99", lang), "loss": analysis.latest[(name, config.ALPHA_VAR)].var * amount})
        es = analysis.latest[(name, config.ALPHA_ES)].es
        if es is not None:
            rows.append({"estimator": label, "measure": t("compare.es975", lang), "loss": es * amount})
    return alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        y=alt.Y("estimator:N", sort=None, axis=alt.Axis(title=None, labelLimit=260)),
        yOffset="measure:N",
        x=alt.X("loss:Q", axis=number_axis(lang, t("compare.loss", lang))),
        color=alt.Color("measure:N", title=t("compare.measure", lang), scale=alt.Scale(range=[ES_COLOR, LOSS_COLOR])),
    ).properties(title=t("compare.chart_title", lang), height=320)


def show_estimator_details(name: str, analysis, amount: float, lang: str) -> None:
    with st.expander(f"{t('app.see_more', lang)}: {t(f'estimator.{name}.name', lang)}"):
        explanations = {alpha: explain_estimator(name, analysis.latest[(name, alpha)], alpha, amount, lang) for alpha in (config.ALPHA_VAR, config.ALPHA_ES)}
        first = explanations[config.ALPHA_ES]
        st.write(first.summary)
        st.latex(first.formula)
        st.markdown(f"**{t('compare.steps', lang)}**")
        for alpha, explanation in explanations.items():
            st.markdown(f"*{t('compare.level', lang, level=format_percent(1 - alpha, lang, 1))}*")
            st.markdown("\n".join(f"{number}. {step}" for number, step in enumerate(explanation.steps, start=1)))
        show_explanation_parts(first, lang)


def show_comparison(analysis, amount: float, lang: str) -> None:
    st.subheader(t("compare.header", lang))
    st.write(t("compare.summary", lang))
    st.dataframe(comparison_table(analysis, amount, lang), hide_index=True, width="stretch")
    show_chart(comparison_chart(analysis, amount, lang), t("compare.chart_caption", lang))
    for name in ESTIMATORS:
        show_estimator_details(name, analysis, amount, lang)


def violations_chart(selected: pd.DataFrame, regimes: pd.Series, lang: str) -> alt.LayerChart:
    frame = pd.DataFrame({
        "date": selected["date"], "return": 100 * selected["return"], "limit": -100 * selected["var"], "hit": selected["hit"],
    })
    first, last = frame["date"].min(), frame["date"].max()
    episodes = crisis_episodes(regimes.loc[first:last])
    bands = alt.Chart(episodes).mark_rect(color=MUTED_COLOR, opacity=0.35).encode(x="start:T", x2="end:T")
    points = alt.Chart(frame).mark_circle(size=6, color=MUTED_COLOR, opacity=0.5).encode(
        x=date_axis(lang), y=alt.Y("return:Q", axis=percent_axis(lang, t("backtest.chart_y", lang))),
    )
    limit = alt.Chart(frame).mark_line(color=LOSS_COLOR, strokeWidth=1.2).encode(x="date:T", y="limit:Q")
    hits = alt.Chart(frame[frame["hit"]]).mark_point(shape="cross", color=ES_COLOR, size=60, filled=True).encode(
        x="date:T", y="return:Q", tooltip=["date:T", alt.Tooltip("return:Q", format=".2f")],
    )
    return (bands + points + limit + hits).properties(title=t("backtest.chart_title", lang), height=340)


def traffic_light_chart(lights: pd.DataFrame, lang: str) -> alt.LayerChart:
    frame = lights.rename_axis("date").reset_index()
    line = alt.Chart(frame).mark_line(color=LOSS_COLOR).encode(
        x=date_axis(lang), y=alt.Y("violations:Q", title=t("light.count", lang)),
    )
    thresholds = pd.DataFrame({"limit": [config.TRAFFIC_LIGHT_GREEN_MAX + 0.5, config.TRAFFIC_LIGHT_YELLOW_MAX + 0.5]})
    rules = alt.Chart(thresholds).mark_rule(strokeDash=[4, 4], color=MUTED_COLOR).encode(y="limit:Q")
    return (line + rules).properties(title=t("light.chart_title", lang), height=220)


def show_traffic_light(analysis, name: str, lang: str) -> None:
    st.markdown(f"**{t('light.header', lang)}**")
    level = analysis.forecasts[(analysis.forecasts["estimator"] == name) & (analysis.forecasts["alpha"] == config.ALPHA_VAR)]
    lights = backtest.traffic_light(level.set_index("date")["hit"])
    if lights.empty:
        st.info(t("light.too_short", lang))
        return
    last = lights.iloc[-1]
    st.write(t("light.summary", lang, estimator=t(f"estimator.{name}.name", lang), zone=t(f"zone.{last['zone']}", lang), count=int(last["violations"])))
    show_chart(traffic_light_chart(lights, lang), t("light.chart_caption", lang))


def p_value_text(value: float, lang: str) -> str:
    if np.isnan(value):
        return t("tests.not_applicable", lang)
    key = "tests.value_pass" if value >= config.SIGNIFICANCE_LEVEL else "tests.value_fail"
    return t(key, lang, value=format_number(value, lang, 3))


def tests_table(tests: pd.DataFrame, alpha: float, lang: str) -> pd.DataFrame:
    """One row per estimator and regime, with the verdict in words next to each p-value."""
    subset = tests[(tests["alpha"] == alpha) & (tests["comparator"] == "") & (tests["test"] != "traffic_light")]
    rows = []
    for name in ESTIMATORS:
        for regime in REGIMES:
            row = subset[(subset["estimator"] == name) & (subset["regime"] == regime)].set_index("test")
            cell = lambda test, column: row[column].get(test, np.nan)
            n_obs, hits = int(cell("kupiec", "n_obs")), int(cell("kupiec", "violations"))
            fz0 = cell("mean_fz0_loss", "statistic")
            rows.append({
                t("compare.estimator", lang): t(f"estimator.{name}.name", lang),
                t("tests.regime", lang): t(f"regime.{regime}", lang),
                t("tests.days", lang): format_number(n_obs, lang, 0),
                t("tests.rate", lang): format_percent(hits / n_obs, lang) if n_obs else t("tests.not_applicable", lang),
                t("tests.kupiec", lang): p_value_text(cell("kupiec", "p_value"), lang),
                t("tests.christoffersen", lang): p_value_text(cell("christoffersen", "p_value"), lang),
                t("tests.z2", lang): p_value_text(cell("z2", "p_value"), lang),
                t("tests.fz0", lang): format_number(fz0, lang, 3) if not np.isnan(fz0) else t("tests.not_applicable", lang),
                t("tests.quantile", lang): format_number(cell("mean_quantile_loss", "statistic"), lang, 5),
            })
    return pd.DataFrame(rows)


def dm_table(tests: pd.DataFrame, regime: str, lang: str) -> pd.DataFrame:
    subset = tests[(tests["alpha"] == config.ALPHA_ES) & (tests["regime"] == regime) & (tests["test"] == "diebold_mariano_fz0")]
    rows = []
    for _, row in subset.iterrows():
        first, second = t(f"estimator.{row['estimator']}.name", lang), t(f"estimator.{row['comparator']}.name", lang)
        significant = row["p_value"] < config.SIGNIFICANCE_LEVEL
        better = (first if row["statistic"] < 0 else second) if significant else t("tests.dm_no_difference", lang)
        rows.append({
            t("tests.dm_first", lang): first, t("tests.dm_second", lang): second,
            t("tests.dm_statistic", lang): format_number(row["statistic"], lang, 2),
            t("tests.dm_p", lang): format_number(row["p_value"], lang, 3),
            t("tests.dm_better", lang): better,
        })
    return pd.DataFrame(rows)


def show_tests(analysis, alpha: float, lang: str) -> None:
    st.markdown(f"**{t('tests.header', lang)}**")
    st.write(t("tests.summary", lang))
    st.dataframe(tests_table(analysis.tests, alpha, lang), hide_index=True, width="stretch")
    with st.expander(t("app.see_more", lang)):
        for test in ("kupiec", "christoffersen", "z2", "fz0", "dm"):
            explanation = test_explanation(test, lang)
            st.write(explanation.summary)
            st.latex(explanation.formula)
            show_explanation_parts(explanation, lang)
        regime = analysis.recommendation.regime
        st.markdown(f"**{t('tests.dm_title', lang)}**")
        st.caption(t("tests.dm_caption", lang, regime=t(f"regime.{regime}", lang)))
        st.dataframe(dm_table(analysis.tests, regime, lang), hide_index=True, width="stretch")
        st.write(t("tests.care", lang))


def show_backtest(analysis, lang: str) -> None:
    st.subheader(t("backtest.header", lang))
    st.write(t("backtest.summary", lang))
    names = list(ESTIMATORS)
    columns = st.columns(2)
    name = columns[0].selectbox(
        t("backtest.estimator", lang), names, index=names.index(analysis.recommendation.estimator),
        format_func=lambda option: t(f"estimator.{option}.name", lang),
    )
    labels = {config.ALPHA_VAR: t("backtest.level_var", lang), config.ALPHA_ES: t("backtest.level_es", lang)}
    alpha = columns[1].selectbox(t("backtest.level", lang), list(labels), format_func=labels.get)
    forecasts = analysis.forecasts
    selected = forecasts[(forecasts["estimator"] == name) & (forecasts["alpha"] == alpha)]
    show_chart(violations_chart(selected, analysis.regimes, lang), t("backtest.chart_caption", lang))
    st.write(t(
        "backtest.violations", lang, count=int(selected["hit"].sum()), days=format_number(len(selected), lang, 0),
        rate=format_percent(selected["hit"].mean(), lang), expected=format_percent(alpha, lang, 1),
    ))
    show_traffic_light(analysis, name, lang)
    show_tests(analysis, alpha, lang)


def list_events(events: list[dict], tickers: tuple[str, ...], lang: str) -> str:
    return "; ".join(
        f"{event['ticker']} {event['date']} ({format_number(event['log_return'], lang, 2)})"
        for event in events if event["ticker"] in tickers
    )


def show_data(analysis, report: data.DataReport, tickers: tuple[str, ...], as_of: str, lang: str) -> None:
    st.subheader(t("data.header", lang))
    returns = analysis.returns
    st.write(t(
        "data.summary", lang, as_of=as_of, first=returns.index[0].strftime("%Y-%m-%d"),
        last=returns.index[-1].strftime("%Y-%m-%d"), days=format_number(len(returns), lang, 0),
    ))
    total = lambda counts: sum(count for ticker, count in counts.items() if ticker in tickers)
    breaks = [event for event in report.breaks_removed if event["ticker"] in tickers]
    st.write(t(
        "data.corrections", lang, filled=total(report.gaps_filled), stale=total(report.zero_volume_days),
        inconsistent=total(report.inconsistent_prices), breaks=len(breaks),
    ))
    frame = pd.DataFrame({"date": returns.index, "value": 100 * np.exp(returns.cumsum()).to_numpy()})
    chart = alt.Chart(frame).mark_line(color=LOSS_COLOR).encode(
        x=date_axis(lang), y=alt.Y("value:Q", axis=number_axis(lang, t("data.chart_y", lang))),
    ).properties(title=t("data.chart_title", lang), height=240)
    show_chart(chart, t("data.chart_caption", lang))
    with st.expander(t("app.see_more", lang)):
        st.write(t("data.more_cleaning", lang, days=config.MAX_FILLED_GAP_DAYS))
        extreme = list_events(report.extreme_returns, tickers, lang)
        st.write(t("data.more_extreme", lang, items=extreme) if extreme else t("data.more_no_extreme", lang))
        if breaks:
            st.write(t("data.more_breaks", lang, items=list_events(breaks, tickers, lang)))
        st.markdown(f"**{t('compare.limitations', lang)}:** {t('data.more_limits', lang)}")
        st.markdown(f"**{t('compare.reference', lang)}:** {t('data.more_reference', lang)}")


def show_download(analysis, lang: str) -> None:
    st.subheader(t("download.header", lang))
    st.download_button(t("download.button", lang), experiment.export_csv(analysis), file_name="riskbt_portfolio.csv", mime="text/csv")
    st.caption(t("download.caption", lang))


def compute(tickers: tuple, weights: tuple, as_of: str, lang: str):
    """Run the analysis, turning failures into translated messages. Returns None on failure."""
    try:
        with st.spinner(t("input.running", lang)):
            return run_analysis(tickers, weights, as_of)
    except data.DataError as error:
        st.error(t("error.unknown_tickers", lang, tickers=", ".join(str(error).split(","))))
    except experiment.InsufficientHistory as error:
        st.error(t("error.too_short", lang, days=error.days, needed=error.needed))
    except (OSError, ValueError, KeyError):
        st.error(t("error.download", lang))
    return None


def main() -> None:
    st.set_page_config(page_title="riskbt", layout="wide")
    current = st.session_state.get("lang", DEFAULT_LANGUAGE)
    lang = st.selectbox(t("app.language", current), list(LANGUAGES), index=list(LANGUAGES).index(DEFAULT_LANGUAGE), format_func=LANGUAGES.get, key="lang")
    st.title(t("app.title", lang))
    st.info(t("app.notice", lang))
    table, amount, pressed = read_inputs(lang)
    tickers, weights, error = parse_portfolio(table, amount, lang)
    if pressed:
        if error:
            st.error(error)
        else:
            st.session_state["request"] = (tickers, weights, date.today().isoformat())
    if "request" not in st.session_state:
        st.write(t("input.start_hint", lang))
        return
    tickers, weights, as_of = st.session_state["request"]
    result = compute(tickers, weights, as_of, lang)
    if result is None:
        return
    analysis, report = result
    show_main_result(analysis, amount, lang)
    show_comparison(analysis, amount, lang)
    show_backtest(analysis, lang)
    show_data(analysis, report, tickers, as_of, lang)
    show_download(analysis, lang)


main()
