# tasks.md — VaR and ES backtesting system

Spec: `projeto_mestrado_VaR_ES.pdf`. Build rules and work order: `CLAUDE.md`.
The previous project's log (put hedging, DQN) is in git history, commit d9b8825.

- [x] Step 1: `config.py`, `data.py`, `portfolio.py`, frozen data version 2026-10-30
- [x] Step 2: `regimes.py` + `scripts/check_regimes.py`
- [x] Step 3: `hs`, `normal`, `garch_t`, `fhs` + tests
- [x] Step 4: `evt_pot`, `caviar` + tests
- [x] Step 5: `rolling.py`, `backtest.py` + no-look-ahead test
- [x] Step 6: `recommend.py`, `i18n.py`, `locales/`, `explain.py`
- [x] Step 7: `app.py`, tested in Chrome with real data (editing the portfolio table not verified there)
- [x] Step 8: `experiment.py`, `scripts/run_experiment.py`, `results/2026-11-02/`
- [x] Step 9: README

## Choices made without a ruling yet (see the summary given to the user)

- Zero-volume quotes are gaps; three confirmed breaks in `config.KNOWN_PRICE_BREAKS` become zero returns.
- "Commodity" profile: 70% of the assets from commodity sectors. Universe is a provisional list.
- HS and normal are also refitted only every 20 days ("same frequency for all", PDF section 5.3).
- Recommendation gate uses the 97.5% level, because 99% has too few violations per regime.

## Data findings (version 2026-10-30)

Checked against B3 COTAHIST files in `data_store/cotahist/` (they end on 2026-10-16).

- Yahoo errors confirmed: UGPA3 2021-06-28 (Yahoo +78% log return, B3 +1%);
  B3SA3 2011-05-09 (Yahoo -108%, B3 +1.2%).
- Stale prices with zero volume: SUZB3 100% of days until 2017; UGPA3 about 100% in
  2008-2010; ENGI11 about 60% in 2011-2015; GGBR4 and BBDC4 high in 2008-2010.
- JBSS32 2025-06-09: price doubles when JBSS3 becomes JBSS32; not an economic return.
- EQTL3 2010-08-25: B3 also shows 18.10 to 10.90, cause not established.
- 2026-10-26: real event, not an error. Ibovespa closed at 176,608 (-13.88%), matching press
  reports of the circuit breaker after the election. Outside COTAHIST coverage.

## End-to-end validation, 2026-11-03 (commit 10fbf90, macOS arm64, Python 3.12.14)

Verdict: not ready. The computational core passes; three items remain open.

- Passed: 176 tests; data version 2026-11-03 (4668 x 68, 2008-01-02 to 2026-10-30, hash checked);
  experiment in `results/2026-11-03/` (5 files, 37 portfolios, mean 9.1 s, max 10.7 s per portfolio);
  0 of 5944 GARCH refits not converged; rerun of one portfolio on this machine identical.
- Not tested: the page in Chrome (browser tool unavailable), including editing the portfolio table
  and the CSV button; reproducibility across machines (`data/frozen/2026-10-30` and
  `results/2026-11-02` are not on this machine). Run without the browser (Streamlit AppTest):
  all blocks in both languages, every estimator and level, no exception.
- CAViaR: fits with b2 near 1 and b4 > 0 lower VaR after a loss. VaR went to 0.33% the day after a
  24% loss and to -0.07% on 2026-10-29 (n5_equal_diversified_1). b4 > 0 in 20% of the 2026 refits.
  VaR 99% below VaR 97.5% on 3.18% of CAViaR days, 0% for the other estimators.
- Crisis results depend on the last days of the sample. With data up to 2026-10-23, 24 of 37
  portfolios pass in crisis; with the data up to 2026-10-30, 6 pass (Z2 is what changes).
- No code was changed by this validation; the proposed fixes await a decision.
- Metadata records neither the Python version nor the git commit; dependencies are not pinned.
- `data_store/collector_status.json` stays modified in git; riskbt does not use it.

## Chrome test, 2026-11-03 (Chrome 156 headless, pt-BR locale, separate temporary profile)

Interface, messages, README and one script message changed; estimators, tests, parameters and
`results/` did not. The default portfolio gives the same numbers before and after. 178 tests pass.

- Verified in Chrome: editing, adding and removing rows, all input errors, short history (BRST3),
  both languages, every "see more" (13 formulas, no error), estimator and level selectors, CSV
  button, light and dark themes, phone width (no horizontal scroll).
- Fixed: a dot typed in a weight was read as a thousands separator in pt-BR browsers (0.5 became 5);
  switching language erased table edits; results stayed on screen with nothing to say which
  portfolio they belonged to; 18 rows of the test table hid behind an inner scroll bar; "1 violations";
  "Failed 3" with no noun; recommended estimator's name cut in Portuguese; "Deploy" button visible; second experiment run on one day ended in a traceback.
- Open: the project's `.venv` predates this system and lacks `statsmodels` and `arch`, so the page
  fails there until `pip install -e ".[dev]"` is run in it. Not changed, it is the user's environment.
- Not tested: a real Yahoo outage; the extension-based browser tool, not installed.
