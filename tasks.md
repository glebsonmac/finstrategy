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
