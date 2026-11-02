"""Fixed parameters of the study (project PDF, chapters 2 and 5)."""

from pathlib import Path

# Data
START_DATE = "2008-01-01"
IBOVESPA_TICKER = "^BVSP"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Data validation
MAX_FILLED_GAP_DAYS = 5  # longer gaps stay missing instead of being invented
EXTREME_RETURN = 0.40  # absolute daily log return flagged for review, never altered
ADJUSTMENT_TOLERANCE = 0.001  # slack on the adj_close / close factor checks

# Artificial jumps confirmed against B3 COTAHIST files. The return of these days becomes zero.
KNOWN_PRICE_BREAKS = {
    ("UGPA3.SA", "2021-06-28"): "Yahoo adjustment error: B3 close went from 18.99 to 19.19",
    ("B3SA3.SA", "2011-05-09"): "Yahoo adjustment error: B3 close went from 11.88 to 12.02",
    ("JBSS32.SA", "2025-06-09"): "JBSS3 replaced by the JBSS32 BDR: price scale changed",
}

# Rolling window
ESTIMATION_WINDOW = 1000
REFIT_EVERY = 20

# Risk levels (tail probability)
ALPHA_VAR = 0.01
ALPHA_ES = 0.025

# Market regime (eq. 2.18)
RV_WINDOW = 21
TRADING_DAYS_PER_YEAR = 252
REGIME_PERCENTILE = 80
REGIME_MIN_HISTORY = 250

# Estimators
EVT_THRESHOLD_QUANTILE = 0.90
CAVIAR_SEED = 20260103
CAVIAR_CANDIDATES = 100
CAVIAR_REFINED_STARTS = 2
TAIL_GRID_POINTS = 50  # points describing each predicted tail, used by the Z2 simulation

# Backtests
Z2_REPLICATIONS = 1000
Z2_SEED = 20260101
TRAFFIC_LIGHT_WINDOW = 250
TRAFFIC_LIGHT_GREEN_MAX = 4
TRAFFIC_LIGHT_YELLOW_MAX = 9
SIGNIFICANCE_LEVEL = 0.05

# Experiment design (PDF, table 5.2)
EXPERIMENT_SEED = 20260102
PORTFOLIO_SIZES = (5, 10, 20)
WEIGHT_SCHEMES = ("equal", "dirichlet")
SECTOR_PROFILES = ("diversified", "commodity")
DRAWS_PER_CONFIGURATION = 3
COMMODITY_PROFILE_SHARE = 0.70  # share of assets drawn from commodity sectors

# Eligibility inside the universe, checked on the clean prices.
UNIVERSE_LAST_START = "2011-12-31"
UNIVERSE_MAX_MISSING_SHARE = 0.02

# Candidate universe of the experiment, ticker -> sector.
# Rule: a median daily traded value of at least R$ 10 million from 2024-11 to 2026-10.
# It uses stocks listed today, so it carries survivorship bias (PDF, section 5.7).
COMMODITY_SECTORS = frozenset({"oil_gas", "mining", "steel", "pulp_paper", "agribusiness", "chemicals"})
UNIVERSE = {
    "PETR3.SA": "oil_gas", "PETR4.SA": "oil_gas", "PRIO3.SA": "oil_gas",
    "CSAN3.SA": "oil_gas", "UGPA3.SA": "oil_gas",
    "VALE3.SA": "mining", "BRAP4.SA": "mining",
    "CSNA3.SA": "steel", "GGBR4.SA": "steel", "GOAU4.SA": "steel", "USIM5.SA": "steel",
    "SUZB3.SA": "pulp_paper", "KLBN4.SA": "pulp_paper",
    "SLCE3.SA": "agribusiness", "SMTO3.SA": "agribusiness", "BEEF3.SA": "agribusiness",
    "MBRF3.SA": "agribusiness", "JBSS32.SA": "agribusiness",
    "BRKM5.SA": "chemicals", "UNIP6.SA": "chemicals",
    "ITUB4.SA": "financials", "BBDC3.SA": "financials", "BBDC4.SA": "financials",
    "BBAS3.SA": "financials", "SANB11.SA": "financials", "ITSA4.SA": "financials",
    "B3SA3.SA": "financials", "PSSA3.SA": "financials",
    "AXIA3.SA": "utilities", "CMIG4.SA": "utilities", "CPLE3.SA": "utilities",
    "CPFE3.SA": "utilities", "EGIE3.SA": "utilities", "ENEV3.SA": "utilities",
    "ENGI11.SA": "utilities", "EQTL3.SA": "utilities", "ISAE4.SA": "utilities",
    "TAEE11.SA": "utilities", "SBSP3.SA": "utilities", "CSMG3.SA": "utilities",
    "WEGE3.SA": "industrials", "EMBJ3.SA": "industrials", "POMO4.SA": "industrials",
    "RAPT4.SA": "industrials", "TUPY3.SA": "industrials", "MOTV3.SA": "industrials",
    "ECOR3.SA": "industrials", "RENT3.SA": "industrials",
    "ABEV3.SA": "consumer", "LREN3.SA": "consumer", "MGLU3.SA": "consumer",
    "ALPA4.SA": "consumer", "GRND3.SA": "consumer", "MDIA3.SA": "consumer",
    "YDUQ3.SA": "consumer",
    "RADL3.SA": "health", "HYPE3.SA": "health", "FLRY3.SA": "health",
    "CYRE3.SA": "real_estate", "MRVE3.SA": "real_estate", "EZTC3.SA": "real_estate",
    "MULT3.SA": "real_estate", "ALOS3.SA": "real_estate", "DXCO3.SA": "real_estate",
    "TOTS3.SA": "technology", "TIMS3.SA": "telecom", "VIVT3.SA": "telecom",
}
