# finstrategy

Ação + put de proteção: dados do Yahoo Finance, Black-Scholes, Monte Carlo (GARCH-t) e
Deep Q-Network para apoiar decisão de entrada na ação e timing de saída do derivativo.
Primeiro recorte: setor elétrico — veja `plan.md` (planejamento) e `tasks.md`
(checklist de implementação e log de execução).

## Como rodar

### 1. Instalar (uma vez)

```bash
cd /Users/josealecrim/Documents/mestrado/disertacao/finstrategy
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Ativar o ambiente (sempre, em cada terminal novo)

```bash
source .venv/bin/activate
```

### 3. Rodar o frontend (forma mais prática de usar o sistema)

```bash
streamlit run src/finstrategy/frontend/app.py
```

Abre em `http://localhost:8501`. Telas disponíveis:

- **Ações** — três formas de adicionar ação à watchlist, em abas: **lista local**
  (~500 ações do S&P 500, filtra instantâneo, sem internet — tipo investing.com),
  **busca ao vivo no Yahoo** (qualquer ação, inclusive fora do S&P 500/B3/ADR) e
  **colar tickers** direto. Em qualquer caso, o sistema busca preço, fundamentalista e
  elegibilidade de derivativo sozinho ao confirmar a adição.
- **Configuração** — edita `AnalysisSettings` (prazos, limites de perda, parâmetros do
  Monte Carlo e das DQNs) sem tocar em código.
- **Rodar agora** — roda a análise completa (ação + put de proteção + DQN de entrada)
  para os tickers escolhidos e gera o PDF para download.
- **Histórico & Relatórios** — cotação e fundamentalista em cartões claros, com botão
  para gerar um relatório novo na hora.
- **Coletor** — liga/desliga o loop de coleta contínua durante o pregão, com status ao
  vivo (ativo/parado, última verificação, o que foi coletado no último ciclo).
- **Modelos** — treina e avalia a DQN de entrada (honesto: treino/teste separados, com
  intervalo de confiança e p-valor contra benchmarks).

Primeira vez com a watchlist vazia: na tela Ações, clique em "Popular com as 20 maiores
ações dos EUA" ou "Popular com a semente do setor elétrico", ou rode:

```bash
PYTHONPATH=src python -c "from finstrategy.data.watchlist import ensure_seeded; ensure_seeded()"
```

### 4. Rodar o coletor contínuo sem UI (fica coletando durante o pregão)

```bash
PYTHONPATH=src python -m finstrategy.collector.scheduler
```

### 5. Coleta noturna (um CSV só, sem precisar do Streamlit aberto)

Pensado para rodar via `cron`/agendador num computador diferente de onde o projeto foi
desenvolvido — não depende do app estar aberto, nem do mercado estar aberto (fora do
pregão usa o último preço negociado, igual ao resto do sistema). Coleta preço +
indicadores fundamentalistas de cada ticker e grava **um CSV por rodada** (mais um
`latest.csv` sempre sobrescrito, caminho estável para outro processo achar a coleta
mais recente).

```bash
# usa config/watchlist.yaml se existir, senão as 20 maiores ações dos EUA (ao vivo por valor de mercado)
python scripts/collect_overnight.py

# ou uma lista específica
python scripts/collect_overnight.py --tickers AAPL,MSFT,DUK,TAEE11.SA

# salva em data_store/overnight/AAAA-MM-DD_essencial.csv e data_store/overnight/latest.csv
```

Exemplo de agendamento com `cron` (roda todo dia às 23h, horário local):

```cron
0 23 * * * cd /caminho/do/projeto/finstrategy && .venv/bin/python scripts/collect_overnight.py >> data_store/overnight/cron.log 2>&1
```

### 6. Rodar os testes

```bash
pytest
```

## Arquitetura

### Visão geral — como os dados fluem

```
 watchlist.yaml  ──┐
 (ou --tickers)    │
                    ▼
     ┌─────────────────────────┐        ┌──────────────────────────┐
     │  data/  (camada de dados) │  ───▶  │  pricing/  (precificação) │
     │  Yahoo: preço, opções,    │        │  GARCH-t Monte Carlo,     │
     │  fundamentalista          │        │  Black-Scholes            │
     └─────────────────────────┘        └──────────────────────────┘
                    │                                   │
                    ▼                                   ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  strategy/  — menu de put de proteção + janela de saída (BS)  │
     └─────────────────────────────────────────────────────────────┘
                    │
                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  rl/  — DQN de entrada (ação) e DQN de saída (derivativo)      │
     └─────────────────────────────────────────────────────────────┘
                    │
                    ▼
     ┌──────────────────────┐        ┌───────────────────────────┐
     │  reporting/            │  ───▶  │  PDF + painel técnico       │
     │  painel + texto simples│        │  (para quem não é técnico)  │
     └──────────────────────┘        └───────────────────────────┘

 Duas formas de orquestrar tudo isso:
   collector/scheduler.py   → loop contínuo durante o pregão (controlado pelo Streamlit)
   collector/overnight.py   → uma rodada só, sem UI (cron) → CSV único

 Tudo isso é editado/visualizado pelo frontend/app.py (Streamlit) e gravado em CSV
 por collector/csv_store.py.
```

### `src/finstrategy/data/` — camada de dados

| Arquivo | O que faz |
|---|---|
| `base.py` | Define o "contrato" `MarketDataProvider` (preço, dividendo, opções, taxa livre de risco) que qualquer fonte de dado precisa implementar, mais os tipos `SymbolMatch`/`SymbolInfo` usados na busca de ticker. |
| `errors.py` | `MarketDataError` — o erro usado em todo o sistema para "esse dado não existe/não está disponível agora" (ex.: ação sem opção). Não é um bug, é um estado esperado. |
| `yahoo_provider.py` | Implementação do `MarketDataProvider` via `yfinance`: histórico de preço, preço atual, dividendo, taxa livre de risco (`^IRX`), cadeia de puts/calls. Extraído do notebook original, sem mudar o comportamento. |
| `mt5_provider.py` | Placeholder do MetaTrader5 — levanta `NotImplementedError` até a decisão (plan.md, seção 0) de entrar no escopo ou não. |
| `symbol_search.py` | Busca/valida ticker **ao vivo** no Yahoo: `search_symbols` (autocomplete por nome/código) e `validate_symbol`/`validate_symbols` (confirma que um ticker existe e tem preço antes de entrar na watchlist). |
| `local_stock_list.py` | Mesma ideia, mas **sem rede**: filtra `config/us_stock_list.csv` (S&P 500) em memória — instantâneo, usado na aba "Lista" do frontend. |
| `fundamentals_provider.py` | Indicadores fundamentalistas normalizados (P/L, ROE, dívida/patrimônio, dividend yield, etc.) a partir do `yfinance`. Cuidado documentado aqui: `dividendYield` do Yahoo vem em escala diferente dos outros campos de percentual — já normalizado. |
| `watchlist.py` | O "banco de dados" de ações que o resto do sistema analisa (`config/watchlist.yaml`). `add_tickers` valida + busca fundamentalista automaticamente; `ensure_seeded` popula a partir de uma semente (setor elétrico ou top-20 EUA) na primeira vez. |

### `src/finstrategy/pricing/` — precificação

| Arquivo | O que faz |
|---|---|
| `garch_montecarlo.py` | Ajusta um GARCH(1,1) com inovações t de Student ao histórico real de retornos (`fit_garch_t`) e simula o preço da ação no vencimento (`simulate_prices_at_expiration`, só o preço final; `simulate_price_paths`, a trajetória dia a dia — usada pela DQN de saída). Mede risco (`risk_metrics`: VaR, ES, P(perda)). |
| `black_scholes.py` | Preço e Greeks (delta, gamma, theta, vega, rho) de opções europeias (`price_and_greeks`, escalar; `price_and_greeks_array`, vetorizado para repreçar muitos caminhos de uma vez) e volatilidade implícita a partir do preço de mercado (`implied_volatility`). |

### `src/finstrategy/strategy/` — a estratégia em si

| Arquivo | O que faz |
|---|---|
| `protective_put.py` | O núcleo "ação + put de proteção": monta o menu de strikes líquidas (`prepare_liquid_puts`), calcula custo/perda máxima/break-even de cada uma (`build_protective_put_plan`), recomenda a melhor (`recommend_strike`) e junta tudo (`analyze_ticker`, o ponto de entrada usado pelo resto do sistema). |
| `exit_timing_bs.py` | Quando vender a put antes do vencimento? Duas curvas: `time_value_curve` (Black-Scholes dia a dia com o preço da ação fixo — isola o efeito do tempo passando) e `expected_monte_carlo_value_curve` (valor esperado via GARCH Monte Carlo). `recommend_exit_day` é a regra fixa (vender quando o valor-tempo cair abaixo de um piso) usada como referência para a DQN de saída. |

### `src/finstrategy/rl/` — Deep Q-Network

| Arquivo | O que faz |
|---|---|
| `entry_dqn.py` | Vale comprar a ação hoje ou esperar? `QNetwork` (MLP em NumPy, Double DQN), `EntryTimingEnvironment`, `train_entry_dqn` e `evaluate_entry_timing` (avaliação honesta: treino/teste separados, bootstrap de intervalo de confiança, p-valor contra 3 referências). Extraído do notebook original. |
| `exit_dqn.py` | Já tenho a put, mantenho ou vendo agora? Reusa a mesma `QNetwork`/Double DQN do `entry_dqn.py`; o que muda é o ambiente (estado com delta/theta/vega/P&L não realizado) e a recompensa (valor de vender agora menos valor de manter até o vencimento — ensina o agente a não abrir mão da proteção quando ela é útil). Treina sobre trajetórias simuladas pelo GARCH (não há histórico real de preço de opção dia a dia no Yahoo). |

### `src/finstrategy/collector/` — coleta e agendamento

| Arquivo | O que faz |
|---|---|
| `market_session.py` | Calendário de pregão via `pandas_market_calendars` (BVMF para B3, NYSE para EUA) — `is_market_open`, `any_market_open`. |
| `csv_store.py` | Escrita append-safe em CSV, particionada por ticker/dia (`market_csv_path`, `fundamentals_csv_path`, `model_runs_csv_path`, `reports_csv_path`). |
| `scheduler.py` | O coletor **contínuo**: `run_forever` fica rodando o dia inteiro, só ativo durante o pregão, controlado pela tela "Coletor" do Streamlit. Escreve `data_store/collector_status.json` a cada ciclo (`write_status`/`read_status`) para o frontend mostrar status ao vivo. Qualquer erro vira status, nunca derruba a thread. |
| `overnight.py` | O coletor de **uma rodada só** (`run_overnight_collection`), pensado para `cron`, sem precisar do Streamlit nem do mercado aberto — grava um CSV único com cotação + fundamentalista de cada ticker. Usado por `scripts/collect_overnight.py`. |

### `src/finstrategy/reporting/` — relatórios

| Arquivo | O que faz |
|---|---|
| `panel.py` | `build_panel_dataframe` (tabela técnica, uma linha por ativo) e `build_plain_language_summary` (texto direto, sem jargão, por ativo — a base do PDF). |
| `pdf_report.py` | `generate_pdf_report` — PDF com capa + uma página por ativo, usando `reportlab`. |

### `src/finstrategy/frontend/app.py` — o app Streamlit

Um arquivo só, dividido por tela via `st.sidebar.radio` (ver seção "Como rodar" acima
para o que cada tela faz). `run_full_pipeline`/`render_pipeline_results`/
`generate_and_offer_pdf` são helpers compartilhados entre as telas "Rodar agora" e
"Histórico & Relatórios", para não duplicar a lógica de "rodar tudo + montar painel +
gerar PDF".

### `config_model.py` — configuração

`AnalysisSettings`: um dataclass só com todos os parâmetros do pipeline (prazos,
limites de risco, parâmetros do Monte Carlo e das DQNs). `load_settings`/
`save_settings` leem/escrevem `config/settings.yaml` (ou a semente
`settings.example.yaml`, se o primeiro ainda não existir).

### `config/` — sementes e listas versionadas

| Arquivo | O que é |
|---|---|
| `settings.example.yaml` | Semente de `AnalysisSettings` ("reset de fábrica"). |
| `tickers_setor_eletrico.yaml` | Semente da watchlist: setor elétrico (foco original da dissertação). |
| `us_top20_megacaps.yaml` | Semente da watchlist: as 20 maiores ações dos EUA por valor de mercado (ranqueadas ao vivo — ver `scripts/rank_us_megacaps.py`). Também o default do coletor noturno. |
| `us_stock_list.csv` | ~503 ações do S&P 500 (ticker/nome/setor), para a aba "Lista" do frontend e `data/local_stock_list.py` — navegação sem rede. |
| `watchlist.yaml` / `settings.yaml` | Gerados em runtime (gitignored) — o estado real do seu projeto, não versionado. |

### `scripts/` — CLIs standalone (não dependem do Streamlit)

| Arquivo | O que faz |
|---|---|
| `collect_overnight.py` | Wrapper de linha de comando para `collector/overnight.py` — a coleta noturna (ver seção 5 acima). |
| `rank_us_megacaps.py` | Reavalia o top-20 EUA ao vivo no Yahoo; usado para gerar/atualizar `config/us_top20_megacaps.yaml`. |

### `tests/`

Um arquivo de teste por módulo de `src/finstrategy/` (mesma nomenclatura), quase todos
usando mocks/fakes para não depender de rede nem do mercado estar aberto — 129 testes
no total. Rodar com `pytest`.
