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

### 5b. Coleta diária via MT5 (simulador/demo)

Pensado para quando a fonte de preço/opções é um terminal MetaTrader5 (simulador ou
conta demo) em vez do Yahoo — por exemplo, para repetir a coleta por vários dias
seguidos até acumular histórico suficiente para o GARCH Monte Carlo e o Black-Scholes.
Só funciona em **Python nativo do Windows**, com o terminal MT5 **aberto e logado**
(é o pacote `MetaTrader5` que exige isso, não o resto do projeto).

**Instalar no Windows** (passos 1–2 da seção "Como rodar" acima são para
macOS/Linux — no Windows, o `venv` ativa diferente):

```powershell
cd caminho\do\projeto\finstrategy
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,mt5]"
```

**Login automático (opcional, via `.env`)**: se o terminal MT5 já está aberto e logado
na conta/simulador certo, pule esta parte — não precisa de nada disso. Se quiser que o
script faça login sozinho (ex. numa tarefa agendada, sem abrir o MT5 na mão toda vez):

```powershell
copy .env.example .env
notepad .env
```

Preencha as linhas (sem aspas, sem espaço antes/depois do `=`):

```
MT5_LOGIN=12345678
MT5_PASSWORD=sua_senha_aqui
MT5_SERVER=NomeDaCorretora-Demo
```

Salve e feche. O `.env` já está no `.gitignore` — nunca vai pro git, fica só na sua
máquina. `MT5Provider()` lê esse arquivo sozinho (via `python-dotenv`, instalado junto
com `pip install -e ".[mt5]"`) toda vez que o script roda, sem precisar configurar
variável de ambiente manualmente no Windows.

**Rodar a coleta** (com o terminal MT5 aberto e logado no simulador/demo):

```powershell
python scripts\collect_mt5_daily.py --tickers AAPL,DUK,NEE

# dá pra rodar fora do horário de pregão (ex. à noite) — o MT5 serve o último
# fechamento/cotação em cache, não precisa do mercado aberto no momento da coleta

# backfill: popula o log de uma vez com 2 anos de histórico de preço (em vez de só
# o fechamento mais recente), uma linha por dia; use uma vez e depois volte a rodar
# sem --years-back pra ir só adicionando o dia mais recente
python scripts\collect_mt5_daily.py --tickers AAPL,DUK,NEE --years-back 2
```

**Agendar (Agendador de Tarefas do Windows, em vez de `cron`)**: crie uma tarefa que
rode diariamente, com "Programa/script" apontando para
`caminho\do\projeto\finstrategy\.venv\Scripts\python.exe` e "Argumentos" para
`scripts\collect_mt5_daily.py --tickers AAPL,DUK,NEE` — o terminal MT5 precisa estar
aberto e logado na hora em que a tarefa rodar.

Cada rodada **anexa** (nunca sobrescreve) uma linha por ticker em dois CSVs de log —
sempre o fechamento do dia, nunca um preço intradiário:

- `data_store/mt5/acoes_diario.csv`: ticker + fechamento (MT5) + indicadores
  fundamentalistas (Yahoo — o MT5 não tem fundamentalista). No backfill
  (`--years-back`), o fundamentalista só é anexado na linha mais recente — não existe
  fundamentalista histórico diário (seria repetir o retrato de hoje em todo o passado).
- `data_store/mt5/opcoes_diario.csv`: cada call/put líquida do vencimento mais
  próximo, com prêmio/strike/vencimento/preço do ativo-objeto na mesma linha. Sempre a
  cadeia **atual** — não existe cadeia de opções histórica no MT5/backfill.

Depois de cada rodada, os dois CSVs são reordenados por data (crescente) — dá para
rodar fora de ordem em dias diferentes que o resultado continua legível. Não
deduplica: rodar duas vezes no mesmo dia grava duas linhas para aquele dia.

**Antes de confiar nos dados de opção**, a forma de nomear símbolos de opção varia
muito entre corretoras no MT5. Use `MT5Provider(...).inspect_symbol("SEU_SIMBOLO")`
num console Python para conferir se os campos `option_right`/`option_strike`/
`expiration_time` vêm preenchidos no seu simulador antes de rodar a coleta de verdade
— ver a docstring de `src/finstrategy/data/mt5_provider.py` para detalhes e limitações
(sem fundamentalista, sem taxa livre de risco padrão).

### 5c. Histórico de ações + opções pela COTAHIST da B3

Quando o Yahoo não cobre o papel/opção (caso de vários nomes da B3 — ver a nota em
`config/tickers_setor_eletrico.yaml`) e o MT5 só mostra as opções que existem *agora*
(contrato vencido some do servidor), a saída é a **COTAHIST**: a série histórica oficial
da B3, um arquivo público e gratuito com **todos** os instrumentos negociados no ano —
ações, calls (070) e puts (080), inclusive séries já vencidas.

Roda em qualquer sistema (macOS/Linux/Windows), **sem** Yahoo e **sem** terminal MT5 —
é só download de arquivo. Baixa **um `.ZIP` por ano** e cacheia em
`data_store/cotahist/`, então o download acontece uma vez só por ano.

```bash
# anos 2023 a 2025, raízes padrão do setor energia (Taesa, CPFL, Engie, Cemig, Copel,
# Eletrobras, Energisa, Eneva, Equatorial, Neoenergia, Auren, CTEEP, Petrobras, PRIO)
python scripts/collect_cotahist.py --years 2023-2025

# anos avulsos e raízes customizadas (a raiz casa a ação E toda a cadeia de opções)
python scripts/collect_cotahist.py --years 2024,2025 --roots PETR,ELET,EGIE
```

Gera, em `data_store/cotahist/` (ou `--output-dir`), um CSV consolidado mais um por
classe (`energia_acao_*.csv`, `energia_call_*.csv`, `energia_put_*.csv`), com
OHLC/volume por pregão e, nas opções, `preexe` (strike) e `datven` (vencimento) na
mesma linha.

> **Nota sobre "randomizar IP/MAC para não ser derrubado pela B3":** não é feito, de
> propósito, porque é desnecessário. A COTAHIST é um arquivo estático cujo uso
> pretendido é ser baixado; com o cache em disco são pouquíssimas requisições na vida
> do projeto, não há rate-limit a contornar. (Randomizar MAC também não teria efeito:
> o endereço MAC não trafega além da rede local, o servidor da B3 nunca o vê.) O módulo
> usa um User-Agent honesto e timeout, como qualquer cliente HTTP bem-comportado — ver
> a docstring de `src/finstrategy/data/cotahist_provider.py`.

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
| `mt5_provider.py` | Implementação do `MarketDataProvider` via o pacote `MetaTrader5` (Windows + terminal/simulador aberto; `pip install -e ".[mt5]"`). Preço e cadeia de opções vêm do MT5; fundamentalista continua vindo do Yahoo (o MT5 não tem). Limitações e como conectar: ver docstring do arquivo. |
| `cotahist_provider.py` | Leitor da **COTAHIST** da B3 — série histórica oficial (arquivo público anual) com OHLC/volume de **todas** as ações e opções (calls 070, puts 080) negociadas no ano, inclusive séries já vencidas. Resolve o que Yahoo (sem opções B3) e MT5 (só séries vigentes) não cobrem. Baixa 1 `.ZIP`/ano e cacheia em `data_store/cotahist/`. `load_setor(roots, years)` filtra o setor e marca `classe` (ação/call/put). Usado por `scripts/collect_cotahist.py`. |
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
| `mt5_daily_log.py` | Coleta diária via MT5 em dois CSVs **log** (`data_store/mt5/`, append-only — cada rodada só adiciona linhas embaixo, nunca sobrescreve): `acoes_diario.csv` (fechamento do dia + fundamentalista) e `opcoes_diario.csv` (cada call/put líquida, com prêmio/strike/vencimento). Reordena os dois por data ao final de cada rodada (`csv_store.sort_csv_by`). Usado por `scripts/collect_mt5_daily.py`. |

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
| `collect_mt5_daily.py` | Wrapper de linha de comando para `collector/mt5_daily_log.py` — a coleta diária via MT5 (ver seção 5b acima). |
| `collect_cotahist.py` | Baixa a COTAHIST da B3 e extrai ações + opções do setor energia para CSV (ver seção 5c acima). Usa `data/cotahist_provider.py`. |
| `rank_us_megacaps.py` | Reavalia o top-20 EUA ao vivo no Yahoo; usado para gerar/atualizar `config/us_top20_megacaps.yaml`. |

### `tests/`

Um arquivo de teste por módulo de `src/finstrategy/` (mesma nomenclatura), quase todos
usando mocks/fakes para não depender de rede nem do mercado estar aberto — 188 testes
no total. Rodar com `pytest`. `test_mt5_provider.py` usa um dublê do pacote
`MetaTrader5` (não instala fora do Windows) — não substitui testar contra um terminal
MT5 de verdade, só garante que a lógica em volta dele está correta.
