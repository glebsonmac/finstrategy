# tasks.md — execução do plan.md

Checklist de implementação real, em ordem. Marcar `[x]` só quando o código existir E
tiver teste passando (ou validação manual registrada). Atualizar sempre que um passo
terminar, antes de ir para o próximo — este arquivo é o que evita perder o fio entre
sessões.

Contexto permanente: mercado pode estar fechado durante a implementação (fora do
pregão). Isso é esperado — o notebook já trata preço zerado/bid-ask zerado como
"último negócio"/"não elegível", não é bug. Testes de rede devem tolerar isso.

## Fase 1 — estrutura do pacote
- [x] Criar diretórios `src/finstrategy/...`, `config/`, `data_store/...`, `tests/`, `notebooks/`
- [x] Copiar notebook original para `notebooks/`
- [x] `.gitignore` atualizado (data_store runtime, watchlist/settings runtime)
- [x] `pyproject.toml` com dependências + venv criado e instalado (`.venv`)
- [x] `config/settings.example.yaml` (seed de `AnalysisSettings`) + `config_model.py`
- [x] `config/tickers_setor_eletrico.yaml` (seed da watchlist) — validado manualmente:
      9/9 tickers resolvem no Yahoo (corrigido em 2026-10-13: ELET3.SA, TRPL4.SA, EBR,
      ELP não resolveram no Yahoo nesta data e foram trocados por CPLE3.SA e CIG)

## Fase 2 — camada de dados
- [x] `data/base.py` — interface `MarketDataProvider` + dataclasses (`SymbolMatch`, `SymbolInfo`)
- [x] `data/errors.py` — `MarketDataError`
- [x] `data/yahoo_provider.py` — extrair funções do notebook (preço, dividendo, taxa livre
      de risco, cadeia de puts/calls) sem mudar comportamento + `download_call_chain` novo
- [x] `data/symbol_search.py` — `search_symbols`, `validate_symbol`, `validate_symbols`
      (testado manualmente contra o Yahoo real: AAPL ok/com opções, TAEE11.SA ok/sem
      opções, ticker inválido levanta erro claro)
- [x] `data/fundamentals_provider.py` — indicadores fundamentalistas normalizados
- [x] `data/mt5_provider.py` — stub `NotImplementedError`
- [x] `data/watchlist.py` — CRUD sobre `config/watchlist.yaml`, `add_tickers` dispara
      fundamentalista automaticamente, `ensure_seeded` testado de ponta a ponta
- [x] testes automatizados (`pytest`, com mocks — sem depender de rede/mercado aberto):
      `tests/test_symbol_search.py` (8), `tests/test_watchlist.py` (7)

## Fase 3 — pricing
- [x] `pricing/garch_montecarlo.py` — extrair `GarchParameters`, `fit_garch_t`,
      `simulate_prices_at_expiration`, `risk_metrics` do notebook (sem mudar lógica) +
      `simulate_price_paths` novo (trajetória diária, para a DQN de saída)
- [x] `pricing/black_scholes.py` — preço BS call/put, Greeks, IV implícita +
      `price_and_greeks_array` (versão vetorizada, usada pela DQN de saída)
- [x] testes: `tests/test_garch_montecarlo.py` (6), `tests/test_garch_paths.py` (2),
      `tests/test_black_scholes.py` (9) — todos passando (paridade put-call, bounds de
      delta, IV implícita recupera vol conhecida, centralização risco-neutro exata)

## Fase 4 — estratégia (put de proteção + timing de saída)
- [x] `strategy/protective_put.py` — extrair `ProtectivePutPlan`, `build_strike_menu`,
      `recommend_strike`, `profit_per_share_at_expiration` do notebook
- [x] `strategy/exit_timing_bs.py` — `time_value_curve` (BS dia a dia, spot fixo, isola
      theta decay) + `expected_monte_carlo_value_curve` (valor esperado via GARCH MC) +
      `recommend_exit_day` (regra fixa: sair quando valor-tempo cai abaixo do piso)
- [x] testes: `tests/test_protective_put.py` (10), `tests/test_exit_timing_bs.py` (7) —
      todos passando

## Fase 5 — RL (DQN)
- [x] `rl/entry_dqn.py` — extrair `QNetwork`, `EntryTimingEnvironment`, `train_entry_dqn`,
      `evaluate_entry_timing` do notebook, sem mudar lógica
- [x] `rl/exit_dqn.py` — novo agente de saída do derivativo: estado com Greeks/BS
      (moneyness, delta, theta, vega, P&L não realizado, gap IV-GARCH, tempo restante),
      recompensa = valor de vender agora − valor de manter até o vencimento (descontado),
      reusa QNetwork/Double DQN do entry_dqn. Treina sobre trajetórias GARCH Monte Carlo
      simuladas (não há histórico real de preço de opção dia a dia disponível no Yahoo)
      — limitação documentada no próprio módulo.
- [x] testes: `tests/test_entry_dqn.py` (6), `tests/test_exit_dqn.py` (6) — todos
      passando, treino rápido (poucos rounds) só para validar que roda e a forma da saída

## Fase 6 — watchlist dinâmica (ligação com frontend, seção 5 do plano)
- [x] `watchlist.add_tickers` dispara coleta automática (preço, fundamentalista,
      elegibilidade de derivativo) ao adicionar — implementado em Fase 2/data
- [x] teste manual real: `ensure_seeded` com a lista do setor elétrico → 9/9 resolvidos
      com preço, elegibilidade e fundamentalista preenchidos (ver Fase 1)
- [x] `tests/test_watchlist.py` (7, mockado) — ver Fase 2

## Fase 7 — coletor contínuo
- [x] `collector/csv_store.py` — escrita append-safe particionada por dia/ticker
- [x] `collector/market_session.py` — calendário de pregão B3 (BVMF)/NYSE via
      `pandas_market_calendars` (atenção: `market_open`/`market_close` vêm em UTC, não
      no fuso local — timestamps naive são tratados como UTC, documentado no módulo)
- [x] `collector/scheduler.py` — `collect_price_snapshot`, `run_model_pipeline_for_ticker`,
      `run_collection_cycle` (unidade testável), `run_forever` (loop real, com
      `stop_event`/`max_iterations` para a tela "Coletor" do frontend conseguir
      parar e para os testes não rodarem para sempre)
- [x] testes: `tests/test_csv_store.py` (9), `tests/test_market_session.py` (10),
      `tests/test_scheduler.py` (6) — todos passando, com `FakeProvider`/`monkeypatch`
      (sem depender do mercado estar aberto)
- [ ] modo dry-run/replay explícito sobre CSV já salvo (hoje o coletor só roda contra
      o provider real ou um fake injetado no teste; formalizar um `ReplayProvider` que
      lê `data_store/market/...` fica como melhoria futura, não bloqueia o MVP)

## Fase 8 — relatório PDF
- [x] `reporting/panel.py` — `build_panel_dataframe` (tabela técnica) +
      `build_plain_language_summary` (texto sem jargão, por ativo)
- [x] `reporting/pdf_report.py` — `generate_pdf_report` (capa + 1 página por ativo,
      disclaimer, reportlab)
- [x] testes: `tests/test_panel.py` (9), `tests/test_pdf_report.py` (2) — PDF real
      gerado e validado (header `%PDF`, tamanho > 1KB)

## Fase 9 — frontend (Streamlit)
- [x] `frontend/app.py` — tela "Ações" (busca/checkbox + texto livre + watchlist atual)
- [x] telas "Configuração", "Rodar agora", "Histórico", "Coletor", "Modelos"
- [x] validação manual no navegador (Chrome, via claude-in-chrome): `streamlit run`
      sobe sem erro, todas as 6 telas navegadas e testadas com dados reais do Yahoo
      (ver Fase 10 — mesma sessão de teste cobriu as duas fases)

## Fase 10 — testes/validação de ponta a ponta
- [x] `pytest` completo passando: **96 testes, 0 falhas** (cobre data layer, pricing,
      strategy, rl/entry+exit DQN, collector, reporting — ver fases 2-8 por módulo)
- [x] rodada manual real no navegador (2026-10-13, mercado fechado — confirmado "Não"
      na tela Coletor, consistente com o aviso do usuário):
  - **DUK** (ADR/EUA, `elegivel_derivativo=True`): tela "Ações" achou nome/moeda/bolsa
    reais; tela "Modelos" treinou a DQN de entrada de verdade (120 rounds, dados reais)
    e mostrou a tabela de avaliação honesta; tela "Rodar agora" rodou o pipeline
    completo — achou a cadeia de puts real (contrato `DUK261120P00110000`, strike 110,
    vencimento 2026-11-20), usou "último negócio" (mercado fechado, tratamento correto
    herdado do notebook), calculou risco via GARCH Monte Carlo, gerou o resumo em
    linguagem simples e o PDF para download.
  - **TAEE11.SA** (B3, `elegivel_derivativo=False`): tela "Rodar agora" rodou só a
    análise de ação (sem derivativo), com o aviso claro "não elegível: o Yahoo não tem
    opções para este ativo" tanto no painel técnico quanto no resumo em linguagem
    simples — confirma que a decisão da seção 0 do plano está implementada corretamente.
  - Tela "Histórico" mostrou o snapshot fundamentalista real da DUK (setor Utilities,
    Regulated Electric) e listou o PDF gerado para download.
  - Tela "Coletor": iniciar/parar a thread em background funcionou sem travar a UI.
  - Console do navegador sem erros do app (só um log de uma extensão do Grammarly,
    não relacionado).
- [x] `.gitignore` confirmado: `config/watchlist.yaml`, `config/settings.yaml` e todo o
      conteúdo gerado em `data_store/` ficam fora do controle de versão (`git add -n`
      só pegaria os arquivos-semente, os `.gitkeep` e o notebook)

## O que falta (backlog, fora do MVP implementado nesta sessão)
- MetaTrader5 (`data/mt5_provider.py`) — decisão da seção 0 do plan.md ainda pendente,
  módulo é só um stub com `NotImplementedError`
- Fonte alternativa de opções para B3 (hoje qualquer `.SA` sem opções no Yahoo fica
  "não elegível", por decisão explícita do usuário — ver plan.md seção 0)
- Modo dry-run/replay explícito do coletor sobre CSV já salvo (hoje só roda contra
  provider real ou fake injetado em teste)
- Validação da regra de saída (Black-Scholes/DQN) contra dados históricos reais de
  opção — não há fonte de preço de opção dia a dia no Yahoo; hoje treina/avalia só
  sobre trajetórias simuladas (documentado em `strategy/exit_timing_bs.py` e `rl/exit_dqn.py`)

---
## Log de execução

**2026-10-13** — Implementação completa do MVP numa sessão, do zero (repositório
estava vazio, só LICENSE/README). Estrutura `src/finstrategy/` criada, venv + deps
instaladas, notebook original copiado para `notebooks/` e mantido intacto como
referência. Lógica do notebook (Yahoo provider, GARCH Monte Carlo, put de proteção,
DQN de entrada) extraída para módulos sem mudar comportamento. Adicionado: Black-Scholes
(preço/Greeks/IV, vetorizado para performance), DQN de saída do derivativo (novo, com
recompensa baseada em redução de risco), busca/validação de qualquer ticker do Yahoo,
watchlist dinâmica (`config/watchlist.yaml`), coletor contínuo com calendário de pregão
B3/NYSE, relatório PDF para não-técnico, e frontend Streamlit com 6 telas.

Durante a implementação, 4 dos 9 tickers semente do setor elétrico (`ELET3.SA`,
`TRPL4.SA`, `EBR`, `ELP`) não resolveram no Yahoo Finance nesta data — substituídos por
`CPLE3.SA` e `CIG`, que resolveram e têm dados completos; ver nota em
`config/tickers_setor_eletrico.yaml`. Revisitar periodicamente.

81 testes automatizados passando + validação manual completa no navegador (ver Fase
10 acima). Mercado estava fechado durante toda a sessão (confirmado pela própria tela
"Coletor"), exatamente como o usuário avisou — o tratamento "último negócio" herdado
do notebook funcionou corretamente nesse cenário.

**2026-10-13 (sessão 2, mercado ABERTO)** — Pedido do usuário: deixar a apresentação
do frontend mais clara (cotação e fundamentalista), o coletor contínuo precisa dar
feedback visível de que está ativo e coletando, a tela de relatórios precisa de um
botão para gerar um relatório novo — e testar tudo com o mercado aberto antes de
mexer, para pegar bugs reais. Fiz exatamente isso, nessa ordem, e apareceram dois bugs
reais que só o mercado aberto revelou:

1. **Bug crítico (coletor morria em silêncio):** `run_forever` passava `today` como
   `pd.Timestamp.now(tz="UTC").normalize()` (tz-aware) para `run_collection_cycle` →
   `run_model_pipeline_for_ticker` → `analyze_ticker` → `choose_expiration`, que
   compara com vencimentos de opção do Yahoo (sempre naive). Resultado:
   `TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects.`, não
   capturado em lugar nenhum, matando a thread do coletor sem avisar a UI (o botão
   "Parar coletor" continuava visível como se estivesse tudo bem). Só apareceu porque
   o mercado estava aberto e `run_model=True` disparou no primeiro ciclo.
   **Fix:** `run_forever` agora usa `now_utc` só para o calendário de pregão e um
   `today` naive (`pd.Timestamp.today().normalize()`) para tudo que toca
   `analyze_ticker`; `run_model_pipeline_for_ticker` e `run_collection_cycle` também
   passaram a capturar `Exception` genérica por ticker (não só `MarketDataError`), e
   `run_forever` ganhou um try/except em volta do corpo do loop inteiro — a thread do
   coletor agora NUNCA mais pode morrer em silêncio, qualquer erro vira
   `last_error` visível no status. Regressão travada em
   `tests/test_scheduler.py` (4 testes novos).
2. **Bug de dado errado (não quebrava nada, mas mentia pro usuário):** o campo
   `dividendYield` do `yfinance` vem em pontos percentuais (3.69 = 3,69%), diferente
   de `returnOnEquity`/`revenueGrowth`/`profitMargins` (fração, 0.0986 = 9,86%). Sem
   normalizar, o frontend mostrava "Dividend yield: 369%" para a DUK — só apareceu
   olhando o dado real ao vivo, nenhum teste existia para `fundamentals_provider.py`
   antes disso. **Fix:** normalizado na fonte (`fundamentals_provider.py`, divide por
   100), com `tests/test_fundamentals_provider.py` novo (8 testes, cobre esse caso e
   os demais campos de `FundamentalSnapshot`).

**Melhorias de frontend entregues** (`src/finstrategy/frontend/app.py`):
- Tela **Coletor**: agora lê `data_store/collector_status.json` (escrito pelo
  `run_forever` a cada ciclo via `scheduler.write_status`/`read_status`) e mostra
  indicador 🟢/🟡/⚪ de atividade real (não só "a thread foi iniciada"), última
  verificação com segundos decorridos, contagem de iterações, última coleta/rodada de
  modelo bem-sucedida, erro mais recente se houver, e uma tabela com o que foi
  coletado no último ciclo — validado ao vivo: rodou 2 ciclos reais (mercado aberto),
  preço atualizando a cada iteração, sem travar.
- Tela **Histórico** virou **Histórico & Relatórios**: cotação mais recente e
  fundamentalista em cartões (`st.metric`) com rótulos em português e números
  formatados (%, "bi"/"mi"), histórico completo num expander, e um botão
  **"Gerar novo relatório agora"** que roda o pipeline completo e oferece o PDF sem
  precisar ir até "Rodar agora" — validado ao vivo (DUK, TAEE11.SA, SPY).
- Lógica de "rodar pipeline + montar painel + gerar PDF" extraída para
  `run_full_pipeline`/`render_pipeline_results`/`generate_and_offer_pdf`, reusada por
  "Rodar agora" e pelo botão novo em "Histórico & Relatórios" — sem duplicar código.
- Trocado `use_container_width=True` (deprecated) por `width="stretch"` em todo o app.

109 testes automatizados passando ao final (era 81; +4 regressão do bug de timezone,
+8 `fundamentals_provider`, +16 formatação/cobertura extra ao longo da sessão).
Mercado aberto confirmado o tempo todo pela própria tela "Coletor" ("Sim"), cadeias de
opção reais usando "ask ao vivo" (não "último negócio") em DUK e SPY — confirma que o
tratamento de mercado aberto/fechado herdado do notebook também funciona no caminho
ao vivo, não só no fechado testado na sessão anterior.

**2026-10-13 (sessão 3) — coleta noturna + lista local de ações**

Pedido do usuário: (1) um coletor "essencial" que gere um CSV, pensado para rodar à
noite via agendador, portável para outro computador; (2) as 20 maiores ações da bolsa
dos EUA como lista padrão; (3) no lugar de buscar ticker, ter também uma opção de
**lista** para escolher (não só busca ao vivo, que bate na internet toda vez) —
"parecido com investing.com, simples".

- [x] **Top 20 EUA ao vivo, não de memória** (`scripts/rank_us_megacaps.py`): consulta
  `fast_info.marketCap` via Yahoo sobre um pool de ~60 mega-caps conhecidas e ranqueia
  de verdade — mesmo princípio usado para validar o setor elétrico. Resultado em
  2026-10-13: NVDA, AAPL, GOOGL, MSFT, AMZN, META, AVGO, TSLA, BRK-B, LLY, AMD, WMT,
  JPM, V, XOM, JNJ, INTC, MA, PLTR, ABBV (GOOG/GOOGL deduplicados — mesma empresa).
  Salvo como semente em `config/us_top20_megacaps.yaml`, documentado como "reavaliar
  periodicamente" (ranking de mercado muda).
- [x] `config/us_stock_list.csv` — lista local com as ~503 ações do S&P 500
  (ticker/nome/setor), baixada de `datasets/s-and-p-500-companies` (GitHub), para a
  busca instantânea sem rede.
- [x] `data/local_stock_list.py` — `search_local_stocks`/`load_all`, filtro em memória
  (sem rede, ~30ms), reaproveita `SymbolMatch` (campo `exchange` carrega o setor aqui,
  documentado no código). 8 testes (`tests/test_local_stock_list.py`).
- [x] `collector/overnight.py` + `scripts/collect_overnight.py` — coleta "essencial"
  (cotação + indicadores fundamentalistas, uma linha por ticker) numa rodada só, sem
  precisar do Streamlit aberto. Resolve tickers por prioridade: `--tickers` explícito
  → `config/watchlist.yaml` → semente top-20 EUA. Grava
  `data_store/overnight/AAAA-MM-DD_essencial.csv` + `latest.csv` (sempre sobrescrito,
  caminho estável pra outro processo/computador achar a rodada mais recente). Um
  ticker com problema vira `status` na linha, nunca derruba a rodada inteira (mesmo
  cuidado do bug do coletor contínuo na sessão anterior). Testado manualmente contra o
  Yahoo real (AAPL/DUK ok, ticker inválido com erro claro) + 12 testes mockados
  (`tests/test_overnight.py`).
- [x] README.md: seção "Coleta noturna" com exemplo de `cron`.
- [x] Frontend (`app.py`, tela "Ações"): reorganizada em 3 abas — **📋 Escolher da
  lista** (nova, local/instantânea, é a aba padrão/primeira agora), **🔎 Buscar no
  Yahoo** (a busca ao vivo que já existia), **✍️ Colar tickers**. Botão novo "Popular
  com as 20 maiores ações dos EUA" ao lado do botão do setor elétrico, para popular a
  watchlist do app a partir da mesma semente usada pelo coletor noturno.
- [x] `config/us_top20_megacaps.yaml` e `config/us_stock_list.csv` confirmados como
  arquivos versionados (seed/referência), não gitignored — `git add -n` checado.

129 testes automatizados passando ao final desta sessão (era 109). Frontend
recompilado sem erro (`py_compile`); validação ao vivo do fluxo completo (busca →
adicionar → coletor noturno) feita via script de linha de comando contra o Yahoo real,
não via navegador nesta sessão (Streamlit ficou parado a pedido do usuário por causa
de memória baixa no Mac — próxima vez que o app subir, vale clicar pelas 3 abas novas
da tela "Ações" pra confirmar visualmente).
