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
- [x] `data/mt5_provider.py` — stub `NotImplementedError` no MVP inicial; implementado
      de verdade na sessão de 2026-10-19 (ver log de execução no fim deste arquivo)
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

## Fase 11 — frontend auto-explicativo e guiado (plan.md, seção 10)

**Implementado e testado em 2026-10-13 (sessão 5)**, com o mercado já fechado — testado
contra dados reais baixados na sessão da tarde (DUK/SPY com derivativo, TAEE11.SA sem)
mais chamadas ao vivo ao Yahoo (histórico, fundamentalista e opções continuam
respondendo fora do pregão). Pesquisa web que fundamentou as decisões de UX: Lollypop
"Investment Dashboard UX Design" (2026), Azure Responsible AI Dashboard e MLOps
(Comet/W&B) pro conceito de "model card" — ver plan.md seção 10 pros links.

### 11.1 — Gráficos do notebook no Streamlit
- [x] `reporting/charts.py` novo módulo: `plot_strike_menu`, `plot_protection_payoff`,
      `plot_entry_evaluation` portados do notebook (mesma lógica/cores, devolvem
      `Figure` em vez de `plt.show()`) + `plot_exit_evaluation` e
      `plot_time_value_curve` novos (não existiam nem no notebook)
- [x] Matplotlib (API orientada a objeto, sem `pyplot` global — seguro pra Streamlit
      re-renderizar sem vazar estado entre reruns)
- [x] Wirado em `render_pipeline_results` (tela "Histórico & Relatórios"), com legenda
      em português explicando cada gráfico — testado ao vivo com DUK: os 3 gráficos
      de proteção + o forest plot da DQN renderizaram corretamente com dados reais
- [x] 6 testes (`tests/test_charts.py`)

### 11.2 — Consolidar "Rodar agora" dentro de "Histórico & Relatórios"
- [x] Página "Rodar agora" removida do `st.sidebar.radio` e do código
- [x] `render_pipeline_results`/`generate_and_offer_pdf` continuam compartilhados,
      agora só chamados por "Histórico & Relatórios" — testado ao vivo, 5 telas

### 11.3 — Tela "Modelos": métricas, explicação, persistência em pickle
- [x] `reporting/model_store.py` novo: `save_model`/`load_model`/`list_models`,
      pickle do payload (`QNetwork` + hiperparâmetros + avaliação completa) em
      `data_store/models/<ticker>_<tipo>_<timestamp>.pkl` + `index.csv`
- [x] Tela "Modelos" reformulada com abas "📂 Modelos salvos" / "🧠 Treinar novo":
      carregar sem retreinar (testado: Q-values idênticos antes/depois do pickle) ou
      treinar novo (entrada sempre disponível; saída só se `elegivel_derivativo`)
- [x] `render_model_card`: parágrafo simples + métricas em destaque + expander
      "Detalhes técnicos" (tabela completa, p-valores, hiperparâmetros) — testado ao
      vivo pra DQN de entrada (TAEE11.SA) e de saída (DUK)
- [x] 6 testes (`tests/test_model_store.py`)
- [x] **2 bugs reais encontrados testando ao vivo, corrigidos:**
  1. CSV round-trip: `pd.read_csv` por padrão lê campo vazio (`sinal_hoje=""` dos
     modelos de saída) de volta como `NaN`, não `""` — `list_models` agora usa
     `keep_default_na=False`. Pego por teste automatizado antes mesmo de abrir o navegador.
  2. **`st.tabs` executa o conteúdo das DUAS abas na mesma rodada de script, em ordem
     de código** — como "Modelos salvos" vinha antes de "Treinar novo", ela lia o
     índice de modelos *antes* do `save_model` da outra aba rodar nessa mesma
     execução, então sempre mostrava o modelo recém-treinado como "nenhum salvo
     ainda" até uma interação qualquer disparar outra rodada. Só apareceu testando no
     navegador de verdade — nenhum teste unitário pega esse tipo de bug de ordenação
     de execução do Streamlit. Corrigido guardando o resultado do treino em
     `st.session_state["last_trained_model"]` + `st.rerun()` logo após salvar, pra
     "Modelos salvos" já nascer atualizada na rodada seguinte.

### 11.4 — Nova seção "Dados brutos" (dentro de "Histórico & Relatórios")
- [x] Botão "🔍 Carregar dados brutos" por ticker: histórico de preço completo
      (`st.line_chart` + tabela + CSV) — testado com DUK, 3716 pontos reais desde 2012
- [x] Cadeia de opções líquidas da última consulta (`analyze_ticker(...)["liquid_puts"]`)
      visível + exportável, só pra tickers elegíveis — testado com DUK (2 puts reais,
      "ask ao vivo")
- [x] Expander de fundamentalista já existente rotulado explicitamente como "dado
      bruto, como veio do Yahoo"
- [x] Decisão: ficou dentro de "Histórico & Relatórios" (não virou tela própria) —
      menos fragmentação, consistente com a consolidação da seção 11.2

### 11.5 — Fluxo guiado de hedge ("wizard")
- [x] Decisão tomada na implementação: **não criou tela nova** — o fluxo guiado é a
      própria sequência de "Histórico & Relatórios" → botão "Gerar novo relatório
      agora", que agora mostra, em ordem: painel técnico (com tooltip/glossário nas
      colunas técnicas) → resumo em linguagem simples → gráficos com legenda →
      download do PDF. Isso evita re-fragmentar em mais uma tela depois de consolidar
      a 11.2.
- [x] Tooltips de glossário via `st.column_config.Column(help=...)` no painel técnico
      (`RISK_METRIC_HELP`) — "perda máxima", "ES 97,5%...", "P(perda)...", "entrada
      (DQN)" — cada um com frase em português do que significa, aparece ao passar o
      mouse no cabeçalho da coluna
- [x] Cada gráfico tem uma legenda de 1-2 frases abaixo explicando como ler

## Fase 12 — Black-Scholes na análise de risco do relatório principal (plan.md, seção 11)

**Implementado e testado em 2026-10-14**, com dados reais no navegador (DUK, com
derivativo, e TAEE11.SA, sem — confirmando os dois caminhos). Toda a lógica combinada
ficou num módulo novo, `strategy/risk_analysis.py` (`build_risk_analysis`), chamado
pela tela "Histórico & Relatórios" via `render_risk_analysis_section` em `app.py`.

### 12.1 — Preço da opção explicado por Black-Scholes
- [x] `build_risk_analysis` calcula `black_scholes.price_and_greeks(...)` pra strike
      recomendada, usando só dados já presentes em `analyze_ticker` (sem rede nova)
- [x] Card "Prêmio pago vs. teórico (Black-Scholes)" com a diferença em % — testado ao
      vivo com DUK: preço teórico 1.7387 vs. prêmio pago 1.80 = **+3.5%** (conferido
      manualmente, bate com o JSON de Greeks mostrado no expander técnico)
- [x] Delta, theta/dia e vega da strike recomendada no expander "Detalhes técnicos",
      cada um com 1 frase em português simples

### 12.2 — Dias até considerar vender a proteção
- [x] `build_risk_analysis` chama `exit_timing_bs.time_value_curve` +
      `recommend_exit_day` pra todo ticker elegível, direto no relatório principal
- [x] Card "Considere vender até" com data concreta (ex.: 11/11/2026) — testado ao vivo
- [x] `charts.plot_time_value_curve` entra no relatório principal, com a linha do dia
      recomendado marcada — testado ao vivo (DUK: "sair no dia 20")
- [ ] Opcional (não implementado, não bloqueava o resto): mostrar o que uma DQN de
      saída já salva diria como segunda opinião — fica pro backlog

### 12.3 — Previsão de tendência (ação e opção)
- [x] P(preço sobe/desce), preço médio esperado, faixa de 90% extraídos de
      `result["simulated_prices"]` e mostrados como métricas — testado ao vivo (DUK:
      P(acima)=50%/P(abaixo)=50%, preço médio esperado 118.24, faixa 109.26–127.45)
- [x] Mesma leitura pro valor da put no vencimento (`max(strike - simulated_prices, 0)`,
      mesma distribuição já calculada — não precisou de `price_and_greeks_array`
      porque no vencimento o valor da put É o payoff intrínseco, sem valor-tempo;
      decisão tomada na implementação, documentada no docstring do módulo)
- [x] Ressalva de que é simulação, não garantia — presente no texto da UI

### 12.4 — "Confiança" do cálculo, definida com rigor (sem número solto inventado)
- [x] Confiança estatística do sinal da DQN — card separado "Confiança do sinal de
      entrada" com p-valor no tooltip, só quando há `entry_evaluation` disponível
- [x] Precisão da simulação Monte Carlo — reaproveitado `bootstrap_mean_interval` de
      `rl/entry_dqn.py` direto (já era genérico o suficiente, não precisou mover/duplicar
      — decisão tomada na implementação), aplicado sobre `simulated_prices`; mostrado
      como "largura do IC em % da média" com aviso explícito de que NÃO é confiança
      direcional — testado ao vivo (DUK: IC 118.17–118.32, 0.1% da média)
- [x] Risco de entrada "naquele momento" (perda máxima, P(perda), ES) continuam no
      card "Perda máxima com proteção" e no painel técnico já existente

### 12.5 — Nova seção "📊 Análise de Risco" no relatório (reorganização visual)
- [x] `render_risk_analysis_section` em `app.py`: 4 cards de destaque → tendência
      (ação e put) → gráfico de valor-tempo → expander "Detalhes técnicos" (Greeks,
      precisão do Monte Carlo) — divulgação progressiva, antes dos gráficos técnicos
      que já existiam (menu de strikes, payoff, forest plot)
- [x] Tooltips (`help=`) em cada `st.metric`, direto na função — não precisou estender
      `RISK_METRIC_HELP` (que é só pras colunas do painel técnico/DataFrame; os cards
      novos usam `st.metric(help=...)`, mais direto pro caso)
- [x] 9 testes (`tests/test_risk_analysis.py`), dados fake — cobre todas as seções,
      incluindo determinismo com seed fixo e casos de borda (put bem OTM, prêmio muito
      acima/abaixo do teórico)
- [x] Try/except ao redor da chamada em `app.py` — um erro na análise de risco mostra
      aviso e não derruba o resto do relatório (mesmo padrão defensivo do resto do app)
- [x] Testado ao vivo pros dois caminhos: DUK (com derivativo, seção completa aparece)
      e TAEE11.SA (sem derivativo, seção pulada sem erro, vai direto pro forest plot
      da DQN de entrada) — nenhum erro no log do servidor durante os testes

## O que falta (backlog, fora do MVP implementado nesta sessão)
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

**2026-10-13 (sessão 4) — planejamento da Fase 11 (frontend auto-explicativo)**

Pedido do usuário: só planejar (sem código) uma reformulação do frontend —
gráficos do notebook faltando no Streamlit, juntar "Rodar agora" com "Histórico &
Relatórios" (redundantes), tela de Modelos com mais explicação/métricas/persistência
em pickle, nova seção de dados brutos/fundamentalistas, e o sistema como um todo
precisa conduzir qualquer pessoa (leiga ou técnica) a fazer um hedge com métricas de
risco. Pedido explícito de pesquisa web para fundamentar antes de planejar.

Pesquisa feita (3 buscas): UX de dashboard de investimento/risco pra público misto
(leigo + técnico) — divulgação progressiva, métrica sempre com explicação em 1 frase,
benchmarking contextual, glossário como tooltip (fonte: Lollypop, "Investment
Dashboard UX Design", maio 2026); "model card" e explicação global/local de modelo
como padrão de MLOps/explicabilidade (Azure Responsible AI Dashboard, Comet/W&B,
InterpretML). Essas referências fundamentam a seção 10 do plan.md (ver lá os links).

Resultado: `plan.md` ganhou a seção 10 (5 subseções: gráficos, consolidar telas,
modelos com pickle, dados brutos, wizard de hedge guiado) com justificativa de cada
decisão de UX amarrada à pesquisa. `tasks.md` ganhou a Fase 11 (checklist espelhando
a seção 10, tudo `[ ]` — nada implementado ainda). Nenhum arquivo de código tocado
nesta sessão, só `plan.md` e `tasks.md`, como pedido.

**2026-10-13 (sessão 5) — implementação completa da Fase 11, mercado fechado**

Pedido do usuário: implementar tudo que foi planejado na Fase 11, usando os dados já
baixados na sessão da tarde pra testar (mercado fechado nessa hora). Implementado na
ordem sugerida pelo plano (11.2 → 11.1 → 11.4 → 11.3 → 11.5) e testado de duas formas:

1. **Script de integração** (`/tmp`, descartado ao final) chamando as mesmas funções
   que o `app.py` usa — `analyze_ticker`, `evaluate_entry_timing`,
   `evaluate_exit_timing`, os 5 gráficos, `model_store.save_model`/`load_model` — contra
   o Yahoo real pra DUK (confirmou: Yahoo responde histórico/fundamentalista/opções
   mesmo com o mercado fechado, só o bid/ask de opção fica "último negócio" às vezes
   em vez de "ask ao vivo"; nesta sessão em particular veio "ask ao vivo" mesmo com o
   pregão fechado, o que é um comportamento conhecido e documentado do Yahoo, não bug).
2. **Navegador real** (Chrome via claude-in-chrome) clicando em cada tela nova:
   watchlist com 3 abas, "Histórico & Relatórios" com dados brutos + gráficos +
   relatório, "Modelos" treinando e carregando modelos de entrada (TAEE11.SA) e saída
   (DUK), "Coletor" confirmando mercado fechado.

Dois bugs reais encontrados e corrigidos (detalhes na Fase 11 acima):
1. `pd.read_csv` lendo campo vazio como `NaN` em vez de string vazia (pego por teste
   automatizado, antes do navegador).
2. **`st.tabs` roda o conteúdo das duas abas na mesma execução, em ordem de código** —
   bug sutil de ordenação que só apareceu testando no navegador de verdade; nenhum
   teste unitário (que não simula o Streamlit rodando) teria pego isso. Reforça por
   que "implemente e teste" com o app real rodando foi importante aqui, não só rodar
   a suíte de testes.

141 testes automatizados passando ao final (mesmo número de antes — a Fase 11 é
majoritariamente frontend, mas ganhou `charts.py` e `model_store.py` com cobertura
própria: 12 testes novos, compensados por não ter mexido em mais nada). Frontend
validado visualmente de ponta a ponta com dados reais; nenhum erro no log do servidor
Streamlit durante toda a sessão de testes.

**2026-10-14 (sessão 6) — planejamento da Fase 12 (Black-Scholes na análise de risco)**

Pedido do usuário: ele notou que faltava a análise de risco "de verdade" no relatório
— confiança do cálculo, valor esperado, risco de entrada, previsão de tendência do
preço da ação e da opção, percentual de perda máxima com proteção — e que o
Black-Scholes precisa entrar na conta pra explicar o preço da opção, o valor de
entrada da put e em quantos dias considerar vender. Pedido explícito, de novo, de só
planejar e colocar nas tasks ("verifique como implementar isso e planeje e coloque nas
tasks também") — sem mexer em código nesta sessão.

Investigação antes de planejar: o gráfico de payoff "ação sem proteção vs. com
proteção" que o usuário lembrou do notebook **já existe** no sistema desde a Fase 11
(`charts.plot_protection_payoff`, visível no relatório principal) — não era isso que
faltava. O que realmente falta, confirmado via `grep` no código: `black_scholes.py` e
`exit_timing_bs.py` (ambos implementados e testados desde as Fases 3/4) **nunca são
chamados fora da tela "Modelos" > "Treinar novo" > "Saída"** — ou seja, o preço
teórico da opção, os Greeks, e a recomendação de "quantos dias até vender" nunca
aparecem pra quem só usa "Histórico & Relatórios" (a tela que a maioria das pessoas
realmente usa pra decidir).

Resultado: `plan.md` ganhou a seção 11 (5 subseções: preço explicado por
Black-Scholes, dias até vender, previsão de tendência, definição rigorosa de
"confiança" — com cuidado explícito pra não inventar um número de confiança
matematicamente enganoso — e a reorganização visual numa seção "📊 Análise de Risco").
`tasks.md` ganhou a Fase 12 espelhando isso, tudo `[ ]`. Nenhum código tocado nesta
sessão, só `plan.md` e `tasks.md`.

**2026-10-14 (sessão 7) — implementação da Fase 12, mesmo dia do planejamento**

Pedido do usuário: "pode implementar e testar" — seguiu direto da sessão de
planejamento pra execução. Implementado na ordem sugerida pelo plano
(12.1 → 12.2 → 12.4 → 12.3 → 12.5, calculados juntos num módulo só) e testado de duas
formas: `pytest` com dados fake (`tests/test_risk_analysis.py`, 9 testes cobrindo
determinismo, casos de borda e cada seção) e navegador real (Chrome via
claude-in-chrome) gerando um relatório com DUK (com derivativo) + TAEE11.SA (sem),
conferindo manualmente que os números batem entre si (preço teórico Black-Scholes do
card de destaque = o mesmo número no JSON de Greeks do expander técnico).

Nenhum bug encontrado desta vez (diferente das sessões 5 e 6/7 anteriores, que tinham
pego bugs reais) — a reutilização cuidadosa do que já existia (`exit_timing_bs.py`,
`black_scholes.py`, `bootstrap_mean_interval`, `charts.plot_time_value_curve`, todos
já testados em sessões anteriores) e o fato de `build_risk_analysis` não fazer
nenhuma chamada de rede nova (só recombina dados que `analyze_ticker` já tinha
calculado) reduziu bastante a superfície de coisas que podiam dar errado.

150 testes automatizados passando ao final (eram 141 antes desta fase, +9 novos de
`test_risk_analysis.py`). Frontend validado visualmente de ponta a ponta com dados
reais; nenhum erro no log do servidor Streamlit.

**2026-10-14 (sessão 8) — relatório do Streamlit/PDF igualado ao do notebook**

Pedido do usuário: o relatório do Streamlit estava "incompleto" comparado ao notebook
original. Comparação célula a célula confirmou 3 blocos de conteúdo que o notebook
mostra e que nunca foram extraídos pra `src/` (só os gráficos derivados deles
existiam, não os dados/texto por trás):
1. Tabela completa de strikes por ativo (notebook, seção 8) — só existia como gráfico.
2. Tabela de avaliação da DQN de entrada (4 estratégias + IC 95%) e o parágrafo de
   veredito (p-valores, dia médio de compra) — só apareciam na tela "Modelos", não no
   relatório principal.
3. Seção "Limitações" (6 avisos) + referências acadêmicas (notebook, seção 11) — no
   app/PDF virou uma frase genérica de disclaimer.

Corrigido (itens 1 e 2 — item 3 revertido, ver abaixo): `reporting/panel.py` ganhou
`MENU_COLUMNS`/`MENU_FORMATS`/`ENTRY_SUMMARY_FORMATS` (extraídos literalmente do
notebook) + `build_strike_menu_header`/`build_strike_menu_table`/
`build_entry_evaluation_table`/`build_entry_evaluation_verdict`. Wirado em
`frontend/app.py` (`render_pipeline_results`, dentro do expander por ticker) e em
`reporting/pdf_report.py` (`generate_pdf_report` ganhou os parâmetros opcionais
`results`/`entry_evaluations`; cada página de ativo agora tem a tabela de strikes —
dividida em 2 tabelas de 8 colunas pra caber na largura A4 — e a avaliação da DQN).

Validado: PDF real gerado com dados sintéticos (menu de 5 strikes, avaliação de DQN
com 4 estratégias) e lido de volta — tabelas e texto legíveis, sem estouro de página.
Um bug cosmético encontrado e corrigido: o marcador "►" da strike recomendada
(herdado do notebook) não existe na fonte Helvetica base do reportlab e virava um
glifo errado no PDF — trocado por ">>" só no caminho do PDF (Streamlit continua
usando "►", que renderiza bem no navegador).

**Correção no mesmo dia (feedback do usuário após o primeiro teste):** 3 ajustes.
1. **Limitações/referências removidas do Streamlit e do PDF** — o usuário esclareceu
   que essa seção é só do notebook, não devia ter entrado no relatório do app/PDF.
   Revertido: `LIMITATION_ITEMS`/`REFERENCE_ITEMS` tirados de `panel.py`, a seção
   "Limitações"/"Referências" tirada de `render_pipeline_results` (app.py) e
   `_limitations_and_references_flowables` removida de `pdf_report.py`.
2. **PDF não baixava com dados reais.** Causa provável: `reportlab.Paragraph`
   interpreta o texto como XML — qualquer `&`/`<`/`>` vindo de dado real (nome de
   ativo, texto de erro) quebra a geração inteira, e o erro não aparecia pro usuário
   porque a chamada de PDF não tinha try/except (a tela de resultado renderizava
   normal, só o botão de download sumia, sem explicação). Corrigido: `_escape_xml`
   novo em `pdf_report.py`, aplicado em todo texto dinâmico que vira `Paragraph`
   (cabeçalhos/células das tabelas novas, headline/corpo do resumo em linguagem
   simples); `generate_and_offer_pdf` (app.py) ganhou try/except com `st.error`
   visível em vez de falhar em silêncio. Validado gerando um PDF com ticker/símbolo
   contendo `&`/`<`/`>` de propósito — texto saiu correto (`&` literal, sem
   `&amp;` vazando) via extração de texto do PDF gerado (`pypdf`).
3. **Relatórios do mesmo dia se sobrescreviam.** `reports_csv_path` (usado antes pra
   nomear o PDF via `.with_suffix(".pdf")`) particiona só por dia (`AAAA-MM-DD.pdf`),
   então gerar dois relatórios no mesmo dia sobrescrevia o primeiro sem aviso.
   `collector/csv_store.py` ganhou `report_pdf_path(data_store_root, generated_at)`,
   particionado por dia+hora+minuto+segundo (`AAAA-MM-DD_HH-MM-SS.pdf`); `app.py`
   passou a usá-la com `pd.Timestamp.now()`. A capa do PDF também passou a mostrar
   a data/hora completa de geração (antes só a data), tanto no caminho padrão quanto
   no texto visível pro usuário (`generate_pdf_report`, parâmetro `report_date`).

150 testes seguem passando (nenhum teste automatizado novo — a lógica nova é
composição de dados já testados em `test_protective_put.py`/`test_entry_dqn.py`/
`test_csv_store.py`; cobertura ficou por validação manual do PDF gerado, incluindo o
caso adversarial de caracteres XML-inseguros).

**2026-10-14 (sessão 8, continuação) — tickers dos dois notebooks trocados pro setor
elétrico dos EUA**

Pedido do usuário: os tickers dos dois notebooks (não do app/watchlist) deviam ser do
setor elétrico dos EUA.

- `notebooks/put_de_protecao_DQN_montecarlos.ipynb` (`AnalysisSettings.tickers`, lista
  de 10): trocado de `("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
  "KO", "SPY")` para `("NEE", "DUK", "AEP", "D", "ETR", "XEL", "EXC", "ED", "PEG",
  "EIX")` — NextEra Energy, Duke Energy, American Electric Power, Dominion Energy,
  Entergy, Xcel Energy, Exelon, Consolidated Edison, Public Service Enterprise Group,
  Edison International. Exemplo ilustrativo na célula de configuração (seção 1)
  também trocado de `("AAPL", "KO")` para `("DUK", "NEE")`.
- `notebooks/black_scholes_dqn_guia.ipynb` (`GuideSettings.ticker`, único): trocado de
  `"AAPL"` para `"DUK"` (Duke Energy — o ativo mais testado ao longo do projeto, com
  cadeia de opções líquida confirmada em várias sessões anteriores).

Validação (não só escolha por nome): rodei `strategy.protective_put.analyze_ticker`
de `src/` — que replica a lógica do notebook sem mudar comportamento — contra o Yahoo
real, com os mesmos critérios do notebook (vencimento alvo 45 dias, strike 80–102% do
preço, open interest mínimo 100). Os 10 escolhidos resolveram e tiveram pelo menos 1
put líquida na faixa (de 1 a 5 strikes cada). Dois candidatos de setor/porte parecidos
que eu tinha escolhido antes **não passaram** nessa validação e foram trocados:
Southern Company (SO) e PG&E (PCG) resolvem no Yahoo mas não tinham puts líquidas
nessa faixa nesta data (2026-10-14) — substituídos por Public Service Enterprise
Group (PEG) e Edison International (EIX), que passaram. Isso reforça o padrão já
visto no setor elétrico da B3 (ver Fase 1): nem todo ticker do setor certo tem
derivativo líquido no Yahoo na data em que se testa — reavaliar periodicamente.

Nenhum teste automatizado afetado (mudança é só nos notebooks, não em `src/`); 150
testes seguem passando.

**2026-10-19 (sessão 9) — MetaTrader5 implementado: provider + coleta diária em log**

Pedido do usuário: implementar a parte de MT5 (até então um stub `NotImplementedError`,
decisão pendente desde a Fase 1/seção 0 do plan.md). Objetivo explícito: coletar do
MT5 (simulador/demo) e gravar dois CSVs **estilo log** — cada rodada só anexa linhas
embaixo das anteriores, nunca sobrescreve — pensado pra rodar uma vez por dia, por
vários dias, com reordenação por data no fim. Cada linha é o **fechamento do dia**
(nunca intradiário). Foco explícito: dados que alimentem direto o Black-Scholes
(entrada ação+opção) e o Monte Carlo (variação), então a estrutura precisa ser
"simples e legível", sem join entre os dois CSVs.

**Limitação de ambiente, registrada com transparência:** o pacote `MetaTrader5` só
instala em Python nativo do Windows e só funciona com o terminal aberto e logado —
nada disso existe neste ambiente de desenvolvimento (macOS, sem terminal MT5). Por
isso a implementação não pôde ser validada contra um MT5 de verdade, ao contrário do
resto do sistema (que sempre foi testado ao vivo antes de marcar como pronto). Pra não
abrir mão de testar o que dá pra testar, `MT5Provider` recebe o módulo `MetaTrader5`
por injeção de dependência (`mt5_module=`, default: importa o pacote de verdade) —
isso permitiu escrever um dublê (`FakeMT5`, mesmo padrão `FakeProvider` já usado em
`test_scheduler.py`/`test_overnight.py`) e testar toda a lógica em volta da API real
(parsing de `copy_rates_range`, fallback de preço, filtro de `option_right`/
`option_strike`/`expiration_time`, etc.) sem precisar do pacote real instalado.

- [x] `data/mt5_provider.py` reescrito — implementa `MarketDataProvider` via
      `MetaTrader5`: `download_price_history` (`copy_rates_range` + `TIMEFRAME_D1`),
      `get_current_stock_price` (tick `last`, cai pro `bid`), `list_option_expirations`/
      `download_put_chain`/`download_call_chain` (via `symbols_get(group=...)`,
      filtrando pelos campos de opção do MT5: `option_right`/`option_strike`/
      `expiration_time`). Duas limitações documentadas na própria docstring do módulo,
      não contornáveis (não são bug, são o que a API do MT5 realmente expõe):
      **sem fundamentalista** (`get_trailing_dividend_yield` sempre `0.0` — o dado
      fundamentalista completo do coletor continua vindo do Yahoo) e **sem taxa livre
      de risco padrão** (`get_risk_free_rate` exige um `risk_free_symbol` configurado
      pelo usuário, levanta `MarketDataError` claro se não tiver). `inspect_symbol(...)`
      novo: dump cru de `symbol_info()` pra o usuário conferir, contra um símbolo real
      do próprio simulador, se os campos de opção vêm preenchidos — a nomenclatura de
      símbolo de opção varia muito entre corretoras no MT5, não dá pra generalizar.
- [x] `collector/csv_store.py` ganhou `sort_csv_by(csv_path, by)` — reordena um CSV
      existente por coluna (usado pra "date" crescente), no-op seguro se o arquivo não
      existir/estiver vazio/não tiver a coluna.
- [x] `collector/mt5_daily_log.py` novo — `collect_stock_row` (fechamento mais recente
      do MT5 + fundamentalista do Yahoo, nunca lança, falha vira `status`, mesmo padrão
      defensivo de `overnight.py`), `collect_option_rows` (uma linha por call/put
      líquida do vencimento mais próximo, com `underlying_price` já na linha — decisão
      deliberada de duplicar esse dado em vez de exigir join, pra bater o pedido de
      "estrutura simples", já que o uso real é abrir o CSV de opções sozinho e rodar
      Black-Scholes direto) e `run_daily_collection` (orquestra os dois, anexa via
      `csv_store.append_rows`, reordena por data via `sort_csv_by` no fim de cada
      rodada). CLI própria (`main`) + `scripts/collect_mt5_daily.py`, mesmo par
      collector/script de `overnight.py`/`collect_overnight.py`.
- [x] `pyproject.toml`: grupo opcional `mt5 = ["MetaTrader5"]` (não é dependência
      padrão — não instala fora do Windows).
- [x] 33 testes novos, todos passando sem precisar do pacote `MetaTrader5` real:
      `tests/test_mt5_provider.py` (20, via `FakeMT5`), `tests/test_mt5_daily_log.py`
      (9, via `FakeProvider`), 4 novos em `tests/test_csv_store.py` (`sort_csv_by`).
      183 testes no total (eram 150).
- [x] README.md: seção "5b. Coleta diária via MT5" (uso, duas colunas do CSV,
      aviso sobre `inspect_symbol`) + linhas atualizadas nas tabelas de `data/`,
      `collector/` e `scripts/` + contagem de testes corrigida (129 → 183, já estava
      desatualizada antes desta sessão).

**O que o usuário ainda precisa validar manualmente** (não dá pra fazer neste
ambiente): conectar num terminal MT5 real/simulador, rodar `inspect_symbol(...)` num
símbolo de opção de verdade do corretor escolhido pra confirmar que `option_right`/
`option_strike`/`expiration_time` vêm preenchidos como o código espera, e ajustar
`option_group_pattern` se o padrão de nome dos símbolos do corretor for diferente de
`"*{ticker}*"`. Fica registrado como item de validação pendente, não como
"funcionalidade incompleta" — a lógica em volta da API está testada, a API em si não
pôde ser exercitada de verdade aqui.
