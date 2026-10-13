# Plano — finstrategy

Plataforma de apoio à decisão para ações + proteção com derivativos (puts), partindo do
notebook `put_de_protecao_DQN_montecarlos.ipynb` como núcleo quantitativo. Objetivo da
dissertação: evoluir o notebook de uma análise "roda uma vez e mostra gráfico" para um
sistema que **coleta dado contínuo no pregão, guarda tudo em CSV, treina/ajusta modelos
de RL, usa Black-Scholes para timing de saída do derivativo, e entrega uma recomendação
em PDF legível por quem não é técnico** — com um frontend simples para configurar todas as
variáveis (tickers, taxas, limites de risco, hiperparâmetros da DQN) sem editar código.

Primeiro recorte de ativos: **setor elétrico**, considerando todos os tipos de ação
disponíveis no Yahoo Finance (B3 `.SA` e ADRs/ETFs dos EUA do setor, sem restrição a
priori — ver decisão na seção 0), depois expandir.

---

## 0. Decisões de arquitetura (travar antes de codar)

- [x] **Risco identificado e decidido (2026-10-13):** o notebook hoje só busca cadeia de
  opções via `yfinance`, que **só publica opções para ativos listados nos EUA**. Ações do
  setor elétrico brasileiro (ex.: `TAEE11.SA`, `CPFE3.SA`, `ELET3.SA`, `EGIE3.SA`,
  `CMIG4.SA`, `TRPL4.SA`) não têm cadeia de opções no Yahoo.
  **Decisão do usuário:** neste primeiro momento, considerar **todos os tipos de ação
  disponíveis no Yahoo Finance**, sem restringir o universo a priori a ETF/ADR dos EUA.
  Na prática isso significa:
  - o universo do setor elétrico inclui tanto tickers B3 (`.SA`) quanto ADRs/ETFs dos
    EUA do setor (ex.: `XLU`, `EBR`, `ELP`) — a lista final fica em
    `config/tickers_setor_eletrico.yaml` (seção 5) e pode misturar os dois;
  - para **qualquer ticker sem cadeia de opções no Yahoo** (a maioria dos B3 `.SA`), o
    pipeline roda normalmente até a análise de ação + fundamentalista + DQN de entrada,
    e a parte de **put de proteção/Black-Scholes/DQN de saída fica marcada como "não
    elegível: sem opções no Yahoo"** para aquele ticker — mesmo tratamento de erro que o
    notebook já usa (`MarketDataError`), não é uma falha, é um estado esperado;
  - para tickers **com** cadeia de opções no Yahoo, o pipeline completo roda (ação +
    fundamentalista + put de proteção + Black-Scholes + DQN de entrada e saída);
  - revisitar essa decisão mais adiante só se a dissertação exigir cobertura de
    derivativo especificamente para ações B3 (aí sim entra a seção "buscar fonte
    alternativa de opções B3" como trabalho futuro, não bloqueando o MVP).
- [ ] Confirmar se **MetaTrader5** é alvo real do mestrado (precisa Windows ou wine, conta
  de corretora, terminal aberto) ou é "bom teria" — isso decide se entra no MVP ou no backlog.
- [ ] Escolher front-end: **Streamlit** (recomendado — um só processo Python, formulários
  para editar `AnalysisSettings`, tabelas, gráficos, botão de "rodar simulação agora",
  menor esforço de manutenção para um projeto de dissertação) vs. FastAPI+React (mais
  trabalho, só se quiser multiusuário/produção). Plano abaixo assume Streamlit.
- [ ] Escolher onde persistir: **CSV como fonte de verdade** (pedido explícito do usuário)
  + leitura via pandas; SQLite fica fora do escopo a menos que CSV vire gargalo.
- [ ] Definir o calendário de pregão a respeitar (B3: 10:00–17:00 BRT / EUA: 9:30–16:00 ET,
  conforme universo escolhido acima) e biblioteca de calendário (`pandas_market_calendars`).

---

## 1. Estrutura de repositório

- [ ] Criar o esqueleto de pacote Python (substitui o notebook solto por módulos
  testáveis; o notebook vira só a camada de exploração/relatório):
  ```
  finstrategy/
    README.md
    plan.md
    requirements.txt
    pyproject.toml
    config/
      settings.yaml            # AnalysisSettings versionado, editável pelo frontend
      tickers_setor_eletrico.yaml  # lista semente (seed) usada para popular a watchlist na 1ª vez
      watchlist.yaml            # NOVO: lista dinâmica de tickers que o usuário gerencia (qualquer ação)
    src/finstrategy/
      data/
        __init__.py
        base.py                 # interface MarketDataProvider (preço, dividendo, opções, taxa livre de risco)
        yahoo_provider.py        # hoje: yfinance (vem do notebook)
        mt5_provider.py          # depois: MetaTrader5
        fundamentals_provider.py # dados fundamentalistas (ver fase 2)
        symbol_search.py         # NOVO: busca/validação de ticker por nome ou código (qualquer ação do Yahoo)
        watchlist.py              # NOVO: CRUD da watchlist (adicionar/remover ticker, persistir em watchlist.yaml)
      pricing/
        black_scholes.py         # preço + Greeks + IV implícita
        garch_montecarlo.py      # hoje: fit_garch_t / simulate_prices_at_expiration (vem do notebook)
      strategy/
        protective_put.py        # ProtectivePutPlan, build_strike_menu, recommend_strike (vem do notebook)
        exit_timing_bs.py         # NOVO: janela de saída do derivativo via Black-Scholes/Greeks
      rl/
        entry_dqn.py              # hoje: QNetwork, EntryTimingEnvironment, train_entry_dqn (vem do notebook)
        exit_dqn.py                # NOVO: DQN de saída do derivativo
        replay_buffer.py
      collector/
        scheduler.py               # loop de coleta durante o pregão
        market_session.py          # calendário de pregão / está aberto agora?
        csv_store.py                # escrita append-safe em CSV (preço, opções, decisões, relatórios)
      reporting/
        panel.py                   # hoje: painel final do notebook
        pdf_report.py               # NOVO: PDF simples para não-técnico
      frontend/
        app.py                     # Streamlit: configurar settings, rodar simulação, ver histórico
    data_store/
      market/                      # CSV de pregão (preço, opções) por ticker/dia
      fundamentals/                 # CSV de dados fundamentalistas por ticker/trimestre
      model_runs/                   # CSV de decisões/saídas do modelo (DQN, recomendação de strike)
      reports/                      # PDFs gerados + CSV de metadados de cada relatório
    notebooks/
      put_de_protecao_DQN_montecarlos.ipynb   # mantido como laboratório/validação
    tests/
      test_black_scholes.py
      test_garch_montecarlo.py
      test_protective_put.py
      test_entry_dqn.py
      test_csv_store.py
  ```
- [ ] Mover as funções do notebook atual para os módulos equivalentes **sem reescrever a
  lógica** (copiar/colar + ajustar imports), validando com um teste simples por módulo
  antes de seguir — o notebook já foi validado, então o objetivo aqui é só organizar.
- [ ] `requirements.txt` → migrar para `pyproject.toml` com deps: `numpy`, `pandas`, `scipy`,
  `yfinance`, `streamlit`, `pandas_market_calendars`, `reportlab` (ou `weasyprint`),
  `pydantic` (validação de `AnalysisSettings`), `pytest`.

---

## 2. Camada de dados (Yahoo → MetaTrader5, fundamentalista, derivativos)

- [ ] Extrair `MarketDataProvider` (interface) com os métodos que já existem no notebook:
  `get_price_history`, `get_current_price`, `get_dividend_yield`, `get_risk_free_rate`,
  `list_option_expirations`, `get_put_chain` — e também `get_call_chain` (hoje só put).
- [ ] Implementar `YahooProvider` = código atual do notebook, sem mudar comportamento.
- [ ] Implementar `MT5Provider` (stub primeiro, com `NotImplementedError` claro) — preencher
  só depois que a decisão da seção 0 confirmar que MT5 entra no MVP.
- [ ] Dados fundamentalistas — decidir fonte por disponibilidade/custo e implementar:
  - [ ] Tickers com cobertura boa no Yahoo (EUA/ADR, e também a maioria dos B3 `.SA`):
    `yfinance` `.info`, `.financials`, `.balance_sheet`, `.cashflow`, `.earnings` (cobre
    P/L, ROE, dívida/patrimônio, margem, crescimento de receita) — primeira fonte, já
    que a decisão da seção 0 é usar o Yahoo para todos os tipos de ação neste momento.
  - [ ] Para tickers B3 onde o Yahoo vier incompleto: avaliar fonte gratuita complementar
    (ex. Fundamentus, dados da B3/CVM) como trabalho futuro, não bloqueando o MVP.
  - [ ] Normalizar tudo em um schema único de indicadores (ex.: `pe_ratio`, `roe`,
    `debt_to_equity`, `dividend_yield_ttm`, `revenue_growth_yoy`, `free_cash_flow`).
- [ ] Dados de derivativos (cadeia de opções) — já existe para puts EUA via Yahoo; adicionar:
  - [ ] cadeia de calls (para completar análise de derivativos, não só proteção);
  - [ ] marcar como "não elegível: sem opções no Yahoo" qualquer ticker sem cadeia
    (a maioria dos B3 `.SA`, conforme decisão da seção 0) em vez de tratar como erro;
  - [ ] fonte alternativa para opções B3 fica como trabalho futuro (fora do MVP).
- [ ] **Busca/validação de ticker (`symbol_search.py`)** — pré-requisito direto da seção 5
  ("a etapa de seleção de ativos tem que funcionar para qualquer ação que eu enviar"):
  - [ ] `search_symbols(texto: str) -> list[SymbolMatch]`: busca por nome da empresa ou
    código parcial e devolve candidatos (`symbol`, `nome`, `bolsa/exchange`, `tipo`) —
    usar o endpoint de autocomplete/busca do Yahoo Finance (o mesmo que alimenta a busca
    do site; `yfinance` tem `Search`/`Lookup` em versões recentes, senão chamar o
    endpoint HTTP diretamente). Isso é o que permite o frontend "listar todas as ações
    que existem" para o usuário marcar;
  - [ ] `validate_symbol(ticker: str) -> SymbolInfo | MarketDataError`: tenta buscar preço
    atual + nome da empresa; se o Yahoo não reconhecer o ticker, devolve erro claro
    ("ticker não encontrado no Yahoo Finance") em vez de deixar o pipeline quebrar mais
    na frente — é a validação usada quando o usuário digita um ticker direto, sem passar
    pela busca.
- [ ] Testes: provider mock que não bate na rede, para rodar CI sem depender do Yahoo estar de pé.

---

## 3. Black-Scholes e timing de saída do derivativo

- [ ] Implementar `black_scholes.py`:
  - [ ] preço teórico de call/put europeia;
  - [ ] Greeks: delta, gamma, theta, vega, rho;
  - [ ] volatilidade implícita a partir do preço de mercado (Newton-Raphson ou Brent),
    para comparar com a volatilidade GARCH já calculada no notebook (gap = sinal de
    opção cara/barata).
- [ ] Projetar `exit_timing_bs.py` — a parte nova pedida pelo usuário ("usar Black and
  Scholes para prever melhor janela de tempo de saída/compra do derivativo"):
  - [ ] modelar a **curva esperada de valor da put ao longo do tempo** (theta decay),
    repricando BS dia a dia com a vol implícita atual e o preço simulado pelo GARCH
    Monte Carlo já existente;
  - [ ] definir regra/objetivo de saída: ex. vender a put quando o valor-tempo restante
    cair abaixo de X% do prêmio pago, ou quando delta/gama indicarem que o custo de
    manter a proteção supera o benefício marginal de risco (ligar com VaR/ES do notebook);
  - [ ] formalizar isso como **função de recompensa** candidata para o DQN de saída
    (seção 4), não só uma regra fixa — o DQN aprende a timing, o BS fornece os sinais
    de entrada (delta, theta, gap de IV) que alimentam o estado do agente.
- [ ] Validar com dados históricos de opções (backtest simples: "se eu tivesse vendido
  a put no dia X sugerido pela regra, qual teria sido o resultado vs. manter até o
  vencimento?").

---

## 4. Modelos de Deep Q-Network

- [ ] **DQN de entrada (ações)** — já existe no notebook (`QNetwork`,
  `EntryTimingEnvironment`, `train_entry_dqn`, avaliação com bootstrap e p-valor contra
  benchmarks). Só precisa ser extraído para `rl/entry_dqn.py` e re-testado.
- [ ] **DQN de saída (derivativo)** — novo agente, pedido pelo usuário, para decidir
  quando encerrar a posição de put (vender a put ou deixar expirar) em vez de usar
  só uma regra fixa de Black-Scholes:
  - [ ] estado: dias restantes até vencimento, delta/theta/vega atuais (de BS), gap
    IV-vs-GARCH, lucro/perda não realizado da put, VaR/ES marginal de manter vs. sair;
  - [ ] ações: manter, vender/encerrar;
  - [ ] recompensa: redução de risco realizada (ES/VaR evitado) menos custo de
    transação, ponderando o objetivo do usuário de "redução máxima de risco financeiro"
    (não é só maximizar retorno — pesar explicitamente a cauda, ex. CVaR na recompensa);
  - [ ] reusar a arquitetura MLP NumPy + Double DQN + replay buffer já validada no
    notebook, trocando só ambiente/estado/recompensa.
- [ ] Avaliação honesta igual à que já existe (treino em 70% antigo, teste em 30% novo
  sem sobreposição, comparação com benchmarks, bootstrap de IC, p-valor) — aplicar ao
  DQN de saída também.
- [ ] Registrar em CSV cada decisão dos dois DQNs (estado, ação escolhida, valor Q de
  cada ação, recompensa realizada quando souber) — isso alimenta a seção 6 (coleta
  contínua) e serve de log para a dissertação (gráficos de desempenho ao longo do tempo).

---

## 5. Fundamentalista + seleção de ativos (qualquer ação, watchlist dinâmica)

**Requisito do usuário (2026-10-13):** esta etapa não pode ficar travada numa lista fixa
do setor elétrico. Tem que funcionar para **qualquer ação que o usuário enviar**, uma ou
várias de uma vez, e o sistema busca a informação sozinho. O setor elétrico é só a
**semente inicial** da watchlist, não um filtro embutido no código.

- [ ] Modelar a watchlist como dado, não como código: `config/watchlist.yaml` com uma
  lista de `{ticker, apelido?, tags: [setor_eletrico, ...], adicionado_em}` — o pipeline
  de dados/fundamentalista/derivativo (seções 2–4) sempre lê **a watchlist**, nunca uma
  lista de tickers hardcoded num módulo Python.
- [ ] Popular `watchlist.yaml` na primeira execução a partir de
  `config/tickers_setor_eletrico.yaml` (a semente), mas permitir adicionar/remover
  qualquer ticker depois, por três caminhos equivalentes:
  - [ ] **API/função Python:** `watchlist.add_tickers(["PETR4.SA", "AAPL", ...])` —
    aceita um ou mais tickers de uma vez;
  - [ ] **Frontend, digitando:** campo de texto que aceita um ticker ou uma lista
    separada por vírgula/linha, valida cada um com `validate_symbol` (seção 2) e mostra
    erro claro para os que não forem reconhecidos, sem travar os que são válidos;
  - [ ] **Frontend, buscando:** campo de busca (chama `search_symbols`, seção 2) que
    lista ações correspondentes (nome, código, bolsa) com **checkbox** para marcar
    quais entram na watchlist — é o "ele mesmo informar todas as ações que existem e eu
    buscar e selecionar marcando as que eu quero" pedido pelo usuário.
- [ ] Ao adicionar um ticker novo à watchlist, disparar automaticamente (sem passo manual
  extra) a primeira coleta: preço atual, histórico, fundamentalista e checagem de
  elegibilidade a derivativo (`elegivel_derivativo: sim/não`) — o usuário só manda o
  ticker, o sistema busca tudo.
- [ ] Rodar a coleta fundamentalista (seção 2) para toda a watchlist e montar um
  **ranking simples** (ex.: dividend yield, ROE, dívida/patrimônio, geração de caixa),
  reaproveitável para qualquer subconjunto de tags (setor elétrico é só uma tag entre
  outras) — isso cobre o pedido "ter todos os dados fundamentalistas de análise de
  ações e análise de derivativos" para qualquer ação, não só o recorte inicial.
- [ ] Documentar critérios de corte do ranking (ex.: liquidez mínima, cobertura de
  dívida) para metodologia explícita na dissertação, mas deixar claro que o corte é um
  filtro de **exibição/priorização**, não de **elegibilidade de entrada na watchlist**
  — qualquer ticker válido entra, o ranking só ordena.

---

## 6. Coleta contínua durante o pregão (CSV)

- [ ] `market_session.py`: função `pregao_esta_aberto(agora, calendario)` usando
  `pandas_market_calendars` (BVMF para B3, NYSE/NASDAQ para EUA conforme universo).
- [ ] `csv_store.py`: escrita append-safe (lock de arquivo ou escrita por partição
  diária) para três famílias de CSV:
  - [ ] `data_store/market/{ticker}/{YYYY-MM-DD}.csv` — preço, bid/ask das puts/calls
    coletadas, volume, open interest, timestamp de coleta;
  - [ ] `data_store/model_runs/{YYYY-MM-DD}.csv` — saída de cada rodada de modelo
    (strike recomendada, ação do DQN de entrada/saída, métricas de risco, Black-Scholes
    greeks usados);
  - [ ] `data_store/reports/{YYYY-MM-DD}.csv` — metadados de cada PDF gerado (caminho,
    ativos cobertos, hash da config usada, para rastreabilidade na dissertação).
- [ ] `scheduler.py`: loop (`while True` com `sleep` ajustável) que, só durante o pregão:
  1. coleta preço/opções de cada ticker configurado;
  2. roda o Monte Carlo + Black-Scholes + DQNs no intervalo configurado (não
     necessariamente a cada tick — ex. a cada 15–30 min para não estourar rate limit
     do Yahoo e não treinar DQN em excesso);
  3. grava tudo em CSV;
  4. ao fim do pregão, dispara a geração do relatório/PDF do dia (seção 7).
- [ ] Tratar rate limit / falha de rede do Yahoo com backoff e log claro (reaproveitar
  o padrão `MarketDataError` do notebook, que já marca "não elegível" com motivo).
- [ ] Modo "dry-run"/replay: permitir rodar o coletor sobre CSV já salvo (sem rede) para
  testar o pipeline fora do horário de pregão.

---

## 7. Relatório PDF para usuário não técnico

- [ ] Definir o layout do PDF (usar a skill de PDF do projeto na hora de gerar):
  - [ ] resumo em 1 página por ativo: "o que é", "comprar/esperar/evitar hoje", "quanto
    você pode perder no máximo", "a partir de que preço começa o lucro" — linguagem
    direta, sem jargão (o notebook já tem esse texto pronto em português na seção 10,
    é a base).
  - [ ] aviso de disclaimer (já existe texto no notebook, reaproveitar).
  - [ ] gráfico simples (payoff ação vs. ação+put) incluído como imagem.
- [ ] Implementar `pdf_report.py` usando `reportlab` ou `weasyprint` (HTML→PDF, mais fácil
  de estilizar) a partir do painel final (`panel.py`).
- [ ] Gerar automaticamente ao fim de cada pregão (seção 6) e também sob demanda pelo
  frontend (seção 8, botão "gerar relatório agora").

---

## 8. Frontend simples (Streamlit)

- [ ] **Tela "Ações" (watchlist)** — tela nova, cobre o pedido da seção 5:
  - [ ] campo de busca por nome/código (chama `search_symbols`) com os resultados em
    tabela com **checkbox por linha**; botão "adicionar selecionadas à watchlist";
  - [ ] campo de texto livre para colar um ou vários tickers de uma vez (ex.
    `AAPL, PETR4.SA, TAEE11.SA`), valida cada um e mostra na hora quais entraram e quais
    falharam (com o motivo do Yahoo);
  - [ ] tabela da watchlist atual, com tag (ex. `setor_eletrico`), preço mais recente
    coletado, `elegivel_derivativo` (sim/não) e botão de remover;
  - [ ] ao adicionar, mostrar feedback de progresso (buscando preço, fundamentalista,
    checando opções) já que a busca de alguns dados pode levar alguns segundos.
- [ ] Tela "Configuração": formulário para editar `AnalysisSettings` (limites de
  perda, taxas de corretagem, parâmetros Monte Carlo, hiperparâmetros da DQN) e salvar em
  `config/settings.yaml` — sem precisar editar código.
- [ ] Tela "Rodar agora": botão que dispara uma rodada completa (dados → BS/GARCH → DQNs
  → painel) fora do loop do coletor, para teste manual.
- [ ] Tela "Histórico": ler os CSVs de `data_store/` e mostrar preço coletado, decisões do
  modelo e link para os PDFs gerados.
- [ ] Tela "Coletor": status do `scheduler.py` (rodando? desde quando? próxima coleta?),
  com botão start/stop — decidir se o coletor roda como processo separado (recomendado,
  mais robusto) e o Streamlit só lê o status, ou se o Streamlit o controla diretamente.
- [ ] Tela "Modelos": re-treinar DQN de entrada/saída com os dados acumulados, ver
  métricas de avaliação (a mesma tabela/IC do notebook) antes de promover o modelo novo.

---

## 9. Testes, backtest e validação (para a dissertação)

- [ ] Testes unitários por módulo (listados na estrutura da seção 1).
- [ ] Backtest de ponta a ponta: rodar o pipeline completo sobre um período histórico e
  comparar decisão do sistema vs. benchmarks simples (comprar e manter, estratégia sem
  derivativo) — isso já existe parcialmente para a DQN de entrada, estender para o
  conjunto (entrada + proteção + saída via BS/DQN).
- [ ] Métricas de avaliação para o capítulo de resultados: redução de VaR/ES com proteção,
  economia média da DQN de entrada (já existe), ganho/perda da DQN de saída vs. "manter
  até o vencimento" e vs. regra fixa de Black-Scholes.
- [ ] Plano de coleta de dados real: quantos dias de pregão rodando o coletor são
  necessários antes de ter amostra suficiente para re-treinar/validar os modelos com
  dados "ao vivo" (não só histórico baixado de uma vez).

---

## 10. Reformulação do frontend — sistema auto-explicativo e guiado (2026-10-13)

**Pedido do usuário:** o frontend (seções 7/8, já implementadas) funciona, mas ficou
técnico demais e fragmentado. O sistema precisa ser capaz de **conduzir sozinho**
qualquer pessoa — de um administrador sem nenhuma bagagem técnica a alguém de IA que
quer entender/ajustar o modelo — até uma decisão de hedge com ações, incluindo as
métricas de risco, sem precisar de ninguém explicando do lado. Cinco pedidos
concretos, nessa ordem:

1. Gráficos de ação e derivativo (como no notebook) não aparecem no Streamlit hoje.
2. A tela "Rodar agora" é redundante com "Histórico & Relatórios" — juntar as duas.
3. A tela "Modelos" (DQN) precisa de mais explicação, métricas visíveis, e salvar os
   modelos treinados em pickle (pesos, métricas, previsões), em vez de treinar do zero
   toda vez e perder tudo ao navegar pra outra tela.
4. Precisa de uma tela que mostre os dados brutos e fundamentalistas que o sistema
   baixou — tanto para o usuário leigo entender "de onde vêm esses números" quanto
   para quem é técnico auditar/depurar.
5. O sistema como um todo precisa ser auto-explicativo para dois públicos ao mesmo
   tempo (administrador leigo + pessoa técnica de IA), e guiar qualquer pessoa até
   fazer um hedge com métricas de risco.

**Pesquisa feita (pedido explícito: "pode fazer uma busca web") antes de planejar —
como ferramentas de risco/investimento reais resolvem "dois públicos, um produto":**

- **Divulgação progressiva (progressive disclosure):** o número simples fica visível
  de cara; o detalhe técnico fica atrás de um toggle/expander, não numa tela separada
  que o leigo nunca vai abrir e o técnico sempre vai precisar abrir. Fonte:
  [Lollypop — Investment Dashboard UX Design 2026](https://lollypop.design/blog/2026/may/investment-dashboard-ux-design-guide/).
- **Métrica nunca sozinha:** todo número de risco vem com uma frase em português do
  que ele significa, não só o valor (ex.: "VaR 95%: 8,2% — em 95% dos cenários
  simulados, a perda não passa disso"). Fonte: práticas descritas em dashboards de
  risco de portfólio (VaR/Sharpe/max drawdown com "nota de uma frase" por métrica).
- **Benchmarking contextual:** nunca "você perdeu X%" sozinho — sempre "o mercado caiu
  Y%, você caiu X%" (dá pra ver se foi bom ou ruim *relativo*, não só em valor
  absoluto). Mesma fonte Lollypop.
- **Glossário como tooltip inline, não como página à parte** — ninguém lê uma página
  de glossário antes de usar o produto; o "?" do lado da métrica, sim.
- **"Model card"** (termo padrão de MLOps/explicabilidade): para cada modelo treinado,
  um resumo com dado usado, hiperparâmetros, métricas de avaliação, e explicação tanto
  **global** (o que pesa na decisão em geral) quanto **local** (por que decidiu isso
  hoje, pra esse ativo) — é como o Azure Responsible AI Dashboard e ferramentas de
  MLOps (Comet, W&B, Evidently) expõem modelo pra público técnico sem exigir que o
  leigo entenda a arquitetura. Fonte: busca "MLOps model card explainability global
  local" (Azure ML docs, Comet/W&B, InterpretML).

### 10.1 Portar os gráficos do notebook para o Streamlit

O notebook (`notebooks/put_de_protecao_DQN_montecarlos.ipynb`) já tem 3 visualizações
validadas que o app Streamlit não mostra hoje (só tabelas e texto):

- [ ] **Menu de strikes** (`plot_strike_menu` no notebook): custo da proteção (%) no
  eixo X vs. perda máxima (%) no eixo Y, uma bolinha por strike líquida, a recomendada
  destacada maior/colorida com anotação, linha tracejada no limite de perda aceitável.
- [ ] **Payoff + distribuição** (`plot_protection_payoff`): painel de cima — resultado
  no vencimento (R$) para "só a ação" vs. "ação + put", com a perda máxima, a strike e
  o break-even marcados; painel de baixo — histograma da distribuição de preços
  simulada pelo Monte Carlo, com o preço atual marcado.
- [ ] **Avaliação da DQN de entrada** (`plot_entry_evaluation`, estilo *forest plot*):
  economia média de cada estratégia (DQN vs. 3 referências) com intervalo de confiança
  de 95%, linha vertical no zero.
- [ ] **Novo (não existe nem no notebook):** curva de valor-tempo da put
  (`strategy.exit_timing_bs.time_value_curve`) plotada — mostra visualmente por que a
  regra de saída recomenda sair naquele dia.
- [ ] **Novo:** mesmo estilo de *forest plot* da DQN de entrada, mas para a DQN de
  saída (`rl.exit_dqn.evaluate_exit_timing`) — DQN vs. "manter até o vencimento" vs.
  regra fixa de Black-Scholes.
- [ ] Decidir biblioteca: reaproveitar `matplotlib` (código já existe e validado no
  notebook, só adaptar pra `st.pyplot`) é o caminho rápido; trocar por `plotly`/`altair`
  dá interatividade (zoom, hover com valor exato) mas exige reescrever os gráficos —
  fazer com `matplotlib` primeiro, avaliar depois se vale a pena.

### 10.2 Consolidar "Rodar agora" dentro de "Histórico & Relatórios"

- [ ] Remover a tela "Rodar agora" (hoje só chama `run_full_pipeline` +
  `render_pipeline_results` + `generate_and_offer_pdf`, exatamente igual ao botão
  "Gerar novo relatório agora" que já existe em "Histórico & Relatórios" — telas
  duplicadas, só uma precisa existir).
  — Ver nota em `app.py`, essas 3 funções já são compartilhadas; a consolidação é só
  remover a página extra do `st.sidebar.radio` e o bloco `elif PAGE == "Rodar agora"`.
- [ ] Resultado: 5 telas em vez de 6 (Ações, Configuração, Histórico & Relatórios,
  Coletor, Modelos) — menos "qual tela eu uso?" pra quem não é técnico.

### 10.3 Tela "Modelos" — mais explicação, métricas visíveis, persistência em pickle

**Problema de hoje:** a DQN treina do zero toda vez que a tela é usada; nenhuma
métrica, peso ou previsão fica salva; não dá pra comparar execuções ao longo do tempo
nem reaproveitar um modelo já treinado sem esperar o treino de novo.

- [ ] **Persistência em pickle**: salvar cada modelo treinado
  (`rl.entry_dqn.QNetwork`/`rl.exit_dqn` + metadados) em
  `data_store/models/<ticker>_<entry|exit>_<timestamp>.pkl`, contendo: pesos da rede
  (`QNetwork.weights`), hiperparâmetros usados (`AnalysisSettings` relevantes),
  métricas de avaliação completas (a tabela de `evaluate_entry_timing`/
  `evaluate_exit_timing`, incluindo IC 95% e p-valor), ação recomendada no momento do
  treino, e um índice (`data_store/models/index.csv`) listando todos os modelos
  salvos (ticker, tipo, data, resumo de 1 linha da métrica principal) pra tela
  carregar sem escanear arquivo por arquivo.
- [ ] **Tela "Modelos" reformulada**: listar os modelos já treinados (do índice),
  permitir **carregar um salvo** (sem retreinar — mostra métricas/pesos/previsão na
  hora) ou **treinar um novo e salvar**. Hoje treina e descarta.
- [ ] **"Model card" por modelo** (explicação pros dois públicos, lado a lado, com
  divulgação progressiva):
  - Visível de cara (pro administrador leigo): 1 parágrafo em português simples —
    "esse modelo olha os últimos N dias da ação e decide se compra hoje ou espera;
    nos testes com dados que ele nunca viu, teria economizado X% em relação a comprar
    direto, e isso foi estatisticamente significativo (ou não)". Reaproveita o texto
    que `describe_entry_signal` (`reporting/panel.py`) já gera, expandido.
  - Atrás de um expander "Detalhes técnicos" (pra quem é de IA): hiperparâmetros
    exatos, tabela completa de métricas (médias, IC 95%, p-valor vs. cada benchmark),
    quais sinais de estado (RSI, retorno 5d, distância da média, etc.) tiveram maior
    peso na decisão do dia (explicação **local** — olhar os pesos da camada de saída
    aplicados ao estado de hoje) e, se o usuário quiser, poder baixar o `.pkl` direto.
- [ ] Mesmo tratamento para a DQN de saída (`rl/exit_dqn.py`), que hoje só é chamável
  via código, nem aparece na tela "Modelos".

### 10.4 Nova tela/seção "Dados" — dados brutos e fundamentalistas baixados

Pedido direto: "deixe uma parte no streamlit que leia os dados fundamentalistas e
dados brutos que o sistema baixou".

- [ ] Nova seção (dentro de "Histórico & Relatórios" ou tela própria "Dados", decidir
  na implementação) mostrando, por ticker escolhido:
  - histórico de preço **completo** baixado (não só o resumo de metric cards de hoje)
    — gráfico de linha do preço ao longo do tempo + tabela crua exportável;
  - **todos** os campos do `FundamentalSnapshot` (hoje só 6 viram card; o resto —
    P/VP, dívida/patrimônio, crescimento de receita, etc. — já existe no expander
    "Ver todos os indicadores", só falta deixar isso claramente rotulado como "dado
    bruto" e não só "mais detalhe");
  - a cadeia de opções coletada na última rodada (`liquid_puts`, de
    `analyze_ticker`), quando o ticker for elegível a derivativo — hoje essa tabela
    existe só durante a rodada de "Rodar agora"/relatório, não fica visível depois;
  - botão de exportar qualquer uma dessas tabelas em CSV.
- [ ] Objetivo duplo: leigo entende "de onde vêm esses números" (transparência), e
  quem é técnico consegue auditar/depurar sem abrir o código.

### 10.5 Fluxo guiado de hedge (novo — "wizard")

Hoje o usuário precisa descobrir sozinho em qual das 5-6 telas fazer cada parte do
processo. Pedido: "esse sistema por si só tem que ser capaz de conduzir qualquer
pessoa a fazer um hedging com ações, inclusive com métricas de risco".

- [ ] Um fluxo passo a passo (reorganiza telas existentes, não duplica lógica nova):
  1. Escolher o ativo (tela Ações, ou direto aqui via busca/lista);
  2. Ver se tem proteção disponível (derivativo elegível ou não, com o motivo);
  3. Ver o menu de strikes com a recomendada já marcada + gráfico (seção 10.1);
  4. Ver o payoff/risco em gráfico, com as métricas de risco explicadas em uma frase
     cada (VaR, ES, P(perda), redução de risco com a put) — divulgação progressiva;
  5. Ver o sinal de entrada da DQN (model card resumido, seção 10.3);
  6. Baixar o relatório em PDF (já existe, seção 7).
- [ ] Decidir se isso é uma tela nova "Fazer um hedge" (um wizard de verdade, com
  `st.session_state` guardando o progresso) ou só a tela "Histórico & Relatórios"
  reorganizada nessa ordem — avaliar na implementação qual fica mais simples sem
  duplicar as 5 telas já existentes.

### Observação importante

Esta seção foi planejada e, na sessão seguinte (2026-10-13, Fase 11 do `tasks.md`),
**implementada e testada** (com o app real no navegador, dados reais). Ver `tasks.md`,
Fase 11, para o que foi feito e os bugs reais encontrados pelo caminho.

---

## 11. Análise de risco com Black-Scholes no relatório principal (2026-10-14)

**Pedido do usuário:** o relatório (seção 10) já mostra o gráfico de payoff "ação sem
proteção vs. ação com proteção" (`plot_protection_payoff`, existe desde a Fase 11) —
mas falta a camada de **análise de risco** que realmente usa Black-Scholes pra
explicar o preço da opção, dizer quando entrar, quantos dias até vender a proteção, e
dar uma leitura de tendência/confiança do cálculo.

**Verificado no código antes de planejar** (2026-10-14): `pricing/black_scholes.py` e
`strategy/exit_timing_bs.py` existem, são usados pela DQN de saída e têm testes — mas
`grep` confirma **zero chamadas** a eles fora da tela "Modelos" > "Treinar novo" >
"Saída". Ou seja: Black-Scholes nunca aparece no relatório principal ("Histórico &
Relatórios" > "Gerar novo relatório agora"), que é a tela que a maioria das pessoas
vai realmente usar pra decidir — ele fica isolado atrás de um fluxo de treino manual
de DQN que a maior parte dos usuários não vai nem saber que existe. Esse é o gap real.

### 11.1 Preço da opção explicado por Black-Scholes
- [ ] Para a strike recomendada de cada ticker elegível, calcular
  `black_scholes.price_and_greeks("put", stock_price, strike, dias_até_vencimento/365,
  taxa_livre_de_risco, implied_volatility, dividend_yield)` — todos os dados já vêm de
  `analyze_ticker`, sem bater na rede de novo.
- [ ] Mostrar lado a lado: **preço de mercado** (prêmio realmente pago) vs. **preço
  teórico** (Black-Scholes) — com leitura em português: "prêmio X% acima/abaixo do
  valor teórico" (sinal de opção cara/barata em relação ao que o modelo acha que ela
  vale, dada a volatilidade implícita do mercado).
- [ ] Mostrar os Greeks da strike recomendada (delta, theta por dia, vega), cada um
  com 1 frase explicando o que significa pra quem não é técnico:
  - delta: quanto a put ganha de valor pra cada R$1 que a ação cai;
  - theta: quanto a put perde de valor por dia só de o tempo passar (o "aluguel" da
    proteção);
  - vega: quanto a put ganha/perde se a volatilidade do mercado aumentar/diminuir.

### 11.2 Quando vender a proteção (dias até a saída, em número concreto)
- [ ] Reaproveitar `exit_timing_bs.time_value_curve` + `recommend_exit_day` (já
  implementados e testados — só nunca chamados fora da tela "Modelos") pra **todo
  ticker elegível a derivativo, direto no relatório principal**, sem precisar que o
  usuário vá treinar a DQN de saída manualmente.
- [ ] Mostrar um número concreto: "pela regra de Black-Scholes, considere vender a
  proteção em até N dias (por volta de DD/MM), quando o valor-tempo cair abaixo de
  X% do que foi pago" — não só o gráfico, a frase em dias é o que o usuário pediu.
- [ ] `plot_time_value_curve` (já existe em `charts.py`, hoje só usado dentro do model
  card da DQN de saída) entra no relatório principal também, com o dia recomendado
  marcado.
- [ ] Opcional, sem bloquear o resto: se já existir um modelo de saída salvo pra esse
  ticker (`model_store.list_models()` filtrado), mostrar o que a DQN diria também,
  como segunda opinião ao lado da regra fixa — sem forçar treino na hora (pode ser
  lento) nem exigir que o usuário tenha treinado um antes.

### 11.3 Previsão de tendência do preço (ação e opção)
- [ ] A partir de `result["simulated_prices"]` (distribuição Monte Carlo GARCH-t, já
  calculada em `analyze_ticker`), extrair e mostrar como métricas (não só como
  histograma, que já existe): probabilidade do preço terminar acima/abaixo do preço
  de hoje, preço médio esperado no vencimento, faixa de 90% (percentil 5 a 95).
- [ ] Repetir a mesma leitura pro valor da **put**, não só da ação: usar
  `black_scholes.price_and_greeks_array` sobre a distribuição de preços simulada pra
  ter a distribuição do valor da put no vencimento, e mostrar a faixa esperada de
  valorização/desvalorização da proteção.
- [ ] Deixar explícito que isso é uma simulação calibrada no histórico (GARCH-t), não
  uma previsão garantida — mesma ressalva que já existe no notebook original e no PDF.

### 11.4 "Confiança" do cálculo — definir com rigor, sem inventar um número solto
Três noções diferentes de "confiança"/"risco", que não podem virar um número só
genérico (isso seria enganoso) — documentar e mostrar cada uma separadamente:
- [ ] **Confiança estatística do sinal da DQN** — já calculada (`p_value_vs_*` +
  `beats_benchmarks` em `evaluate_entry_timing`/`evaluate_exit_timing`). É sobre o
  SINAL de entrada/saída, não sobre o preço. Mostrar como "a DQN bateu as referências
  com significância estatística (p-valor Y)" quando `beats_benchmarks=True`, e um
  aviso claro de "sinal fraco" quando não bateu (o texto em `panel.py` já faz isso
  parcialmente — reaproveitar/unificar aqui).
- [ ] **Precisão da simulação de Monte Carlo** — não é "confiança de que o preço vai
  subir", é o quão estreito é o intervalo de confiança da média simulada (bootstrap).
  Generalizar `bootstrap_mean_interval` (hoje só em `rl/entry_dqn.py`) pra uma função
  utilitária reaproveitável sobre qualquer array, aplicar em `simulated_prices` e
  mostrar a largura do IC como proxy de precisão da simulação — **nunca** apresentar
  isso como "confiança de alta/queda", seria matematicamente errado (o Monte Carlo é
  neutro ao risco por construção, não aposta em direção).
- [ ] **Risco de entrada "naquele momento"** — já existem (P(perda), ES 97,5%, seção 6
  do plano original); só precisam ficar reunidos nesta nova seção de "Análise de
  Risco" em vez de espalhados em tabelas separadas.

### 11.5 Nova seção "📊 Análise de Risco" no relatório (reorganização visual)
- [ ] Juntar 11.1–11.4 numa seção clara dentro do expander de cada ticker no
  relatório, **antes** dos gráficos técnicos que já existem. Ordem sugerida (mesma
  divulgação progressiva da seção 10): 3-4 métricas de destaque (perda máxima com
  proteção, preço teórico vs. pago, dias até considerar vender, confiança do sinal de
  entrada) → gráfico de payoff (já existe) → gráfico de valor-tempo (novo aqui) →
  "Detalhes técnicos" atrás de expander (Greeks completos, p-valores, largura do IC).
- [ ] Estender `RISK_METRIC_HELP` (tooltips já implementados na seção 10) com as
  métricas novas desta seção, mesmo padrão.
- [ ] Testes: `pricing/black_scholes.py` e `strategy/exit_timing_bs.py` já têm
  cobertura própria (plan.md seção 3); o que falta testar é a função nova que monta
  o pacote de "análise de risco" por ticker (11.1-11.4 combinados) com dados fake,
  igual ao padrão já usado em `test_charts.py`/`test_model_store.py`.

### Observação importante

Planejado e implementado em 2026-10-14 (mesmo dia — seguiu direto pra execução depois
do plano). Ver `tasks.md`, Fase 12, para o que foi feito e como foi testado (navegador
real, dados reais de DUK e TAEE11.SA).

---

## Ordem recomendada de execução

1. Seção 0 (decisões) — sem isso tudo depois pode mudar de escopo.
2. Seção 1 (estrutura) + extrair código do notebook para módulos, com testes.
3. Seção 2 (fundamentalista + derivativos adicionais, inclui busca/validação de ticker)
   e Seção 5 (watchlist dinâmica, semeada com o setor elétrico).
4. Seção 3 (Black-Scholes) — pré-requisito de dados para a DQN de saída.
5. Seção 4 (DQN de saída) — depende de 3.
6. Seção 6 (coletor contínuo) — só depois que o pipeline de modelo (2–5) estiver estável.
7. Seção 7 (PDF) e Seção 8 (frontend) — podem andar em paralelo, ambos consomem o que já
   existe nas seções anteriores.
8. Seção 9 (validação/backtest) — contínuo, mas o capítulo de resultados da dissertação
   depende de ter dados reais coletados pela seção 6 por um tempo mínimo.
9. Seção 10 (reformulação do frontend) — depende de 3/4/5/7/8 já estarem prontos
   (estão). Ordem interna sugerida: 10.2 (consolidar telas, rápido e reduz confusão
   antes de adicionar mais coisa) → 10.1 (gráficos) → 10.4 (dados brutos) → 10.3
   (persistência de modelo, mais trabalho) → 10.5 (wizard, amarra tudo por último).
10. Seção 11 (Black-Scholes no relatório principal) — depende de 10.1 (`charts.py` já
    existe) e reaproveita `pricing/black_scholes.py`/`strategy/exit_timing_bs.py` da
    seção 3, que já existem mas nunca foram chamados fora da tela "Modelos". Ordem
    interna sugerida: 11.1 (preço/Greeks, mais simples, só cálculo) → 11.2 (dias até
    vender, reaproveita função já pronta) → 11.4 (definir confiança com rigor, trava
    o vocabulário antes de montar a tela) → 11.3 (tendência, mais cálculo novo) → 11.5
    (juntar tudo na UI, por último, depois que todo o conteúdo já existe).
