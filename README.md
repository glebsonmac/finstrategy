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

- **Ações** — busca ou cola tickers (qualquer ação do Yahoo Finance) para adicionar à
  watchlist; o sistema busca preço, fundamentalista e elegibilidade de derivativo sozinho.
- **Configuração** — edita `AnalysisSettings` (prazos, limites de perda, parâmetros do
  Monte Carlo e das DQNs) sem tocar em código.
- **Rodar agora** — roda a análise completa (ação + put de proteção + DQN de entrada)
  para os tickers escolhidos e gera o PDF para download.
- **Histórico** — mostra o que já foi coletado em CSV (preço, fundamentalista, relatórios).
- **Coletor** — liga/desliga o loop de coleta contínua durante o pregão.
- **Modelos** — treina e avalia a DQN de entrada (honesto: treino/teste separados, com
  intervalo de confiança e p-valor contra benchmarks).

Primeira vez com a watchlist vazia: clique em "Popular com a semente do setor elétrico"
na tela Ações, ou rode:

```bash
PYTHONPATH=src python -c "from finstrategy.data.watchlist import ensure_seeded; ensure_seeded()"
```

### 4. Rodar o coletor contínuo sem UI (fica coletando durante o pregão)

```bash
PYTHONPATH=src python -m finstrategy.collector.scheduler
```

### 5. Rodar os testes

```bash
pytest
```

## Estrutura

```
src/finstrategy/
  data/        # Yahoo (e, futuramente, MetaTrader5), busca de ticker, watchlist, fundamentalista
  pricing/     # Black-Scholes e Monte Carlo (GARCH-t)
  strategy/    # put de proteção e timing de saída via Black-Scholes
  rl/          # DQN de entrada (ação) e de saída (derivativo)
  collector/   # calendário de pregão, CSV, loop de coleta contínua
  reporting/   # painel e relatório PDF
  frontend/    # app Streamlit
notebooks/     # notebook original, mantido como laboratório/validação
data_store/    # CSV gerado em runtime (gitignored, exceto .gitkeep)
config/        # settings.example.yaml e tickers_setor_eletrico.yaml são versionados;
               # settings.yaml e watchlist.yaml são gerados em runtime (gitignored)
```
