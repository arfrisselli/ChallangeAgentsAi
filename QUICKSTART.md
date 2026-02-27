# ChallangeAgentsAi - Quick Start

## TL;DR - 3 passos para rodar

```bash
# 1. Configure API keys
cp .env.example .env
# Edite .env e preencha: OPENAI_API_KEY, TAVILY_API_KEY, OPENWEATHERMAP_API_KEY

# 2. Suba tudo com Docker
docker compose up --build

# 3. Acesse
# http://localhost:8501
```

---

## O que foi implementado

✅ **Multi-agentes com LangGraph**
- UserNode, PlannerNode, ExecutorNode, MemoryNode, FallbackSearchNode
- Edge condicional: fallback para web search quando não há contexto

✅ **4 Tools evocadas pelo modelo** (sem botões)
- `web_search` - Busca web via Tavily
- `search_docs` - RAG com Chroma
- `sql_db` - PostgreSQL read-only com sqlglot
- `weather_api` - OpenWeatherMap com retry/backoff

✅ **Segurança SQL**
- Apenas SELECT permitido
- Queries parametrizadas
- Validação com sqlglot (rejeita DDL/DML)

✅ **Streaming em tempo real**
- Tokens chegam progressivamente na UI

✅ **Docker First**
- Postgres, Chroma e App sobem com `docker compose up`
- Ingest e init_db automáticos na inicialização

✅ **Testes**
- 18 unit tests (web_search, vector_search, sql_db, weather_api)
- 3 E2E tests (chat, chat/stream, health)
- 21/21 passando com Python 3.13.1

✅ **Documentação**
- [docs/architecture.md](docs/architecture.md) - Rationale técnico
- [docs/architecture.puml](docs/architecture.puml) - Diagrama PlantUML

---

## APIs necessárias (obtenha gratuitamente)

| API | URL | Uso |
|-----|-----|-----|
| OpenAI | https://platform.openai.com/api-keys | LLM + embeddings |
| Tavily | https://tavily.com/ | Busca web |
| OpenWeatherMap | https://openweathermap.org/api | Previsão do tempo |

---

## Perguntas de exemplo para testar

Após acessar http://localhost:8501:

1. **Web Search**: "Quais as últimas notícias sobre inteligência artificial?"
2. **RAG**: "O que é LangGraph?" (busca nos docs em `data/`)
3. **SQL**: "Quantos produtos temos no banco?"
4. **Weather**: "Qual a temperatura em São Paulo?"
5. **Multi-tool**: "Consulte o clima em Londres e me diga se há produtos no banco"

---

## Troubleshooting

**"Connection refused" ao subir**: 
- Certifique-se que as portas 8501, 8000, 5432 e 8000 estão livres

**"API key not found"**:
- Verifique se preencheu o `.env` corretamente
- Reinicie os containers: `docker compose down && docker compose up`

**"No documents found"**:
- Execute: `docker compose exec app python -m vector_db.ingest`

**"Table products does not exist"**:
- Execute: `docker compose exec app python scripts/init_db.py`

---

## Estrutura de pastas

```
ChallangeAgentsAi/
├── app/              # FastAPI backend
├── agents/           # Definições de agentes
├── graph/            # LangGraph: state, nodes, edges
├── tools/            # web_search, search_docs, sql_db, weather_api
├── vector_db/        # Chroma config + ingest.py
├── ui/               # Streamlit frontend
├── tests/            # Unit + E2E tests
├── data/             # Documentos para RAG
├── docs/             # Documentação técnica
└── scripts/          # init_db.py
```

---

## Próximos passos (opcional)

1. Adicionar mais documentos em `data/` e rodar ingest
2. Criar tabelas customizadas no Postgres para queries mais interessantes
3. Gerar PNG do diagrama PlantUML para o README
4. Adicionar dataset de avaliação automatizada (bônus)

---

**Dúvidas?** Consulte [README.md](README.md) ou [docs/architecture.md](docs/architecture.md)
