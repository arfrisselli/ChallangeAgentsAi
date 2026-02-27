# ChallangeAgentsAi - Documentação Técnica

## Visão Geral

Este projeto implementa um assistente conversacional RAG-based com múltiplas ferramentas orquestradas via LangChain e LangGraph.

## Arquitetura

O sistema é composto por:

- **Backend FastAPI**: API REST com streaming de respostas
- **Frontend Streamlit**: Interface de chat em tempo real
- **LangGraph**: Orquestração de agentes (Planner, Executor, Memory)
- **Tools**: Web search (Tavily), Vector search (Chroma), SQL (PostgreSQL), Weather (OpenWeatherMap)

## Componentes

### Agentes (Multi-Agent)

1. **UserNode**: Normaliza entrada do usuário
2. **PlannerNode**: Decide estratégia (fallback web ou execução completa)
3. **ExecutorNode**: Executa tools via ReAct pattern
4. **MemoryNode**: Mantém contexto conversacional
5. **FallbackSearchNode**: Busca web quando não há contexto suficiente

### Tools

Todas as tools são evocadas automaticamente pelo modelo baseado no contexto da pergunta:

- **web_search**: Busca informações atuais na web (Tavily API)
- **search_docs**: Busca em documentos internos via RAG (Chroma)
- **sql_db**: Queries read-only no PostgreSQL (validação com sqlglot)
- **weather_api**: Previsão do tempo via OpenWeatherMap

## Segurança

- **SQL**: Apenas SELECT, queries parametrizadas, validação com sqlglot
- **Secrets**: Todas as chaves vêm de .env, nunca expostas
- **Prompt Injection**: Tools validam inputs independentemente do modelo
- **Rate Limiting**: Retry com backoff para APIs externas

## Performance

- **Streaming**: Tokens em tempo real para melhor UX
- **Fallback**: Planner reduz custos evitando tools desnecessárias
- **Caching**: State persistente via LangGraph checkpointer

## Custos Estimados

Por requisição (aproximado):
- Planner: 1 chamada LLM (~100 tokens)
- Executor: 2-5 chamadas LLM (~500-2000 tokens total)
- Embeddings RAG: ~0.0001 USD por query

Total: ~0.001-0.01 USD por conversa completa (usando gpt-4o-mini)
