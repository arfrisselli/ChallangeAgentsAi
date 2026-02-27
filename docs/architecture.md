# Arquitetura e Rationale Técnico

## Componentes

- **UI (Streamlit)**: Chat em tempo real com streaming; consome a API FastAPI. Porta 8501.
- **Backend (FastAPI)**: Rota `/chat` e `/chat/stream`, health checks, inicialização do grafo LangGraph. Porta 8000 (interno).
- **LangGraph**: Grafo com nós UserNode, PlannerNode, ConversationNode, WeatherNode, ExecutorNode, FallbackSearchNode e MemoryNode; roteamento condicional de 4 vias no Planner.
- **Persona (Atlas)**: Identidade consistente do assistente definida em `graph/persona.py`. Nome fixo ("Atlas"), respostas sociais multilíngues, detecção por regex (sem LLM).
- **Tools**: `web_search` (Tavily), `search_docs` (Chroma), `sql_db` (PostgreSQL read-only com sqlglot), `weather_api` (OpenWeatherMap).
- **PostgreSQL**: Dados de exemplo (tabela `products`); uso apenas pela tool SQL, com queries parametrizadas e validação.
- **Chroma**: Base vetorial para RAG; alimentada por `vector_db/ingest.py`.

## Fluxo principal

1. Usuário envia mensagem no chat (Streamlit).
2. Streamlit chama `POST /chat/stream` (ou `/chat`) na FastAPI.
3. FastAPI invoca o grafo LangGraph com a mensagem.
4. **UserNode** repassa o state.
5. **PlannerNode** classifica a pergunta em 4 rotas:
   - **Conversacional** (saudações, identidade, capacidades) → `ConversationNode`
   - **Clima** (detectado por keywords: clima, weather, etc.) → `WeatherNode`
   - **Web Fallback** (conhecimento geral, notícias) → `FallbackSearchNode`
   - **Execução** (docs, SQL, múltiplas tools) → `ExecutorNode`
6. **ConversationNode**: Responde perguntas sociais com identidade fixa (Atlas). Sem LLM.
7. **WeatherNode**: Extrai cidade do texto e chama OpenWeatherMap diretamente. Sem LLM.
8. **FallbackSearchNode**: Chama Tavily web search e sintetiza resultado com LLM.
9. **ExecutorNode**: Roda agente ReAct com todas as 4 tools até resposta final.
10. **MemoryNode** atualiza contexto (número de mensagens).
11. Resposta é enviada ao cliente (streaming NDJSON ou JSON).

## Roteamento do PlannerNode

```
Pergunta → PlannerNode
  ├── Regex: saudação/identidade?  → ConversationNode (sem LLM)
  ├── Keywords: clima/weather?     → WeatherNode      (sem LLM)
  ├── LLM: conhecimento geral?    → FallbackSearchNode
  └── LLM: tools especializadas?  → ExecutorNode
```

Perguntas de conversação e clima são roteadas **sem chamar LLM**, economizando tokens e respondendo instantaneamente.

## Escolhas técnicas

- **LangGraph**: Orquestração explícita (nós e edges), roteamento condicional de 4 vias, separação Planner/Executor conforme diretrizes.
- **WeatherNode dedicado**: Extrai cidade via regex e chama API diretamente, sem custo de tokens OpenAI. Funciona mesmo com quota excedida.
- **ConversationNode**: Identidade fixa ("Atlas") garantida por padrões regex, sem variação de LLM. Respostas consistentes e multilíngues.
- **Streamlit**: UI rápida, integração simples com HTTP streaming; porta 8501 alinhada ao exemplo do desafio.
- **Chroma**: VetorDB local com docker-compose; embeddings OpenAI ou Azure configuráveis via .env.
- **FastAPI + NDJSON**: Streaming token-a-token para a UI sem depender de SSE específico; health/ready para Postgres e Chroma.
- **Tratamento de erros**: Rate limit da OpenAI (429) capturado com mensagem amigável ao usuário em vez de stack trace.

## Trade-offs e custos

- **PlannerNode**: Detecção por keywords (conversa, clima) evita chamadas LLM desnecessárias. Apenas perguntas ambíguas usam LLM para classificar.
- **WeatherNode vs ExecutorNode+weather_api**: WeatherNode é mais rápido e barato (0 tokens), mas não interpreta perguntas complexas como "compare o clima de SP e RJ". Nesse caso, o ExecutorNode usaria a tool weather_api via LLM.
- **ReAct no Executor**: Uma tool por vez; múltiplas chamadas de LLM em perguntas que exigem várias tools.
- **Embeddings**: Custo por documento no ingest e por query no RAG; controle via tamanho do corpus e top_k.

## Segurança

- **SQL**: Queries apenas parametrizadas (binds do driver); validação com **sqlglot** (apenas `Select`); rejeição de DDL/DML (DROP, TRUNCATE, UPDATE, DELETE etc.). Tentativas proibidas retornam mensagem clara e log em WARNING.
- **Prompt injection**: As tools validam input independentemente (SQL rejeita comando proibido; weather sanitiza cidade/país). Não confiam só no output do modelo.
- **Secrets**: Todas as chaves e DSNs vêm de `.env`; nunca logadas nem expostas em respostas.
- **Rate-limit (weather)**: Retry com backoff; mensagem amigável quando o limite é atingido.
- **Rate-limit (OpenAI)**: Capturado com `try/except openai.RateLimitError`, retorna mensagem clara ao usuário.

## Memória e fallback

- **Memória**: MemoryNode mantém contagem de mensagens no contexto (últimas 20); pode ser estendido para resumo ou persistência.
- **Fallback**: Perguntas classificadas como "só web" pelo Planner vão direto para a tool de busca web, evitando uso desnecessário de RAG/SQL/weather e reduzindo alucinações.

## Sistema de Identidade (Persona)

- **Nome**: Atlas (definido em `graph/persona.py`, nunca muda)
- **Detecção**: Regex patterns para PT e EN (saudações, nome, capacidades, agradecimentos)
- **Respostas**: Predefinidas, consistentes, multilíngues (responde no idioma da pergunta)
- **Zero custo**: Nenhuma chamada LLM para perguntas sociais
