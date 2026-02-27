### 1. Propósito do desafio

Desenvolver **um mini‑assistente conversacional** (RAG‑based) que orquestra múltiplas _tools_ via **LangChain + LangGraph** e oferece **interface gráfica** para o usuário final.  
A entrega será avaliada pelo nosso time de engenharia durante uma entrevista técnica.

---

### 2. Requisitos funcionais (obrigatórios)

| Cenário               | O agente deve ser capaz de…                                                                               | Sugestões de implementação                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Busca na Web**      | Receber perguntas no chat, executar buscas no Google e retornar informações consolidadas com links‑fonte. | SerpAPI, Tavily ou Google CSE wrapper do LangChain       |
| **Base vetorial**     | Fazer _similarity search_ em embeddings previamente carregados (docs ou FAQs).                            | Chroma, Qdrant ou Azure AI Search (se optar por Foundry) |
| **PostgreSQL**        | Executar queries parametrizadas e retornar resultados formatados.                                         | `SQLDatabaseToolkit` do LangChain                        |
| **Previsão do tempo** | Consultar uma API pública (ex. OpenWeatherMap) e responder em linguagem natural.                          | HTTP tool via LangChain                                  |

Todas as funções devem ser **evocadas pelo modelo** conforme o _prompt_ do usuário, sem botões dedicados.
As funcionalidades devem ser expressas em multi-agents!

---

### 3. Requisitos não‑funcionais (obrigatórios)

1. **Organização do código**

```BASH
/app
/agents
/graph # nós & edges do LangGraph
/tools
/vector_db
/ui # frontend
/tests
docker-compose.yml
README.md
```

2. **Interface gráfica** – escolha livre (ex.: Streamlit, React + FastAPI, Gradio).

- Chats em tempo real (streaming).
- Logs de debug em painel lateral ou console.

3. **Docker first** – `docker compose up` precisa trazer tudo (Postgres, vetorDB, serviço da app).
4. **LangGraph** obrigatório **caso não adote Azure Foundry**.

---

### 4. Pontos extra (bonus)

| Tema                       | Exemplo prático que vale pontos                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Azure Foundry**          | Provisionar agentes como workflows do Foundry; usar Azure AI Search como vetorDB e Azure SQL.                             |
| **Avaliação automatizada** | Testes de _prompt‑eval_ via `langchain-bench`, `ragas` ou `LangSmith` (datasets simples simulando perguntas).             |
| **Arquitetura**            | PlantUML ou C4 diagram no repo (`docs/architecture.puml` + PNG gerado).                                                   |
| **Rationale técnico**      | README explicando escolhas de libs, trade‑offs, custos, segurança (rate‑limit da API do tempo, sanitização de SQL, etc.). |

---

### 5. Diretrizes de implementação

- **Agents**
- _ReAct_ ou _Plan‑and‑Execute_ pattern.
- Cada _tool_ em isolamento: entrada dataclass, saída tipada.
- **LangGraph**
- Nós: `UserNode`, `PlannerNode`, `ExecutorNode`, `MemoryNode`.
- Edge de fallback → `SearchTool` para perguntas sem contexto.
- **Persistência vetorial**
- Embeddings: `OpenAIEmbeddings` ou `azure_openai` caso Foundry.
- Alimentar via script `python ingest.py`.
- **Segurança**
- Variáveis sensíveis (.env) montadas no container.
- Limitar queries SQL (whitelist ou `sqlglot` parse‑safe).
- **Testes**
- Unitário por tool (`pytest`).
- _E2E_ ≥ 2 casos cobrindo fluxo completo (pergunta → resposta).
- **Observabilidade**
- Logs estruturados (JSON) + simples tracer (`LangChainTracer` ou OpenTelemetry).

---

### 6. Exemplo de `docker-compose.yml` (mínimo)

```yaml
version: "3.9"
services:
app:
  build: .
  env_file: .env
  ports: ["8501:8501"] # Streamlit
  depends_on: [postgres, chroma]

postgres:
  image: postgres:16
  environment:
    POSTGRES_PASSWORD: postgres
  volumes: ["pgdata:/var/lib/postgresql/data"]

chroma:
  image: chromadb/chroma
  ports: ["8000:8000"]

volumes:
pgdata:
```

---

### 7. Como rodar

**Passo 1: Clone o repositório**

```bash
git clone <repo-url>
cd ChallangeAgentsAi
```

**Passo 2: Configure as variáveis de ambiente**

```bash
# Copie o template
cp .env.example .env

# Edite .env e preencha as API keys obrigatórias:
# - OPENAI_API_KEY (https://platform.openai.com/api-keys)
# - TAVILY_API_KEY (https://tavily.com/)
# - OPENWEATHERMAP_API_KEY (https://openweathermap.org/api)
```

Ou use o script helper:

```bash
./setup.sh
```

**Passo 3: Execute com Docker**

```bash
docker compose up --build
```

**URLs importantes**:

- **UI (Streamlit)**: http://localhost:8501 ← **ACESSE AQUI**
- **API (FastAPI)**: http://localhost:8000
- **Health check**: http://localhost:8000/health
- **Readiness check**: http://localhost:8000/health/ready

**O que acontece automaticamente na inicialização**:

1. Postgres é criado com healthcheck
2. Chroma é inicializado
3. Tabela `products` é criada no Postgres (para demo SQL)
4. Documentos em `data/` são ingeridos no Chroma (RAG)
5. FastAPI inicia na porta 8000
6. Streamlit inicia na porta 8501

**Comandos úteis**:

```bash
# Ver logs
docker compose logs -f app

# Reingerir documentos manualmente
docker compose exec app python -m vector_db.ingest

# Recriar tabela SQL
docker compose exec app python scripts/init_db.py

# Parar tudo
docker compose down

# Limpar volumes (reset completo)
docker compose down -v
```

---

### 8. Como enviar

Repositório Git público ou privado com acesso ao avaliador.

A aplicação deve estar funcional com `docker compose up --build` após preencher o `.env`.

**URL da UI**: http://localhost:8501

---

### 9. Rationale técnico

Escolhas de libs, trade-offs, custos e estratégias de segurança (SQL, secrets, rate-limit, memória e fallback) estão documentados em **[docs/architecture.md](docs/architecture.md)**.
