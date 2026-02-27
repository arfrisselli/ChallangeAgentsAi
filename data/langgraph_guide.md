# LangGraph - Orquestração de Agentes

## O que é LangGraph?

LangGraph é um framework da LangChain para construir aplicações com múltiplos agentes usando grafos de estados. Permite orquestração complexa com controle fino sobre o fluxo de execução.

## Conceitos Principais

### State
Estado compartilhado entre nós, tipado com TypedDict ou Pydantic.

### Nodes
Funções que processam o estado e retornam atualizações. Cada nó representa uma etapa do workflow.

### Edges
Conexões entre nós. Podem ser:
- **Simples**: A → B (sempre segue)
- **Condicionais**: A → B ou C (baseado em lógica)

### Checkpointer
Persiste estado entre execuções, permitindo pausar/retomar workflows.

## Padrões Comuns

### ReAct (Reasoning + Acting)
Modelo decide quando usar ferramentas e raciocina sobre resultados.

### Plan-and-Execute
Separação entre planejamento (Planner) e execução (Executor).

### Multi-Agent
Vários agentes especializados colaborando em uma tarefa.

## Vantagens sobre Chains

- **Controle**: Fluxo explícito vs implícito
- **Debug**: Estado visível em cada etapa
- **Flexibilidade**: Loops, condicionais, fallbacks
- **Streaming**: Suporte nativo para respostas progressivas
