# RAG (Retrieval-Augmented Generation)

## Conceito

RAG combina recuperação de informação com geração de linguagem natural. Em vez de depender apenas do conhecimento do modelo (limitado e desatualizado), buscamos informações relevantes em uma base de dados antes de gerar a resposta.

## Fluxo RAG

1. **Ingest**: Documentos são divididos em chunks e convertidos em embeddings (vetores numéricos)
2. **Store**: Embeddings são armazenados em um vector database (Chroma, Pinecone, etc.)
3. **Query**: Pergunta do usuário é convertida em embedding
4. **Retrieve**: Busca por similaridade retorna chunks mais relevantes
5. **Generate**: LLM usa os chunks como contexto para gerar resposta

## Quando usar RAG?

- Documentação interna da empresa
- Base de conhecimento de produtos
- FAQs e políticas
- Informações que mudam frequentemente
- Dados proprietários ou confidenciais

## Vantagens

- **Atualização**: Adicionar novos docs sem retreinar o modelo
- **Precisão**: Respostas baseadas em fontes reais
- **Transparência**: Pode citar documentos de origem
- **Reduz alucinações**: Modelo tem contexto factual

## Limitações

- **Custo**: Embeddings + armazenamento + inferência
- **Latência**: Busca vetorial adiciona tempo
- **Qualidade**: Depende da qualidade dos documentos originais
