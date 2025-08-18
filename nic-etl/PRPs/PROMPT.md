● 📚 Plano para Notebook de RAG com Pesquisa Híbrida

Baseado na análise do projeto, aqui está o plano detalhado para o
notebook etl-7-rag-retrieval.ipynb:

🎯 Arquitetura do Sistema RAG

1. Endpoint REST API (GET /api/v1/search)

# Parâmetros da requisição:
- query: string (texto da pesquisa)
- top_k: int (número de resultados, default: 5)
- score_threshold: float (threshold mínimo de similaridade,
default: 0.7)
- filters: dict (filtros de metadata opcionais)

2. Pipeline de Processamento

Query → Embedding → Pesquisa Híbrida → Reranking → Formatação →
Response

📋 Estrutura do Notebook

Célula 1: Documentação e Overview

# 🔍 RAG RETRIEVAL - Sistema de Busca Semântica

## 📋 O que este notebook faz
Sistema de retrieval que processa queries e retorna documentos
relevantes do Qdrant usando pesquisa híbrida (vetorial +
metadata).

## 🎯 Endpoints disponíveis
- `GET /api/v1/search` - Busca semântica híbrida
- `GET /api/v1/search/similar` - Busca por similaridade
- `GET /api/v1/search/metadata` - Busca por metadata

Célula 2: Configuração e Imports

# Bibliotecas necessárias
- qdrant_client
- sentence_transformers (BAAI/bge-m3)
- json, os, pathlib
- numpy para operações vetoriais

# Configurações do ambiente
- QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
- EMBEDDING_MODEL = "BAAI/bge-m3"
- Parâmetros de busca (top_k, score_threshold)

Célula 3: Conexão com Qdrant

# Inicializar cliente Qdrant
# Verificar collection existe
# Obter estatísticas da collection

Célula 4: Modelo de Embeddings

# Carregar mesmo modelo usado no pipeline (BAAI/bge-m3)
# Cache do modelo para performance
# Função de geração de embeddings normalizada

Célula 5: Endpoint Principal - Busca Híbrida

# GET /api/v1/search
def hybrid_search(query, top_k=5, filters=None):
    # 1. Gerar embedding da query
    query_embedding = model.encode(query, normalize=True)

    # 2. Construir filtros de metadata
    search_filters = build_filters(filters)

    # 3. Pesquisa vetorial no Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=search_filters,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    # 4. Formatar resposta
    return format_search_results(results)

Célula 6: Filtros de Metadata

def build_filters(filter_params):
    """
    Filtros disponíveis:
    - repo: string (repositório específico)
    - branch: string (branch específica)
    - relpath: string (caminho do arquivo)
    - lang: string (idioma do documento)
    - date_range: dict (from/to dates)
    - embed_model_major: string (versão do modelo)
    """

Célula 7: Reranking e Scoring

def rerank_results(results, query):
    """
    Estratégias de reranking:
    1. Score de similaridade vetorial (cosine)
    2. Boost por metadata relevante
    3. Recência do documento (last_updated)
    4. Frequência de termos no texto
    """

Célula 8: Formatação de Resposta

def format_search_results(results):
    """
    Resposta estruturada:
    {
        "query": "texto original",
        "total_results": N,
        "results": [
            {
                "score": 0.95,
                "text": "chunk de texto",
                "metadata": {
                    "source_document": "arquivo.md",
                    "chunk_index": 2,
                    "repo": "nic/documentacao",
                    "commit": "abc123"
                },
                "highlights": ["texto relevante destacado"]
            }
        ],
        "search_metadata": {
            "model": "BAAI/bge-m3",
            "collection": "nic",
            "filters_applied": {...}
        }
    }
    """

Célula 9: Endpoint de Similaridade

# GET /api/v1/search/similar
def find_similar(document_id, top_k=5):
    """
    Busca documentos similares a um documento existente
    """

Célula 10: Endpoint de Metadata

# GET /api/v1/search/metadata
def search_by_metadata(filters):
    """
    Busca apenas por filtros de metadata (sem vetores)
    Útil para listar documentos de um repo/branch específico
    """

Célula 11: Cache e Performance

# Sistema de cache para queries frequentes
# Pool de conexões com Qdrant
# Batch processing para múltiplas queries

Célula 12: Monitoramento e Métricas

# Logging de queries e performance
# Métricas de uso (queries/segundo, latência)
# Top queries mais frequentes

🔧 Funcionalidades Avançadas

1. Pesquisa Híbrida Verdadeira:
- Combinar busca vetorial com busca por keywords
- Score fusion entre diferentes métodos
- Pesos configuráveis para cada tipo de busca
2. Context Window:
- Retornar chunks adjacentes para contexto
- Merge de chunks contíguos
- Expansão dinâmica baseada em relevância
3. Query Enhancement:
- Expansão de query com sinônimos
- Correção ortográfica
- Identificação de entidades
4. Filtros Inteligentes:
- Auto-detecção de filtros na query
- Sugestão de filtros baseada em resultados
- Faceted search

📊 Integração com Pipeline ETL

- Usar mesmas configurações do pipeline (modelo, collection)
- Compatibilidade com metadados inseridos
- Versionamento consistente
- Suporte a múltiplas collections/ambientes

🚀 Deploy e Uso

1. Via Jupyter Kernel Gateway:
- Adicionar endpoints ao rest-api.ipynb
- Ou criar notebook separado rag-api.ipynb
2. Testes Incluídos:
- Queries de exemplo
- Validação de resultados
- Benchmarks de performance
3. Documentação OpenAPI:
- Schemas detalhados
- Exemplos de requisição/resposta
- Integração com /api/v1

📈 Métricas de Sucesso

- Latência < 200ms para queries simples
- Precisão > 90% para queries conhecidas
- Suporte a 100+ queries simultâneas
- Cache hit rate > 60%

Este plano cria um sistema RAG completo que se integra
perfeitamente com o pipeline ETL existente, mantendo consistência
de arquitetura e padrões do projeto.