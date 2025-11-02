## Recomendações

> 1. Revisar chunks das queries sem resultado ("self checkout
> pagamento", "NIC inteligência conhecimento")
> 2. Adicionar sinônimos no processamento de chunks
> 3. Implementar query expansion para termos relacionados
> 4. Cache mais agressivo para queries > 200ms

## Explicaões

Vou elaborar cada uma das 4 recomendações de forma prática e
específica para o projeto:

1. Revisar chunks das queries sem resultado

O que fazer:
- Investigar por que "self checkout pagamento" e "NIC
inteligência conhecimento" não retornaram resultados
- Verificar se esses termos existem nos documentos originais
- Analisar como os chunks foram segmentados

Ação prática:
# Vamos verificar se os termos existem na collection
def investigate_missing_terms():
    # Buscar por "self checkout" sem threshold
    results1 = client.query_points(
        collection_name="nic",
        query=embedding_for("self checkout"),
        limit=10,
        score_threshold=0.3  # Threshold mais baixo
    )

    # Buscar por palavras individuais nos payloads
    scroll_results = client.scroll(
        collection_name="nic",
        scroll_filter=Filter(
            should=[
                FieldCondition(key="text",
match=MatchText(text="checkout")),
                FieldCondition(key="text",
match=MatchText(text="pagamento"))
            ]
        )
    )

Possíveis problemas:
- Chunks muito pequenos que separaram "self" de "checkout"
- Termos em contextos diferentes (ex: "pagamento" pode estar
em chunk separado de "self checkout")
- Score threshold muito alto (0.7) filtrando resultados
relevantes

2. Adicionar sinônimos no processamento de chunks

Sinônimos específicos para o domínio NIC:
DOMAIN_SYNONYMS = {
    "checkout": ["caixa", "pdv", "ponto de venda",
"terminal"],
    "pagamento": ["cobrança", "faturamento", "transação",
"valor"],
    "cliente": ["usuário", "consumidor", "comprador"],
    "cupom": ["nota", "recibo", "comprovante"],
    "fiscal": ["operador", "funcionário", "vendedor"],
    "NIC": ["núcleo inteligência", "sistema conhecimento",
"base conhecimento"],
    "cancelamento": ["estorno", "reversão", "anulação"],
    "menu": ["interface", "tela", "painel"]
}

def expand_chunk_with_synonyms(chunk_text):
    expanded_text = chunk_text
    for term, synonyms in DOMAIN_SYNONYMS.items():
        if term.lower() in chunk_text.lower():
            # Adicionar sinônimos como metadata ou expandir 
texto
            expanded_text += f" {' '.join(synonyms)}"
    return expanded_text

Implementação no pipeline:
- Modificar o notebook etl-4-segmentacao-chunks.ipynb
- Adicionar sinônimos como metadata nos chunks
- Ou expandir o texto dos chunks com termos relacionados

3. Query Expansion (não "Kerec-Spanch")

Query Expansion = expandir a query com termos relacionados
antes da busca.

Implementação prática:
def expand_query(original_query):
    expanded_terms = []
    words = original_query.lower().split()

    for word in words:
        expanded_terms.append(word)
        # Adicionar sinônimos
        if word in DOMAIN_SYNONYMS:
            expanded_terms.extend(DOMAIN_SYNONYMS[word][:2])
# Máx 2 sinônimos

    # Gerar múltiplas variações da query
    variations = [
        original_query,
        " ".join(expanded_terms),
        " ".join(words)  # Query original sem sinônimos
    ]

    return variations

def hybrid_search_with_expansion(query, top_k=5):
    query_variations = expand_query(query)
    all_results = []

    for variation in query_variations:
        results = basic_search(variation, top_k)
        all_results.extend(results)

    # Deduplicar e reordenar por score
    unique_results = dedup_by_point_id(all_results)
    return sorted(unique_results, key=lambda x: x['score'],
reverse=True)[:top_k]

4. Cache mais agressivo nos notebooks

Sim, podemos implementar cache nos notebooks!

Opções de cache:

A) Cache em memória (atual):
# Cache atual - funciona mas é limitado
query_cache = {}  # Perdido quando notebook reinicia

B) Cache persistente em arquivo:
import pickle
from pathlib import Path

CACHE_FILE = Path("pipeline-data/cache/rag_cache.pkl")

def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache_dict):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_dict, f)

# Usar no notebook
persistent_cache = load_cache()

C) Cache Redis/SQLite (mais robusto):
import sqlite3
import json

def init_sqlite_cache():
    conn = sqlite3.connect('pipeline-data/cache/rag_cache.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            results TEXT,
            timestamp REAL,
            response_time REAL
        )
    ''')
    return conn

def get_cached_result(query_hash):
    conn = init_sqlite_cache()
    result = conn.execute(
        'SELECT results FROM query_cache WHERE query_hash = ? 
AND timestamp > ?',
        (query_hash, time.time() - CACHE_TTL)
    ).fetchone()
    return json.loads(result[0]) if result else None

Cache mais agressivo - implementação:

# No rag-api.ipynb
def aggressive_hybrid_search(query, top_k=5, 
score_threshold=0.7):
    # 1. Cache com TTL mais longo para queries lentas
    cache_key = f"{query}:{top_k}:{score_threshold}"

    # 2. Cache em níveis
    if cache_key in memory_cache:
        return add_cache_metadata(memory_cache[cache_key],
"memory")

    if cached_result := load_from_persistent_cache(cache_key):
        memory_cache[cache_key] = cached_result
        return add_cache_metadata(cached_result, "disk")

    # 3. Busca real
    start_time = time.time()
    result = perform_search(query, top_k, score_threshold)
    search_time = time.time() - start_time

    # 4. Cache agressivo para queries lentas
    if search_time > 0.2:  # 200ms
        save_to_persistent_cache(cache_key, result, ttl=3600)
# 1 hora
    else:
        save_to_persistent_cache(cache_key, result, ttl=300)
# 5 min

    memory_cache[cache_key] = result
    return result

Resumo das implementações:

1. Revisar chunks: Investigar com threshold baixo + busca por
palavras-chave
2. Sinônimos: Dicionário de termos do domínio NIC + expansão
de chunks
3. Query expansion: Múltiplas variações da query + merge de
resultados
4. Cache: SQLite persistente + cache em memória + TTL baseado
em performance