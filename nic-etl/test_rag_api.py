#!/usr/bin/env python3
"""
Script de teste para o notebook RAG API
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

# Configurações
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://qdrant.codrstudio.dev:6333")
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY", "")
os.environ["QDRANT_COLLECTION"] = os.getenv("QDRANT_COLLECTION", "nic")
os.environ["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

print("🧪 Testando funcionalidades do RAG API\n")
print("=" * 60)

# Teste 1: Imports
print("1️⃣ Testando imports...")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    from sentence_transformers import SentenceTransformer
    import numpy as np
    print("✅ Todos os imports funcionando\n")
except ImportError as e:
    print(f"❌ Erro de import: {e}\n")
    sys.exit(1)

# Teste 2: Conexão Qdrant
print("2️⃣ Testando conexão com Qdrant...")
try:
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    collection_info = client.get_collection(os.environ["QDRANT_COLLECTION"])
    print(f"✅ Conectado ao Qdrant")
    print(f"   Collection: {os.environ['QDRANT_COLLECTION']}")
    print(f"   Pontos: {collection_info.points_count}")
    print(f"   Status: {collection_info.status}\n")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}\n")
    
# Teste 3: Modelo de embeddings
print("3️⃣ Testando modelo de embeddings...")
try:
    model = SentenceTransformer(os.environ["EMBEDDING_MODEL"])
    test_embedding = model.encode("teste", normalize_embeddings=True)
    print(f"✅ Modelo carregado: {os.environ['EMBEDDING_MODEL']}")
    print(f"   Dimensões: {len(test_embedding)}")
    print(f"   Magnitude: {np.linalg.norm(test_embedding):.3f}\n")
except Exception as e:
    print(f"❌ Erro no modelo: {e}\n")

# Teste 4: Busca simples
print("4️⃣ Testando busca semântica...")
try:
    query = "self checkout"
    query_embedding = model.encode(query, normalize_embeddings=True)
    
    results = client.search(
        collection_name=os.environ["QDRANT_COLLECTION"],
        query_vector=query_embedding.tolist(),
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    
    print(f"✅ Busca por '{query}':")
    print(f"   Resultados encontrados: {len(results)}")
    if results:
        print(f"   Melhor score: {results[0].score:.4f}")
        print(f"   Documento: {results[0].payload.get('source_document', 'N/A')}\n")
except Exception as e:
    print(f"❌ Erro na busca: {e}\n")

# Teste 5: Filtros de metadata
print("5️⃣ Testando filtros de metadata...")
try:
    filter_obj = Filter(
        must=[
            FieldCondition(
                key="branch",
                match=MatchValue(value="main")
            )
        ]
    )
    
    filtered_results = client.search(
        collection_name=os.environ["QDRANT_COLLECTION"],
        query_vector=query_embedding.tolist(),
        query_filter=filter_obj,
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    
    print(f"✅ Busca com filtro branch='main':")
    print(f"   Resultados: {len(filtered_results)}")
    if filtered_results:
        print(f"   Branch do primeiro: {filtered_results[0].payload.get('branch', 'N/A')}\n")
except Exception as e:
    print(f"❌ Erro nos filtros: {e}\n")

# Teste 6: Range filters (para datas)
print("6️⃣ Testando Range filters...")
try:
    # Criar um filtro Range simples
    range_filter = Filter(
        must=[
            FieldCondition(
                key="chunk_index",
                range=Range(gte=0, lte=10)
            )
        ]
    )
    
    range_results = client.search(
        collection_name=os.environ["QDRANT_COLLECTION"],
        query_vector=query_embedding.tolist(),
        query_filter=range_filter,
        limit=3,
        with_payload=True,
        with_vectors=False
    )
    
    print(f"✅ Busca com Range filter (chunk_index 0-10):")
    print(f"   Resultados: {len(range_results)}")
    if range_results:
        indices = [r.payload.get('chunk_index', -1) for r in range_results]
        print(f"   Chunk indices: {indices}\n")
except Exception as e:
    print(f"❌ Erro no Range filter: {e}\n")

print("=" * 60)
print("📊 Resumo dos testes:")
print("   - Imports: ✅")
print(f"   - Qdrant: {'✅' if 'client' in locals() else '❌'}")
print(f"   - Embeddings: {'✅' if 'model' in locals() else '❌'}")
print(f"   - Busca: {'✅' if 'results' in locals() else '❌'}")
print(f"   - Filtros: {'✅' if 'filtered_results' in locals() else '❌'}")
print(f"   - Range: {'✅' if 'range_results' in locals() else '❌'}")
print("\n✨ Teste concluído!")