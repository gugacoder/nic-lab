# -*- coding: utf-8 -*-
"""
Utilitários para integração com Qdrant
Biblioteca ultrassimples - apenas algoritmos puros
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def store_embeddings_in_qdrant(embeddings_data: List[Dict[str, Any]], 
                              qdrant_url: str, collection_name: str,
                              api_key: str, batch_size: int = 100) -> Dict[str, Any]:
    """
    Armazena embeddings no Qdrant
    Função pura - confia nos parâmetros do notebook
    """
    
    print(f"💾 Armazenando embeddings no Qdrant")
    print(f"🔗 URL: {qdrant_url}")
    print(f"📦 Collection: {collection_name}")
    print(f"📊 Registros: {len(embeddings_data)}")
    print(f"🔢 Batch size: {batch_size}")
    
    # Conectar ao Qdrant
    client = QdrantClient(url=qdrant_url, api_key=api_key)
    
    stored_records = []
    failed_records = []
    
    # Processar em lotes
    for i in range(0, len(embeddings_data), batch_size):
        batch = embeddings_data[i:i + batch_size]
        
        print(f"🔄 Processando lote {i//batch_size + 1}: {len(batch)} registros")
        
        try:
            # Preparar pontos para inserção
            points = []
            for record in batch:
                point = PointStruct(
                    id=record.get("id", str(uuid.uuid4())),
                    vector=record["vector"],
                    payload=record.get("payload", {})
                )
                points.append(point)
            
            # Inserir no Qdrant
            result = client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Registrar sucessos
            for record in batch:
                stored_records.append({
                    "id": record.get("id", "unknown"),
                    "vector_size": len(record["vector"]),
                    "payload_keys": list(record.get("payload", {}).keys())
                })
            
        except Exception as e:
            # Registrar falhas
            for record in batch:
                failed_records.append({
                    "record_id": record.get("id", "unknown"),
                    "error": str(e)
                })
            print(f"❌ Erro no lote: {str(e)}")
    
    print(f"✅ Armazenados: {len(stored_records)}")
    if failed_records:
        print(f"❌ Falharam: {len(failed_records)}")
    
    return {
        "total_records": len(embeddings_data),
        "successfully_stored": len(stored_records),
        "failed_records": len(failed_records),
        "stored_ids": [record["id"] for record in stored_records],
        "failed_items": failed_records,
        "collection_name": collection_name,
        "timestamp": datetime.now().isoformat()
    }


def create_qdrant_collection(qdrant_url: str, collection_name: str, 
                           api_key: str, vector_size: int = 1024) -> bool:
    """
    Cria collection no Qdrant se não existir
    Função simples - apenas cria
    """
    
    print(f"📦 Criando collection: {collection_name}")
    print(f"📐 Dimensões do vetor: {vector_size}")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        
        # Verificar se collection já existe
        try:
            collection_info = client.get_collection(collection_name)
            print(f"✅ Collection '{collection_name}' já existe")
            return True
        except:
            # Collection não existe, criar nova
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"✅ Collection '{collection_name}' criada com sucesso")
            return True
            
    except Exception as e:
        print(f"❌ Erro ao criar collection: {str(e)}")
        return False


def validate_qdrant_connection(qdrant_url: str, api_key: str) -> bool:
    """
    Valida conexão com Qdrant
    Função pura - detecta automaticamente protocolo HTTP/HTTPS
    """
    
    print(f"🔍 Validando conexão com Qdrant: {qdrant_url}")
    
    if not api_key:
        print("❌ API Key não fornecida")
        return False
    
    # Detectar protocolo automaticamente da URL
    is_https = qdrant_url.lower().startswith('https://')
    protocol = "HTTPS" if is_https else "HTTP"
    
    print(f"🔗 Protocolo detectado: {protocol}")
    
    try:
        client = QdrantClient(
            url=qdrant_url, 
            api_key=api_key,
            prefer_grpc=False,
            https=is_https,
            timeout=30
        )
        # Testar conexão listando collections
        collections = client.get_collections()
        print(f"✅ Conexão validada ({protocol}) - {len(collections.collections)} collections encontradas")
        return True
        
    except Exception as e:
        print(f"❌ Falha na validação: {str(e)}")
        return False


def search_similar_vectors(qdrant_url: str, collection_name: str, api_key: str,
                          query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Busca vetores similares no Qdrant
    Função simples - retorna resultados reais
    """
    
    print(f"🔍 Buscando vetores similares")
    print(f"📦 Collection: {collection_name}")
    print(f"📊 Limite: {limit}")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        
        print(f"✅ Encontrados {len(results)} resultados")
        return results
        
    except Exception as e:
        print(f"❌ Erro na busca: {str(e)}")
        return []


def get_collection_info(qdrant_url: str, collection_name: str, api_key: str) -> Dict[str, Any]:
    """
    Obtém informações da collection
    Função simples - retorna estatísticas reais
    """
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        collection_info = client.get_collection(collection_name)
        
        info = {
            "collection_name": collection_name,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "points_count": collection_info.points_count,
            "segments_count": collection_info.segments_count,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value
            }
        }
        
        print(f"📊 Collection '{collection_name}':")
        print(f"   📄 Vetores: {info['vectors_count']}")
        print(f"   📊 Pontos: {info['points_count']}")
        
        return info
        
    except Exception as e:
        print(f"❌ Erro ao obter informações: {str(e)}")
        return {}


def batch_store_from_files(embeddings_files: List[str], qdrant_url: str,
                          collection_name: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Armazena embeddings de múltiplos arquivos no Qdrant
    Função que combina leitura + armazenamento
    """
    
    results = []
    
    for embeddings_file in embeddings_files:
        try:
            print(f"📄 Processando: {Path(embeddings_file).name}")
            
            # Carregar embeddings
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get("chunks_with_embeddings", [])
            
            if not chunks:
                results.append({
                    "file": embeddings_file,
                    "status": "skipped",
                    "reason": "Nenhum chunk com embedding encontrado"
                })
                continue
            
            # Converter para formato Qdrant
            qdrant_records = []
            for chunk in chunks:
                record = {
                    "id": f"{Path(embeddings_file).stem}_{chunk['chunk_id']}",
                    "vector": chunk["embedding"],
                    "payload": {
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "source_file": embeddings_file,
                        "chunk_id": chunk["chunk_id"],
                        "embedding_model": chunk.get("embedding_model", "unknown")
                    }
                }
                qdrant_records.append(record)
            
            # Armazenar no Qdrant
            storage_result = store_embeddings_in_qdrant(
                qdrant_records, qdrant_url, collection_name, api_key
            )
            
            results.append({
                "file": embeddings_file,
                "status": "success",
                "records_stored": storage_result["successfully_stored"],
                "records_failed": storage_result["failed_records"]
            })
            
            print(f"✅ {storage_result['successfully_stored']} registros armazenados")
            
        except Exception as e:
            print(f"❌ Erro em {Path(embeddings_file).name}: {e}")
            results.append({
                "file": embeddings_file,
                "status": "failed",
                "error": str(e)
            })
    
    return results


def cleanup_collection(qdrant_url: str, collection_name: str, api_key: str) -> bool:
    """
    Limpa todos os registros de uma collection
    Função simples - remove tudo
    """
    
    print(f"🧹 Limpando collection: {collection_name}")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        
        # Deletar e recriar collection (mais eficiente que deletar pontos individualmente)
        collection_info = client.get_collection(collection_name)
        vector_size = collection_info.config.params.vectors.size
        
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        print("✅ Collection limpa e recriada")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao limpar collection: {str(e)}")
        return False