# -*- coding: utf-8 -*-
"""
Utilitários para geração de embeddings
Biblioteca ultrassimples - apenas algoritmos puros
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer


def generate_embeddings(texts: List[str], model_name: str = "BAAI/bge-m3",
                       batch_size: int = 32) -> List[List[float]]:
    """
    Gera embeddings para lista de textos
    Função pura - confia nos parâmetros do notebook
    """
    
    print(f"🧠 Gerando embeddings")
    print(f"🤖 Modelo: {model_name}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📝 Textos: {len(texts)}")
    
    # Carregar modelo
    model = SentenceTransformer(model_name)
    
    # Gerar embeddings em lotes
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        print(f"🔄 Processando lote: {i + 1}-{min(i + batch_size, len(texts))}/{len(texts)}")
        
        # Gerar embeddings para o lote
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
        embeddings.extend(batch_embeddings.tolist())
    
    print(f"✅ {len(embeddings)} embeddings gerados")
    return embeddings


def process_chunks_to_embeddings(chunks_file: str, output_dir: str,
                                model_name: str = "BAAI/bge-m3") -> Dict[str, Any]:
    """
    Processa arquivo de chunks para gerar embeddings
    Função que combina leitura + embedding generation
    """
    
    chunks_path = Path(chunks_file)
    
    print(f"🧠 Processando chunks: {chunks_path.name}")
    
    # Carregar chunks
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
    except Exception as e:
        return {
            "error": f"Erro ao ler chunks: {e}",
            "status": "failed"
        }
    
    chunks = chunks_data.get("chunks", [])
    
    if not chunks:
        return {
            "error": "Nenhum chunk encontrado",
            "status": "failed"
        }
    
    # Extrair textos dos chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # Gerar embeddings
    embeddings = generate_embeddings(texts, model_name)
    
    # Combinar chunks com embeddings
    enriched_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched_chunk = {
            **chunk,
            "embedding": embedding,
            "embedding_model": model_name,
            "embedding_dimensions": len(embedding)
        }
        enriched_chunks.append(enriched_chunk)
    
    # Salvar embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename_stem = chunks_path.stem.replace("_chunks", "")
    embeddings_file = output_path / f"{filename_stem}_embeddings.json"
    
    embeddings_data = {
        "source_chunks_file": str(chunks_file),
        "model_info": {
            "model_name": model_name,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "total_embeddings": len(embeddings)
        },
        "chunks_with_embeddings": enriched_chunks,
        "statistics": {
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
            "avg_text_length": sum(len(chunk["text"]) for chunk in chunks) / len(chunks) if chunks else 0,
            "embedding_size_mb": (len(embeddings) * len(embeddings[0]) * 4) / (1024*1024) if embeddings else 0  # float32
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Embeddings salvos: {embeddings_file.name}")
    
    return {
        "source_chunks_file": str(chunks_file),
        "embeddings_file": str(embeddings_file),
        "embedding_count": len(embeddings),
        "model_name": model_name,
        "status": "success"
    }


def batch_process_chunks_to_embeddings(chunks_files: List[str], output_dir: str,
                                     model_name: str = "BAAI/bge-m3") -> List[Dict[str, Any]]:
    """
    Processa lista de arquivos de chunks para embeddings
    Função simples - itera sobre lista
    """
    
    results = []
    
    for chunks_file in chunks_files:
        try:
            result = process_chunks_to_embeddings(chunks_file, output_dir, model_name)
            results.append(result)
            print(f"✅ Embeddings gerados para: {Path(chunks_file).name}")
        except Exception as e:
            print(f"❌ Erro em {Path(chunks_file).name}: {e}")
            results.append({
                "source_chunks_file": chunks_file,
                "error": str(e),
                "status": "failed"
            })
    
    return results


def validate_embeddings(embeddings_dir: str) -> Dict[str, Any]:
    """
    Valida embeddings gerados
    Função simples - estatísticas básicas
    """
    
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        return {"valid": False, "error": "Diretório não existe"}
    
    embedding_files = list(embeddings_path.glob("*_embeddings.json"))
    
    total_embeddings = 0
    total_dimensions = 0
    models_used = set()
    
    for embedding_file in embedding_files:
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = data.get("chunks_with_embeddings", [])
                total_embeddings += len(chunks)
                
                if chunks:
                    total_dimensions += chunks[0].get("embedding_dimensions", 0)
                    models_used.add(chunks[0].get("embedding_model", "unknown"))
                    
        except Exception as e:
            print(f"⚠️ Erro ao validar {embedding_file.name}: {e}")
    
    return {
        "valid": True,
        "embedding_files_count": len(embedding_files),
        "total_embeddings": total_embeddings,
        "avg_dimensions": total_dimensions / len(embedding_files) if embedding_files else 0,
        "models_used": list(models_used)
    }


def extract_embeddings_for_qdrant(embeddings_file: str) -> List[Dict[str, Any]]:
    """
    Extrai embeddings em formato adequado para Qdrant
    Função que converte formato interno para Qdrant
    """
    
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao ler embeddings: {e}")
        return []
    
    chunks = data.get("chunks_with_embeddings", [])
    
    qdrant_records = []
    for chunk in chunks:
        record = {
            "id": chunk["chunk_id"],
            "vector": chunk["embedding"],
            "payload": {
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "source_document": data.get("source_chunks_file", "unknown"),
                "chunk_position": chunk.get("start_position", 0),
                "has_overlap": chunk.get("has_overlap", False),
                "embedding_model": chunk.get("embedding_model", "unknown")
            }
        }
        qdrant_records.append(record)
    
    print(f"📦 Preparados {len(qdrant_records)} registros para Qdrant")
    return qdrant_records