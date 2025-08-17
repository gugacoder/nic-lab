# -*- coding: utf-8 -*-
"""
Utilitários para segmentação de texto (chunking)
Biblioteca ultrassimples - apenas algoritmos puros
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap_size: int = 100,
                          respect_sentences: bool = True) -> List[Dict[str, Any]]:
    """
    Divide texto em chunks com sobreposição
    Função pura - confia nos parâmetros do notebook
    """
    
    print(f"✂️ Segmentando texto em chunks")
    print(f"📏 Tamanho do chunk: {chunk_size} tokens")
    print(f"🔗 Overlap: {overlap_size} tokens")
    print(f"📝 Respeitar frases: {respect_sentences}")
    
    # Tokenização simples (divisão por espaços e pontuação)
    tokens = _simple_tokenize(text)
    
    if len(tokens) <= chunk_size:
        # Texto cabe em um chunk só
        return [{
            "chunk_id": 1,
            "text": text,
            "token_count": len(tokens),
            "start_position": 0,
            "end_position": len(text),
            "has_overlap": False
        }]
    
    chunks = []
    current_position = 0
    chunk_id = 1
    
    while current_position < len(tokens):
        # Determinar fim do chunk
        end_position = min(current_position + chunk_size, len(tokens))
        
        # Extrair tokens do chunk
        chunk_tokens = tokens[current_position:end_position]
        chunk_text = _tokens_to_text(chunk_tokens, text)
        
        # Ajustar para respeitar frases se solicitado
        if respect_sentences and end_position < len(tokens):
            chunk_text = _adjust_chunk_to_sentence_boundary(chunk_text)
        
        # Recalcular contagem de tokens do chunk ajustado
        adjusted_tokens = _simple_tokenize(chunk_text)
        
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text.strip(),
            "token_count": len(adjusted_tokens),
            "start_position": current_position,
            "end_position": current_position + len(adjusted_tokens),
            "has_overlap": chunk_id > 1
        })
        
        print(f"📄 Chunk {chunk_id}: {len(adjusted_tokens)} tokens")
        
        # Avançar posição com overlap
        current_position = current_position + chunk_size - overlap_size
        chunk_id += 1
        
        # Evitar loop infinito
        if current_position >= len(tokens) - overlap_size:
            break
    
    print(f"✅ Texto segmentado em {len(chunks)} chunks")
    return chunks


def _simple_tokenize(text: str) -> List[str]:
    """Tokenização simples por espaços e pontuação"""
    # Remove múltiplos espaços e quebras de linha
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Divide por espaços
    tokens = text.split()
    
    return tokens


def _tokens_to_text(tokens: List[str], original_text: str) -> str:
    """Converte tokens de volta para texto"""
    # Reconstrução simples - join com espaços
    return ' '.join(tokens)


def _adjust_chunk_to_sentence_boundary(chunk_text: str) -> str:
    """Ajusta chunk para terminar em fim de frase"""
    
    # Procurar último ponto final, exclamação ou interrogação
    sentence_endings = ['.', '!', '?', ':', ';']
    
    for i in range(len(chunk_text) - 1, -1, -1):
        if chunk_text[i] in sentence_endings:
            # Verificar se há espaço ou fim de string depois
            if i == len(chunk_text) - 1 or chunk_text[i + 1].isspace():
                return chunk_text[:i + 1]
    
    # Se não encontrou fim de frase, retorna o chunk original
    return chunk_text


def process_document_chunks(document_path: str, output_dir: str,
                           chunk_size: int = 500, overlap_size: int = 100) -> Dict[str, Any]:
    """
    Processa um documento completo em chunks
    Função que combina leitura + chunking
    """
    
    doc_path = Path(document_path)
    
    print(f"📖 Processando documento: {doc_path.name}")
    
    # Ler texto do arquivo
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return {
            "error": f"Erro ao ler arquivo: {e}",
            "status": "failed"
        }
    
    # Gerar chunks
    chunks = split_text_into_chunks(text, chunk_size, overlap_size)
    
    # Salvar chunks
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename_stem = doc_path.stem
    chunks_file = output_path / f"{filename_stem}_chunks.json"
    
    chunk_data = {
        "source_document": str(document_path),
        "chunking_config": {
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "total_chunks": len(chunks)
        },
        "chunks": chunks,
        "statistics": {
            "original_length": len(text),
            "total_tokens": sum(chunk["token_count"] for chunk in chunks),
            "avg_chunk_size": sum(chunk["token_count"] for chunk in chunks) / len(chunks) if chunks else 0
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Chunks salvos: {chunks_file.name}")
    
    return {
        "source_document": str(document_path),
        "chunks_file": str(chunks_file),
        "chunk_count": len(chunks),
        "total_tokens": chunk_data["statistics"]["total_tokens"],
        "status": "success"
    }


def batch_process_documents_chunking(document_list: List[str], output_dir: str,
                                   chunk_size: int = 500, overlap_size: int = 100) -> List[Dict[str, Any]]:
    """
    Processa lista de documentos para chunking
    Função simples - itera sobre lista
    """
    
    results = []
    
    for doc_path in document_list:
        try:
            result = process_document_chunks(doc_path, output_dir, chunk_size, overlap_size)
            results.append(result)
            print(f"✅ Chunks criados para: {Path(doc_path).name}")
        except Exception as e:
            print(f"❌ Erro em {Path(doc_path).name}: {e}")
            results.append({
                "source_document": doc_path,
                "error": str(e),
                "status": "failed"
            })
    
    return results


def validate_chunks(chunks_dir: str) -> Dict[str, Any]:
    """
    Valida chunks gerados
    Função simples - estatísticas básicas
    """
    
    chunks_path = Path(chunks_dir)
    
    if not chunks_path.exists():
        return {"valid": False, "error": "Diretório não existe"}
    
    chunk_files = list(chunks_path.glob("*_chunks.json"))
    
    total_chunks = 0
    total_tokens = 0
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_chunks += len(data.get("chunks", []))
                total_tokens += data.get("statistics", {}).get("total_tokens", 0)
        except Exception as e:
            print(f"⚠️ Erro ao validar {chunk_file.name}: {e}")
    
    return {
        "valid": True,
        "chunk_files_count": len(chunk_files),
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": total_tokens / total_chunks if total_chunks > 0 else 0
    }