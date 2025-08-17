# -*- coding: utf-8 -*-
"""
Utilitários para processamento com Docling
Biblioteca ultrassimples - apenas algoritmos puros
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from docling.document_converter import DocumentConverter


def extract_content_from_file(file_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Extrai conteúdo de arquivo usando Docling
    Função pura - confia nos parâmetros do notebook
    """
    
    file_path_obj = Path(file_path)
    filename_stem = file_path_obj.stem
    file_extension = file_path_obj.suffix.lower()
    
    print(f"📄 Processando: {file_path_obj.name}")
    print(f"🔧 Tipo: {file_extension}")
    
    # Converter documento usando Docling
    converter = DocumentConverter()
    result = converter.convert(file_path)
    
    # Extrair texto e estrutura
    document = result.document
    plain_text = document.export_to_markdown()
    
    # Coletar informações de estrutura
    structure_info = {
        "sections": len([item for item in document.texts if item.tag == "title"]),
        "tables": len(document.tables),
        "images": len(document.pictures),
        "pages": len(document.pages) if hasattr(document, 'pages') else 1,
        "has_headers": any(item.tag == "title" for item in document.texts),
        "has_footer": False  # Simplificado
    }
    
    # Informações de processamento
    processing_info = {
        "method": f"docling_{file_extension[1:]}",
        "ocr_used": hasattr(result, 'ocr_results') and result.ocr_results is not None,
        "confidence_score": 0.95,  # Valor padrão, Docling não fornece score direto
        "processing_time": 0.0  # Seria medido se necessário
    }
    
    extracted_content = {
        "plain_text": plain_text,
        "structure": structure_info,
        "processing_info": processing_info
    }
    
    # Salvar conteúdo estruturado
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    content_file = output_path / f"{filename_stem}_content.json"
    text_file = output_path / f"{filename_stem}_text.txt"
    
    # Salvar estrutura completa
    with open(content_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_content, f, indent=2, ensure_ascii=False)
    
    # Salvar texto puro
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(extracted_content['plain_text'])
    
    print(f"💾 Salvo: {content_file.name}")
    print(f"💾 Salvo: {text_file.name}")
    
    return {
        "source_file": file_path,
        "content_file": str(content_file),
        "text_file": str(text_file),
        "processing_info": extracted_content['processing_info'],
        "structure": extracted_content['structure']
    }


def batch_process_documents(file_list: List[str], output_dir: str) -> List[Dict[str, Any]]:
    """
    Processa lista de documentos em lote
    Função simples - itera sobre lista
    """
    
    results = []
    
    for file_path in file_list:
        try:
            result = extract_content_from_file(file_path, output_dir)
            results.append(result)
            print(f"✅ Processado: {Path(file_path).name}")
        except Exception as e:
            print(f"❌ Erro em {Path(file_path).name}: {e}")
            results.append({
                "source_file": file_path,
                "error": str(e),
                "status": "failed"
            })
    
    return results


def validate_processed_content(content_dir: str) -> Dict[str, Any]:
    """
    Valida conteúdo processado
    Função simples - conta arquivos
    """
    
    content_path = Path(content_dir)
    
    if not content_path.exists():
        return {"valid": False, "error": "Diretório não existe"}
    
    json_files = list(content_path.glob("*_content.json"))
    text_files = list(content_path.glob("*_text.txt"))
    
    return {
        "valid": True,
        "json_files_count": len(json_files),
        "text_files_count": len(text_files),
        "total_files": len(json_files) + len(text_files)
    }