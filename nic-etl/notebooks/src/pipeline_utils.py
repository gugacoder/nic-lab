# -*- coding: utf-8 -*-
"""
Utilitários básicos para o pipeline NIC ETL
Biblioteca ultrassimples - confia 100% nos notebooks
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class PipelineState:
    """Gerenciador de estado compartilhado entre notebooks"""
    
    def __init__(self, base_dir: str = "./pipeline_data"):
        self.base_dir = Path(base_dir)
        self.metadata_dir = self.base_dir / "metadata"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        # Criar diretórios se não existirem
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def save_stage_data(self, stage_number: int, data: Dict[str, Any]) -> None:
        """Salva dados de uma etapa"""
        filename = f"stage_{stage_number:02d}_{self._get_stage_name(stage_number)}.json"
        filepath = self.metadata_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def load_stage_data(self, stage_number: int) -> Dict[str, Any]:
        """Carrega dados de uma etapa"""
        filename = f"stage_{stage_number:02d}_{self._get_stage_name(stage_number)}.json"
        filepath = self.metadata_dir / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def mark_stage_completed(self, stage_number: int) -> None:
        """Marca uma etapa como concluída"""
        lock_file = self.checkpoints_dir / f"stage_{stage_number:02d}_completed.lock"
        with open(lock_file, 'w') as f:
            f.write(f"Completed at: {datetime.now().isoformat()}")
    
    def is_stage_completed(self, stage_number: int) -> bool:
        """Verifica se uma etapa foi concluída"""
        lock_file = self.checkpoints_dir / f"stage_{stage_number:02d}_completed.lock"
        return lock_file.exists()
    
    def get_pipeline_progress(self) -> List[Dict[str, Any]]:
        """Retorna progresso de todas as etapas"""
        stages = []
        for i in range(1, 8):  # 7 etapas
            stages.append({
                "stage": i,
                "name": self._get_stage_name(i),
                "completed": self.is_stage_completed(i)
            })
        return stages
    
    def _get_stage_name(self, stage_number: int) -> str:
        """Retorna nome da etapa"""
        names = {
            1: "foundation",
            2: "gitlab", 
            3: "docling",
            4: "chunking",
            5: "embeddings", 
            6: "qdrant",
            7: "validation"
        }
        return names.get(stage_number, "unknown")


def check_prerequisites(current_stage: int) -> bool:
    """
    Verifica se todas as etapas anteriores foram executadas
    Confia que o notebook vai validar corretamente
    """
    state = PipelineState()
    
    for prev_stage in range(1, current_stage):
        if not state.is_stage_completed(prev_stage):
            return False
    return True


def show_pipeline_progress() -> None:
    """Mostra progresso visual do pipeline"""
    state = PipelineState()
    progress = state.get_pipeline_progress()
    
    stage_names = [
        "🏗️ Fundação e Preparação",
        "📥 Coleta GitLab", 
        "⚙️ Processamento Docling",
        "🔪 Segmentação em Chunks",
        "🧠 Geração de Embeddings",
        "💾 Armazenamento Qdrant", 
        "📊 Validação e Resultados"
    ]
    
    print("🎯 PROGRESSO DO PIPELINE NIC ETL")
    print("═" * 60)
    
    for i, stage_info in enumerate(progress):
        stage_num = stage_info["stage"]
        completed = stage_info["completed"]
        name = stage_names[i]
        
        if completed:
            print(f"✅ {stage_num:02d}. {name}")
        else:
            print(f"⏳ {stage_num:02d}. {name}")
    
    print("═" * 60)




def format_file_size(size_bytes: int) -> str:
    """Formata tamanho de arquivo para exibição"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_timestamp() -> str:
    """Retorna timestamp atual formatado"""
    return datetime.now().isoformat()


def ensure_directory(path: str) -> None:
    """Garante que diretório existe"""
    Path(path).mkdir(parents=True, exist_ok=True)