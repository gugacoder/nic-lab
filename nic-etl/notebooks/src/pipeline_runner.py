"""
Pipeline Runner - Gerenciador de execução em background do pipeline ETL
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict


class PipelineRunner:
    """
    Gerencia a execução do pipeline ETL em background com controle de trava
    e redirecionamento de logs.
    """
    
    def __init__(self):
        self.notebook = "etl.ipynb"
        self.notebook_name = Path(self.notebook).stem  # etl.ipynb -> etl
        self.lock_file = Path(f"pipeline-data/{self.notebook_name}.lock")
        self.status_file = Path("pipeline-data/report.json")
        self.log_file = Path("logs/background-job.log")
        self.wrapper_script = Path("src/run_notebook_wrapper.py")
        
        # Criar diretórios se não existirem
        self.lock_file.parent.mkdir(exist_ok=True)
        self.log_file.parent.mkdir(exist_ok=True)
    
    def is_running(self) -> bool:
        """
        Verifica se o pipeline já está em execução através do arquivo de trava.
        
        Returns:
            bool: True se estiver rodando, False caso contrário
        """
        if not self.lock_file.exists():
            return False
        
        # Verificar se o processo ainda existe
        try:
            with open(self.lock_file, "r") as f:
                content = f.read()
                for line in content.strip().split('\n'):
                    if line.startswith('wrapper_pid='):
                        pid = int(line.split('=')[1])
                        # Verificar se processo ainda existe
                        try:
                            os.kill(pid, 0)
                            return True  # Processo existe
                        except OSError:
                            # Processo não existe mais, remover trava órfã
                            self.lock_file.unlink()
                            return False
        except (ValueError, FileNotFoundError):
            # Arquivo corrompido, remover
            if self.lock_file.exists():
                self.lock_file.unlink()
            return False
        
        return False
    
    def start_background(self) -> Dict[str, str]:
        """
        Inicia a execução do pipeline em background.
        
        Returns:
            Dict: Status da operação
        """
        if self.is_running():
            return {"status": "job_running"}
        
        # Inicializar status no report.json
        initial_status = {
            "pipeline_info": {
                "version": "1.0.0",
                "started_at": datetime.now().isoformat() + "Z",
                "environment": "development"
            },
            "summary": {
                "pipeline_status": "RUNNING",
                "message": "Pipeline iniciado em background via wrapper",
                "current_stage": "INITIALIZING"
            },
            "api_metadata": {
                "job_started_at": datetime.now().isoformat() + "Z",
                "execution_mode": "background_wrapper",
                "log_file": str(self.log_file),
                "wrapper_script": str(self.wrapper_script)
            }
        }
        
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(initial_status, f, indent=2, ensure_ascii=False)
        
        # Executar notebook via wrapper em background com nohup
        try:
            # Usar wrapper que gerencia a trava automaticamente
            subprocess.Popen([
                "nohup", "python", str(self.wrapper_script),
                self.notebook,
                "--log-file", str(self.log_file)
            ], stdout=subprocess.DEVNULL, 
               stderr=subprocess.STDOUT,
               cwd=Path.cwd(),
               preexec_fn=os.setsid)  # Criar nova sessão para desacoplar do terminal
                
        except Exception as e:
            # Atualizar status com erro (não precisamos remover trava, wrapper não foi iniciado)
            error_status = {
                "pipeline_info": {
                    "version": "1.0.0",
                    "started_at": datetime.now().isoformat() + "Z",
                    "environment": "development"
                },
                "summary": {
                    "pipeline_status": "FAILED",
                    "message": f"Erro ao iniciar wrapper: {str(e)}",
                    "current_stage": "STARTUP_ERROR"
                },
                "api_metadata": {
                    "job_started_at": datetime.now().isoformat() + "Z",
                    "execution_mode": "background_wrapper",
                    "error": str(e)
                }
            }
            
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(error_status, f, indent=2, ensure_ascii=False)
            
            return {"status": "error", "message": str(e)}
        
        return {"status": "job_running"}
    
    def get_status(self) -> Dict:
        """
        Obtém o status atual do pipeline a partir do report.json.
        
        Returns:
            Dict: Status atual do pipeline
        """
        if not self.status_file.exists():
            return {
                "pipeline_info": {
                    "version": "1.0.0",
                    "last_execution": None,
                    "environment": "unknown"
                },
                "summary": {
                    "pipeline_status": "IDLE",
                    "message": "Pipeline não foi executado ainda"
                },
                "api_metadata": {
                    "endpoint": "/api/v1/pipelines/gitlab-qdrant/runs/last",
                    "served_at": datetime.now().isoformat() + "Z",
                    "report_exists": False
                }
            }
        
        try:
            with open(self.status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
            
            # Adicionar metadados da API
            status.setdefault("api_metadata", {})
            status["api_metadata"].update({
                "endpoint": "/api/v1/pipelines/gitlab-qdrant/runs/last",
                "served_at": datetime.now().isoformat() + "Z",
                "report_exists": True,
                "is_running": self.is_running()
            })
            
            return status
            
        except json.JSONDecodeError as e:
            return {
                "error": "Invalid report format",
                "message": f"O arquivo de report existe mas contém JSON inválido: {str(e)}",
                "status_code": 500,
                "timestamp": datetime.now().isoformat() + "Z"
            }
        except Exception as e:
            return {
                "error": "Internal server error", 
                "message": f"Erro ao ler o report: {str(e)}",
                "status_code": 500,
                "timestamp": datetime.now().isoformat() + "Z"
            }
    
    def cleanup_lock(self):
        """
        Remove o arquivo de trava manualmente (o wrapper já faz isso automaticamente).
        Método mantido para compatibilidade.
        """
        if self.lock_file.exists():
            self.lock_file.unlink()