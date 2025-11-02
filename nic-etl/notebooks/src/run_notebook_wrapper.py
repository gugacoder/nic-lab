#!/usr/bin/env python3
"""
Wrapper genérico para executar notebooks com gerenciamento automático de trava.

Uso:
    python run_notebook_wrapper.py <notebook.ipynb>

Funcionalidades:
- Cria trava antes da execução
- Remove trava automaticamente (try/finally)
- Executa qualquer notebook via jupyter nbconvert
- Logs de execução com timestamps
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path


def main():
    # Nomear o processo para facilitar identificação no ps
    sys.argv[0] = "nic-etl-pipeline"
    
    parser = argparse.ArgumentParser(description="Executa notebook com gerenciamento de trava")
    parser.add_argument("notebook", help="Nome do notebook a ser executado (ex: etl.ipynb)")
    parser.add_argument("--lock-dir", default="pipeline-data", 
                       help="Diretório para arquivo de trava (padrão: pipeline-data)")
    parser.add_argument("--log-file", help="Arquivo de log opcional")
    
    args = parser.parse_args()
    
    # Verificar se notebook existe
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"❌ Erro: Notebook '{args.notebook}' não encontrado", file=sys.stderr)
        sys.exit(1)
    
    # Configurar paths
    lock_dir = Path(args.lock_dir)
    lock_dir.mkdir(exist_ok=True)
    
    # Nome da trava baseado no notebook
    notebook_name = notebook_path.stem  # etl.ipynb -> etl
    lock_file = lock_dir / f"{notebook_name}.lock"
    
    # Logging setup
    def log(message):
        timestamp = datetime.now().isoformat()
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        if args.log_file:
            try:
                with open(args.log_file, "a") as f:
                    f.write(log_message + "\n")
            except Exception:
                pass  # Se não conseguir logar, continua
    
    log(f"🚀 Iniciando execução do notebook: {args.notebook}")
    log(f"📍 PID do wrapper: {os.getpid()}")
    log(f"🔒 Arquivo de trava: {lock_file}")
    
    # Verificar se já existe trava
    if lock_file.exists():
        log(f"⚠️ Trava já existe: {lock_file}")
        try:
            with open(lock_file, "r") as f:
                existing_lock = f.read().strip()
            log(f"🔍 Conteúdo da trava existente: {existing_lock}")
        except Exception:
            pass
        
        print(f"❌ Erro: Notebook '{notebook_name}' já está em execução", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Criar arquivo de trava
        with open(lock_file, "w") as f:
            f.write(f"notebook={args.notebook}\n")
            f.write(f"started_at={datetime.now().isoformat()}\n")
            f.write(f"wrapper_pid={os.getpid()}\n")
            f.write(f"hostname={os.uname().nodename}\n")
        
        log(f"✅ Trava criada: {lock_file}")
        
        # Executar notebook
        log(f"▶️ Executando: jupyter nbconvert --execute --inplace {args.notebook}")
        
        result = subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook", 
            "--execute",
            "--inplace",
            str(notebook_path)
        ], 
        capture_output=True, 
        text=True, 
        check=True)
        
        log(f"✅ Notebook executado com sucesso")
        
        # Log de outputs se houver
        if result.stdout:
            log(f"📤 STDOUT: {result.stdout.strip()}")
        if result.stderr:
            log(f"📤 STDERR: {result.stderr.strip()}")
            
        return 0
        
    except subprocess.CalledProcessError as e:
        log(f"❌ Erro na execução do notebook (código: {e.returncode})")
        
        if e.stdout:
            log(f"📤 STDOUT: {e.stdout.strip()}")
        if e.stderr:
            log(f"📤 STDERR: {e.stderr.strip()}")
            
        return e.returncode
        
    except Exception as e:
        log(f"❌ Erro inesperado: {str(e)}")
        return 1
        
    finally:
        # SEMPRE remover trava, independente de sucesso ou falha
        try:
            if lock_file.exists():
                lock_file.unlink()
                log(f"🔓 Trava removida: {lock_file}")
            else:
                log(f"⚠️ Trava não encontrada para remoção: {lock_file}")
        except Exception as e:
            log(f"⚠️ Erro ao remover trava: {e}")
        
        log(f"🏁 Wrapper finalizado para {args.notebook}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)