# -*- coding: utf-8 -*-
"""
Utilitários para integração com GitLab
Biblioteca ultrassimples - apenas algoritmos puros
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import gitlab
import io


def download_documents(gitlab_url: str, repository: str, branch: str, 
                      folder_path: str, token: str, target_dir: str,
                      file_extensions: List[str] = None) -> Dict[str, Any]:
    """
    Baixa documentos do GitLab
    Função pura - extrai automaticamente URL da instância de URLs de repositório
    """
    
    # Extrair URL da instância GitLab automaticamente
    def extract_instance_url(url: str) -> str:
        """Extrai URL da instância GitLab de qualquer URL fornecida"""
        # Remover .git se presente
        clean_url = url.rstrip('.git')
        
        # Usar urlparse para dividir componentes
        from urllib.parse import urlparse
        parsed = urlparse(clean_url)
        
        # Reconstruir apenas protocolo + netloc (host:porta)
        instance_url = f"{parsed.scheme}://{parsed.netloc}"
        
        return instance_url
    
    instance_url = extract_instance_url(gitlab_url)
    
    print(f"🔗 Conectando ao GitLab: {instance_url}")
    print(f"📂 Repositório: {repository}")
    print(f"🌿 Branch: {branch}")
    print(f"📁 Pasta: {folder_path}")
    
    # Conectar ao GitLab
    gl = gitlab.Gitlab(instance_url, private_token=token)
    project = gl.projects.get(repository, lazy=True)
    
    downloaded_files = []
    errors = []
    
    # Criar diretório de destino
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Listar arquivos na pasta
        items = project.repository_tree(path=folder_path, ref=branch, recursive=True)
        
        for item in items:
            if item['type'] != 'blob':  # Pular diretórios
                continue
                
            filename = item['name']
            file_path_in_repo = item['path']
            
            # Verificar extensão se filtros foram especificados
            if file_extensions:
                file_ext = Path(filename).suffix.lower()
                if file_ext not in [ext.lower() for ext in file_extensions]:
                    continue
            
            try:
                # Baixar arquivo
                file_content = project.files.get(file_path=file_path_in_repo, ref=branch)
                content = file_content.decode()
                
                # Salvar arquivo localmente
                local_file_path = target_path / filename
                
                if isinstance(content, bytes):
                    with open(local_file_path, 'wb') as f:
                        f.write(content)
                else:
                    with open(local_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                # Adicionar à lista de arquivos baixados
                downloaded_files.append({
                    "gitlab_path": file_path_in_repo,
                    "local_path": str(local_file_path),
                    "file_info": {
                        "size_bytes": len(content.encode('utf-8') if isinstance(content, str) else content),
                        "extension": Path(filename).suffix,
                        "gitlab_metadata": {
                            "commit_id": item.get('id', 'unknown'),
                            "last_modified": item.get('last_modified', datetime.now().isoformat()),
                            "author": "unknown"
                        }
                    }
                })
                
                print(f"📄 Baixado: {filename}")
                
            except Exception as e:
                error_msg = f"Erro ao baixar {filename}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
                
    except Exception as e:
        error_msg = f"Erro ao listar arquivos na pasta {folder_path}: {str(e)}"
        errors.append(error_msg)
        print(f"❌ {error_msg}")
    
    return {
        "downloaded_files": downloaded_files,
        "statistics": {
            "total_files_found": len(downloaded_files) + len(errors),
            "successfully_downloaded": len(downloaded_files), 
            "failed_downloads": len(errors),
            "total_size_mb": sum(f["file_info"]["size_bytes"] for f in downloaded_files) / (1024*1024)
        },
        "errors": errors
    }


def validate_gitlab_connection(gitlab_url: str, token: str) -> bool:
    """
    Valida conexão com GitLab
    Função pura - extrai automaticamente URL da instância de URLs de repositório
    """
    print(f"🔍 Validando conexão com {gitlab_url}")
    
    if not token:
        print("❌ Token não fornecido")
        return False
    
    # Extrair URL da instância GitLab automaticamente
    def extract_instance_url(url: str) -> str:
        """Extrai URL da instância GitLab de qualquer URL fornecida"""
        # Remover .git se presente
        clean_url = url.rstrip('.git')
        
        # Usar urlparse para dividir componentes
        from urllib.parse import urlparse
        parsed = urlparse(clean_url)
        
        # Reconstruir apenas protocolo + netloc (host:porta)
        instance_url = f"{parsed.scheme}://{parsed.netloc}"
        
        return instance_url
    
    instance_url = extract_instance_url(gitlab_url)
    
    # Mostrar URLs se diferentes (para debug)
    if instance_url != gitlab_url:
        print(f"🔗 URL original: {gitlab_url}")
        print(f"🔗 URL da instância extraída: {instance_url}")
    
    try:
        gl = gitlab.Gitlab(instance_url, private_token=token)
        # Testar conexão fazendo uma requisição simples
        gl.auth()
        user = gl.user
        print(f"✅ Conexão validada para usuário: {user.username}")
        return True
        
    except Exception as e:
        print(f"❌ Falha na validação: {str(e)}")
        return False


def list_repository_files(gitlab_url: str, repository: str, branch: str,
                         folder_path: str, token: str) -> List[Dict[str, Any]]:
    """
    Lista arquivos no repositório GitLab
    Função pura - extrai automaticamente URL da instância de URLs de repositório
    """
    
    try:
        # Extrair URL da instância GitLab automaticamente
        def extract_instance_url(url: str) -> str:
            """Extrai URL da instância GitLab de qualquer URL fornecida"""
            # Remover .git se presente
            clean_url = url.rstrip('.git')
            
            # Usar urlparse para dividir componentes
            from urllib.parse import urlparse
            parsed = urlparse(clean_url)
            
            # Reconstruir apenas protocolo + netloc (host:porta)
            instance_url = f"{parsed.scheme}://{parsed.netloc}"
            
            return instance_url
        
        instance_url = extract_instance_url(gitlab_url)
        
        gl = gitlab.Gitlab(instance_url, private_token=token)
        project = gl.projects.get(repository, lazy=True)
        
        items = project.repository_tree(path=folder_path, ref=branch, recursive=True)
        
        files = []
        for item in items:
            if item['type'] == 'blob':  # Apenas arquivos
                files.append({
                    "name": item['name'],
                    "path": item['path'],
                    "size": item.get('size', 0),
                    "last_modified": item.get('last_modified', 'unknown')
                })
        
        return files
        
    except Exception as e:
        print(f"❌ Erro ao listar arquivos: {str(e)}")
        return []


def clean_download_directory(target_dir: str) -> bool:
    """
    Limpa diretório de downloads
    Função simples - remove tudo
    """
    target_path = Path(target_dir)
    
    if target_path.exists():
        shutil.rmtree(target_path)
        print(f"🧹 Diretório {target_dir} limpo")
    
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Diretório {target_dir} recriado")
    
    return True