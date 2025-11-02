"""
Cliente GitLab customizado para NIC ETL
Substitui completamente a biblioteca python-gitlab
Usa apenas requests + APIs GitLab v4 + raw files
"""

import requests
from urllib.parse import quote, urlparse
from typing import List, Dict, Optional
import json


class NicGitlabProject:
    """Representa um projeto GitLab com operações necessárias para o NIC ETL"""
    
    def __init__(self, client: 'NicGitlabClient', project_path: str):
        self.client = client
        self.project_path = project_path
        self._project_id = None
    
    @property
    def project_id(self) -> int:
        """Obtém o ID do projeto (lazy loading)"""
        if self._project_id is None:
            self._project_id = self.client._get_project_id(self.project_path)
        return self._project_id
    
    def repository_tree(self, path: str = "", ref: str = "main", 
                       recursive: bool = False, all: bool = True, 
                       per_page: int = 100) -> List[Dict]:
        """
        Lista árvore do repositório - compatível com python-gitlab
        
        Args:
            path: Caminho dentro do repositório
            ref: Branch ou commit
            recursive: Se deve listar recursivamente  
            all: Se deve listar todos os itens
            per_page: Itens por página
            
        Returns:
            Lista de dicts com name, path, type, etc.
        """
        url = f"{self.client.instance_url}/api/v4/projects/{self.project_id}/repository/tree"
        
        params = {
            "ref": ref,
            "per_page": per_page if not all else 100,
            "recursive": recursive
        }
        
        if path:
            params["path"] = path
            
        headers = {"PRIVATE-TOKEN": self.client.private_token}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            items = response.json()
            
            # Se all=True e há mais páginas, buscar todas
            if all and len(items) == per_page:
                all_items = items[:]
                page = 2
                while True:
                    params["page"] = page
                    response = requests.get(url, params=params, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    page_items = response.json()
                    if not page_items:
                        break
                        
                    all_items.extend(page_items)
                    page += 1
                    
                return all_items
            
            return items
            
        except requests.RequestException as e:
            raise Exception(f"Erro ao listar árvore do repositório: {e}")
    
    @property
    def commits(self):
        """Retorna manager de commits compatível com python-gitlab"""
        return CommitsManager(self)
    
    def download_file_raw(self, file_path: str, ref: str = "main") -> bytes:
        """
        Baixa conteúdo de arquivo usando raw URL
        Substitui project.files.raw() da biblioteca gitlab
        
        Args:
            file_path: Caminho do arquivo no repositório
            ref: Branch ou commit
            
        Returns:
            Conteúdo do arquivo em bytes
        """
        # Codificar o file_path para URL
        encoded_path = quote(file_path, safe='/')
        
        # URL raw: /{project_path}/-/raw/{ref}/{file_path}
        url = f"{self.client.instance_url}/{self.project_path}/-/raw/{ref}/{encoded_path}"
        
        headers = {"PRIVATE-TOKEN": self.client.private_token}
        
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            return response.content
            
        except requests.RequestException as e:
            raise Exception(f"Erro ao baixar arquivo {file_path}: {e}")


class NicGitlabClient:
    """Cliente GitLab customizado usando apenas requests + APIs v4"""
    
    def __init__(self, instance_url: str, private_token: str):
        """
        Inicializa cliente GitLab
        
        Args:
            instance_url: URL da instância GitLab (ex: http://gitlab.processa.info)
            private_token: Token de acesso privado
        """
        self.instance_url = instance_url.rstrip('/')
        self.private_token = private_token
        
        # Validar conectividade
        self._validate_connection()
    
    def _validate_connection(self):
        """Valida se consegue conectar ao GitLab"""
        url = f"{self.instance_url}/api/v4/user"
        headers = {"PRIVATE-TOKEN": self.private_token}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"Erro ao conectar ao GitLab: {e}")
    
    def _get_project_id(self, project_path: str) -> int:
        """
        Obtém ID do projeto pelo path (namespace/projeto)
        
        Args:
            project_path: Caminho do projeto (ex: nic/documentacao/base-de-conhecimento)
            
        Returns:
            ID numérico do projeto
        """
        # Buscar projetos que contenham o nome do projeto
        project_name = project_path.split('/')[-1]
        url = f"{self.instance_url}/api/v4/projects"
        
        params = {
            "search": project_name,
            "membership": False,
            "simple": False,
            "per_page": 100
        }
        
        headers = {"PRIVATE-TOKEN": self.private_token}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            projects = response.json()
            
            # Procurar projeto com path_with_namespace exato
            for project in projects:
                if project.get("path_with_namespace") == project_path:
                    return project["id"]
            
            raise ValueError(f"Projeto não encontrado: {project_path}")
            
        except requests.RequestException as e:
            raise Exception(f"Erro ao buscar projeto: {e}")
    
    def get_project(self, project_path: str) -> NicGitlabProject:
        """
        Obtém projeto pelo path - compatível com gl.projects.get()
        
        Args:
            project_path: Caminho do projeto (ex: nic/documentacao/base-de-conhecimento)
            
        Returns:
            Instância de NicGitlabProject
        """
        return NicGitlabProject(self, project_path)


# Classe compatível com sintaxe da biblioteca gitlab
class NicGitlab:
    """Interface de compatibilidade com sintaxe gitlab.Gitlab()"""
    
    def __init__(self, instance_url: str, private_token: str):
        self.client = NicGitlabClient(instance_url, private_token)
        self.projects = ProjectsManager(self.client)


class CommitsManager:
    """Manager para commits - compatível com project.commits"""
    
    def __init__(self, project: NicGitlabProject):
        self.project = project
    
    def list(self, ref_name: str = "main", per_page: int = 20) -> List[Dict]:
        """
        Lista commits do projeto
        
        Args:
            ref_name: Nome do branch
            per_page: Número de commits por página
            
        Returns:
            Lista de commits com id, committed_date, etc.
        """
        url = f"{self.project.client.instance_url}/api/v4/projects/{self.project.project_id}/repository/commits"
        
        params = {
            "ref_name": ref_name,
            "per_page": per_page
        }
        
        headers = {"PRIVATE-TOKEN": self.project.client.private_token}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            raise Exception(f"Erro ao listar commits: {e}")


class ProjectsManager:
    """Manager para projetos - compatível com gl.projects"""
    
    def __init__(self, client: NicGitlabClient):
        self.client = client
    
    def list(self, search: str, all: bool = True) -> List[Dict]:
        """
        Lista projetos - compatível com gl.projects.list()
        Retorna objetos que simulam projetos para get_project_id()
        """
        url = f"{self.client.instance_url}/api/v4/projects"
        params = {
            "search": search,
            "membership": False, 
            "simple": False,
            "per_page": 100 if all else 20
        }
        
        headers = {"PRIVATE-TOKEN": self.client.private_token}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            raise Exception(f"Erro ao listar projetos: {e}")
    
    def get(self, project_path_or_id) -> NicGitlabProject:
        """Obtém projeto por path ou ID"""
        if isinstance(project_path_or_id, int):
            # Se for ID, precisamos converter para path
            # Para simplicidade, vamos assumir que sempre passamos path
            raise NotImplementedError("Use project path ao invés de ID")
        
        return self.client.get_project(project_path_or_id)