"""
NIC ETL Source Library
Biblioteca customizada para substituir dependências externas problemáticas
"""

from .gitlab_client import NicGitlab, NicGitlabClient, NicGitlabProject

__all__ = ['NicGitlab', 'NicGitlabClient', 'NicGitlabProject']