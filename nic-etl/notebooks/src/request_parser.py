"""
Parser para objetos REQUEST do Jupyter Kernel Gateway.

Fornece uma interface limpa para extrair parâmetros de requisições HTTP
vindas do Jupyter Kernel Gateway.
"""

import json
from typing import Dict, Any, Optional, List, Union


def parse_request(request: Optional[Any] = None) -> Dict[str, Any]:
    """
    Faz o parse do objeto REQUEST do Jupyter Kernel Gateway.
    REQUEST pode vir como dict, string JSON ou bytes.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway (dict, str, bytes) ou None
        
    Returns:
        Dict com estrutura padronizada:
        {
            "args": {},          # Query parameters
            "headers": {},       # HTTP headers
            "body": None,        # Request body
            "method": "GET",     # HTTP method
            "path": ""          # Request path
        }
    """
    # Se None, retornar mock
    if request is None:
        return {
            "args": {},
            "headers": {},
            "body": None,
            "method": "GET",
            "path": ""
        }
    
    # Se já é dict, usar direto
    if isinstance(request, dict):
        return {
            "args": request.get("args", {}),
            "headers": request.get("headers", {}),
            "body": request.get("body"),
            "method": request.get("method", "GET"),
            "path": request.get("path", "")
        }
    
    # Se é bytes/bytearray, converter para string
    if isinstance(request, (bytes, bytearray)):
        try:
            request = request.decode("utf-8")
        except:
            # Se falhar decode, retornar estrutura vazia
            return {
                "args": {},
                "headers": {},
                "body": None,
                "method": "GET",
                "path": ""
            }
    
    # Se é string, tentar parse JSON
    if isinstance(request, str):
        try:
            parsed = json.loads(request)
            return {
                "args": parsed.get("args", {}),
                "headers": parsed.get("headers", {}),
                "body": parsed.get("body"),
                "method": parsed.get("method", "GET"),
                "path": parsed.get("path", "")
            }
        except:
            # Se falhar parse JSON, retornar estrutura vazia
            return {
                "args": {},
                "headers": {},
                "body": None,
                "method": "GET",
                "path": ""
            }
    
    # Tipo desconhecido, retornar estrutura vazia
    return {
        "args": {},
        "headers": {},
        "body": None,
        "method": "GET",
        "path": ""
    }


def get_query_param(request: Optional[Any], 
                    param_name: str, 
                    default: Optional[Any] = None) -> Optional[Any]:
    """
    Extrai um parâmetro específico da query string.
    Lida com parâmetros que podem vir como lista.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway (dict, str, bytes)
        param_name: Nome do parâmetro a extrair
        default: Valor padrão se parâmetro não existir
        
    Returns:
        Valor do parâmetro ou default
    """
    parsed = parse_request(request)
    value = parsed["args"].get(param_name, default)
    
    # Se valor é uma lista, pegar o primeiro elemento
    if isinstance(value, list) and value:
        return value[0]
    
    return value


def get_all_query_params(request: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Retorna todos os parâmetros da query string.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        
    Returns:
        Dict com todos os parâmetros
    """
    parsed = parse_request(request)
    return parsed["args"]


def has_query_param(request: Optional[Dict[str, Any]], param_name: str) -> bool:
    """
    Verifica se um parâmetro existe na query string.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        param_name: Nome do parâmetro
        
    Returns:
        True se parâmetro existe, False caso contrário
    """
    parsed = parse_request(request)
    return param_name in parsed["args"]


def validate_action(request: Optional[Dict[str, Any]], 
                   valid_actions: List[str]) -> Union[str, Dict[str, Any]]:
    """
    Valida o parâmetro 'action' contra uma lista de ações válidas.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        valid_actions: Lista de ações válidas
        
    Returns:
        str: Nome da ação se válida
        dict: Erro formatado se inválida ou None
    """
    action = get_query_param(request, "action")
    
    if action is None:
        return None
    
    if action in valid_actions:
        return action
    
    # Ação inválida
    return {
        "error": "invalid_action",
        "message": f"action '{action}' é inválido.",
        "supported_actions": valid_actions
    }


def get_header(request: Optional[Dict[str, Any]], 
               header_name: str, 
               default: Optional[str] = None) -> Optional[str]:
    """
    Extrai um header específico da requisição.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        header_name: Nome do header
        default: Valor padrão se header não existir
        
    Returns:
        Valor do header ou default
    """
    parsed = parse_request(request)
    return parsed["headers"].get(header_name, default)


def get_request_body(request: Optional[Dict[str, Any]]) -> Optional[Any]:
    """
    Extrai o body da requisição.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        
    Returns:
        Body da requisição ou None
    """
    parsed = parse_request(request)
    return parsed["body"]


def is_mock_request(request: Optional[Any]) -> bool:
    """
    Verifica se é uma requisição mock (desenvolvimento local).
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        
    Returns:
        True se for mock, False se for requisição real
    """
    return request is None or not request


def get_action(request: Optional[Any]) -> Optional[str]:
    """
    Extrai o parâmetro 'action' da requisição.
    Busca primeiro em args (query params), depois no body JSON.
    
    Args:
        request: Objeto REQUEST do Jupyter Kernel Gateway
        
    Returns:
        Valor de action ou None
    """
    parsed = parse_request(request)
    
    # Primeiro tentar pegar de args
    args = parsed.get("args", {})
    action = args.get("action")
    
    # Se action é uma lista, pegar o primeiro elemento
    if isinstance(action, list) and action:
        action = action[0]
    
    # Se não encontrou em args, tentar no body
    if action is None:
        body = parsed.get("body")
        
        # Se body é bytes/bytearray, decodificar
        if isinstance(body, (bytes, bytearray)):
            try:
                body = body.decode("utf-8")
            except:
                body = None
        
        # Se body é string JSON, fazer parse
        if isinstance(body, str) and body.strip():
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    action = data.get("action", action)
            except:
                pass
    
    return action