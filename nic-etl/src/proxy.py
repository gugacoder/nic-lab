"""
FastAPI Reverse Proxy para NIC Lab

Serve arquivos estáticos, renderiza Markdown com templates Jinja2,
e faz proxy de /api/* para o Jupyter Kernel Gateway.
"""

import os
from pathlib import Path
from typing import Optional

import httpx
import markdown
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown.extensions import extra, codehilite, toc

# Configuração
KERNEL_GATEWAY_URL = "http://127.0.0.1:5001"
RAG_API_URL = "http://127.0.0.1:5002"
STATIC_DIR = Path("static")
TEMPLATES_DIR = STATIC_DIR / "templates"

# Inicializar FastAPI
app = FastAPI(
    title="NIC Lab",
    description="Núcleo de Inteligência e Conhecimento - ETL Pipeline",
    version="1.0.0"
)

# Montar arquivos estáticos
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

# Templates Jinja2
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Configurar Markdown
md = markdown.Markdown(
    extensions=[
        'extra',           # Tables, footnotes, def lists
        'codehilite',      # Code syntax highlighting
        'fenced_code',     # ```code blocks```
        'tables',          # | Markdown | Tables |
        'toc',            # Table of contents
        'attr_list',      # {: .class} attributes
        'md_in_html'      # HTML blocks with markdown
    ],
    extension_configs={
        'codehilite': {
            'css_class': 'highlight',
            'use_pygments': True
        },
        'toc': {
            'permalink': True
        }
    }
)


async def render_page(page_name: str, request: Request) -> HTMLResponse:
    """
    Renderiza uma página com fallback HTML → Markdown.
    
    Procura primeiro por {page_name}.html, depois {page_name}.md
    Se encontrar .md, renderiza com template base.html
    """
    # Tentar arquivo HTML primeiro
    html_file = STATIC_DIR / f"{page_name}.html"
    if html_file.exists():
        return FileResponse(html_file)
    
    # Tentar arquivo Markdown
    md_file = STATIC_DIR / f"{page_name}.md"
    if md_file.exists():
        try:
            # Ler e renderizar Markdown
            content = md_file.read_text(encoding="utf-8")
            html_content = md.convert(content)
            
            # Extrair título do primeiro H1 ou usar nome da página
            title = "NIC Lab"
            if hasattr(md, 'toc_tokens') and md.toc_tokens:
                title = f"{md.toc_tokens[0]['name']} - NIC Lab"
            
            # Renderizar com template
            return templates.TemplateResponse("base.html", {
                "request": request,
                "title": title,
                "content": html_content,
                "page_name": page_name
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao renderizar {page_name}.md: {str(e)}")
    
    # Arquivo não encontrado
    raise HTTPException(status_code=404, detail=f"Página '{page_name}' não encontrada")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Homepage - renderiza index.html ou index.md"""
    return await render_page("index", request)


@app.get("/{page_name}", response_class=HTMLResponse)
async def page(page_name: str, request: Request):
    """
    Página genérica - renderiza {page_name}.html ou {page_name}.md
    """
    return await render_page(page_name, request)


@app.api_route("/rag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def proxy_to_rag_api(path: str, request: Request):
    """
    Proxy transparente para o RAG API Kernel Gateway.
    
    Todas as requisições /rag/* são redirecionadas para o RAG API (porta 5002)
    mantendo método, headers, query params e body.
    """
    # Construir URL do RAG API
    url = f"{RAG_API_URL}/api/{path}"
    
    # Copiar headers da requisição original
    headers = dict(request.headers)
    # Remover headers que podem causar problemas no proxy
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    # Obter body da requisição
    body = await request.body()
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
        
        # Retornar resposta do RAG API sem modificações
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Erro ao conectar com RAG API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no proxy RAG: {str(e)}"
        )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def proxy_to_kernel_gateway(path: str, request: Request):
    """
    Proxy transparente para o Jupyter Kernel Gateway.
    
    Todas as requisições /api/* são redirecionadas para o Kernel Gateway
    mantendo método, headers, query params e body.
    """
    # Construir URL do Kernel Gateway
    url = f"{KERNEL_GATEWAY_URL}/api/{path}"
    
    # Copiar headers da requisição original
    headers = dict(request.headers)
    # Remover headers que podem causar problemas no proxy
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    # Obter body da requisição
    body = await request.body()
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
        
        # Retornar resposta do Kernel Gateway sem modificações
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )
        
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Erro ao conectar com Kernel Gateway: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no proxy: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check do proxy"""
    return {
        "status": "ok",
        "service": "NIC Lab Proxy",
        "kernel_gateway": KERNEL_GATEWAY_URL,
        "rag_api": RAG_API_URL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)