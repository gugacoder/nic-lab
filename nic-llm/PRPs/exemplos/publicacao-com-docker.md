Use **docker-compose** com Traefik na sua rede `codr-net`. Não use Prefect para servir.

## Passo a passo

1. **Estrutura**

```
/srv/apps/nic-llm-gateway/
  ├─ src/               # coloque aqui o arquivo nic_llm_gateway.py (o wrapper)
  ├─ requirements.txt
  ├─ Dockerfile
  ├─ .env
  └─ docker-compose.yml
```

2. **requirements.txt**

```
fastapi
uvicorn
httpx
pydantic
qdrant-client
openai
anthropic
google-genai
orjson
uvloop
httptools
```

3. **Dockerfile**

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .   # deve conter nic_llm_gateway.py com o app FastAPI
CMD ["uvicorn","nic_llm_gateway:app","--host","0.0.0.0","--port","8001","--workers","1","--http","httptools","--loop","uvloop"]
```

4. **.env (exemplo)**

```
GATEWAY_PORT=8001
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=nic_docs
QDRANT_API_KEY=CHAVE_OPCIONAL
EMBED_MODEL=openai:text-embedding-3-small

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-7-sonnet-latest

GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-2.5-flash

OPENROUTER_API_KEY=...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

5. **docker-compose.yml**

```yaml
version: "3.8"

services:
  nic-llm-gateway:
    build: .
    container_name: nic-llm-gateway
    env_file: .env
    restart: unless-stopped
    networks: [codr-net]
    # Se seu Qdrant já roda em outro compose, remova depends_on
    depends_on:
      - qdrant
    labels:
      - traefik.enable=true
      - traefik.http.routers.nic-llm-gw.rule=Host(`nic-llm.codrstudio.dev`)
      - traefik.http.routers.nic-llm-gw.entrypoints=websecure
      - traefik.http.routers.nic-llm-gw.tls=true
      # ajuste o certresolver conforme seu Traefik:
      - traefik.http.routers.nic-llm-gw.tls.certresolver=letsencrypt
      - traefik.http.services.nic-llm-gw.loadbalancer.server.port=8001
      # CORS opcional para uso por apps externos
      - traefik.http.middlewares.nic-llm-gw-cors.headers.accessControlAllowOriginList=*
      - traefik.http.routers.nic-llm-gw.middlewares=nic-llm-gw-cors
    healthcheck:
      test: ["CMD-SHELL","curl -fsS http://localhost:8001/docs >/dev/null || exit 1"]
      interval: 30s
      timeout: 3s
      retries: 5

  # Remova esta seção se já possui Qdrant em outro stack
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    restart: unless-stopped
    networks: [codr-net]
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
    labels:
      - traefik.enable=false

networks:
  codr-net:
    external: true
```

6. **Subir**

```bash
cd /srv/apps/nic-llm-gateway
docker compose up -d --build
```

7. **Teste**

* Não-stream:

```bash
curl -sS https://nic-llm.codrstudio.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nic-llm-v1","stream":false,"messages":[{"role":"user","content":"ping"}]}'
```

* Streaming SSE:

```bash
curl -N https://nic-llm.codrstudio.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nic-llm-v1","stream":true,"messages":[{"role":"user","content":"ping"}]}'
```

## Notas de desempenho

* Uvicorn com `uvloop` e `httptools` entrega baixa latência.
* Traefik suporta SSE nativamente. Não precisa “desligar buffering”.
* Escale por processo: `--workers` ou réplicas do serviço e deixe o Traefik balancear.
* Use keep-alive e HTTP/2 no upstream quando disponível.
* O tempo do provedor LLM continuará dominando a latência total.
