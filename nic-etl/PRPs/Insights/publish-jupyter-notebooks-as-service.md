# Visão rápida

* O **Kernel Gateway** em modo **`notebook-http`** publica **células anotadas** do notebook como **rotas HTTP**. ([Jupyter Kernel Gateway][1])
* Você aponta um notebook via `KG_SEED_URI` e diz que a “personalidade” é `kernel_gateway.notebook_http`. ([Jupyter Kernel Gateway][1])
* Ele gera até um **Swagger JSON** em `/_api/spec/swagger.json`. ([Jupyter Kernel Gateway][1])

---

# 1) Estrutura de pastas (no host)

```
/srv/notebooks/
  api.ipynb        # seu notebook com as rotas
Dockerfile
docker-compose.yml
```

---

# 2) Exemplo mínimo de notebook (`api.ipynb`)

No seu notebook, cada **célula handler** começa com um comentário com **método e rota**. O objeto da requisição chega em `REQUEST` (JSON string). ([Jupyter Kernel Gateway][1])

**Exemplo de células:**

**GET /ping**

```python
# GET /ping
print("pong")
```

**POST /sum** (JSON no body)

```python
# POST /sum
import json
req = json.loads(REQUEST)          # REQUEST é string JSON
a = req["body"].get("a", 0)        # body já vem parseado conforme Content-Type
b = req["body"].get("b", 0)
print(json.dumps({"sum": a + b}))  # response body
```

**Opcional: metadados de resposta (status/headers)**

```python
# ResponseInfo POST /sum
import json
print(json.dumps({
  "status": 200,
  "headers": {"Content-Type": "application/json"}
}))
```

Referência de **REQUEST**, body e “ResponseInfo”: ([Jupyter Kernel Gateway][1])

---

# 3) Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Instala Kernel Gateway
RUN pip install --no-cache-dir jupyter_kernel_gateway

# Copia notebooks para a imagem
WORKDIR /app
COPY /srv/notebooks /app/notebooks

# Configurações do Kernel Gateway via env (ver doc de config)
ENV KG_API=kernel_gateway.notebook_http \
    KG_SEED_URI=/app/notebooks/api.ipynb \
    KG_PORT=8888 \
    KG_ALLOW_ORIGIN=*    \
    JUPYTER_TOKEN=       \
    JUPYTER_PASSWORD=

EXPOSE 8888

# Sobe ouvindo em 0.0.0.0
CMD ["jupyter", "kernelgateway", "--ip=0.0.0.0"]
```

Opções `KG_*` e `--KernelGatewayApp.*` documentadas aqui. ([Jupyter Kernel Gateway][2])

---

# 4) docker-compose.yml

```yaml
version: "3.8"
services:
  jkgs:
    build: .
    container_name: jkgs
    ports:
      - "8888:8888"
    environment:
      - KG_API=kernel_gateway.notebook_http
      - KG_SEED_URI=/app/notebooks/api.ipynb
      - KG_PORT=8888
      - KG_ALLOW_ORIGIN=*
    volumes:
      - /srv/notebooks:/app/notebooks:ro
    restart: unless-stopped
```

> Dica: monte `:ro` (somente leitura) para não permitir mutação do notebook em produção.

---

# 5) Subir

```bash
docker compose up -d --build
```

---

# 6) Testar os endpoints

```bash
# GET /ping
curl http://localhost:8888/ping
# -> pong

# POST /sum
curl -s -X POST http://localhost:8888/sum \
  -H "Content-Type: application/json" \
  -d '{"a": 2, "b": 5}'
# -> {"sum":7}

# Swagger gerado
curl http://localhost:8888/_api/spec/swagger.json
```

O comportamento e rotas acima são exatamente os do **modo notebook-http**. ([Jupyter Kernel Gateway][1])

---

## Escalar/otimizar

* **Pré-spawn de kernels** para paralelismo:

  ```yaml
  environment:
    - KG_PRESPAWN_COUNT=5
  ```

  (requests serão distribuídos entre 5 kernels). ([Jupyter Kernel Gateway][1])

* **Múltiplos notebooks (múltiplos serviços)**: o Kernel Gateway usa **um `seed_uri` por processo**. Para vários notebooks, suba **vários containers**, cada um com seu `KG_SEED_URI` e porta/rota próprias. ([Jupyter Kernel Gateway][2])

* **CORS/Auth/Exposição pública**: por padrão, não há autenticação “forte” no modo HTTP. Use um **reverse proxy** (Traefik/Nginx) com **auth**, **TLS** e **rate-limit**. (Você também pode ajustar `KG_ALLOW_ORIGIN`/bind IP). ([Jupyter Kernel Gateway][2])

---

## Referências oficiais

* **notebook-http (anotações, REQUEST, ResponseInfo, Swagger, prespawn)**. ([Jupyter Kernel Gateway][1])
* **Instalação/execução e flags/env `KG_*`**. ([Jupyter Kernel Gateway][3])
* **Resumo do Kernel Gateway** (mecanismo headless/API). ([Jupyter Kernel Gateway][4], [PyPI][5])

Se quiser, eu já te deixo um **`/srv/notebooks/api.ipynb` de exemplo** com essas células prontas (e um compose com Traefik) — é só me dizer se você quer expor isso em `api.codrstudio.dev` atrás de TLS.

[1]: https://jupyter-kernel-gateway.readthedocs.io/en/latest/http-mode.html "notebook-http Mode — Jupyter Kernel Gateway 3.0.1 documentation"
[2]: https://jupyter-kernel-gateway.readthedocs.io/en/latest/config-options.html?utm_source=chatgpt.com "Configuration options - Jupyter Kernel Gateway - Read the Docs"
[3]: https://jupyter-kernel-gateway.readthedocs.io/en/latest/getting-started.html?utm_source=chatgpt.com "Getting started — Jupyter Kernel Gateway 3.0.1 documentation"
[4]: https://jupyter-kernel-gateway.readthedocs.io/?utm_source=chatgpt.com "Jupyter Kernel Gateway - Read the Docs"
[5]: https://pypi.org/project/jupyter-kernel-gateway/2.0.2/?utm_source=chatgpt.com "jupyter-kernel-gateway"
