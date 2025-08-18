# Dockerfile para NIC ETL Pipeline
FROM python:3.11-slim

# Metadados
LABEL maintainer="NIC Team"
LABEL description="NIC ETL Pipeline - Processamento de documentos com embeddings"
LABEL version="1.0.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_ENABLE_LAB=yes \
    JUPYTER_PORT=5001 \
    PROXY_PORT=5000

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cron \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN groupadd -r nicuser && useradd -r -g nicuser nicuser

# Configurar diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro para cache de layers
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p pipeline-data/documents \
             pipeline-data/processed \
             pipeline-data/chunks \
             pipeline-data/embeddings \
             pipeline-data/metadata \
             pipeline-data/checkpoints \
             logs \
             static/assets

# Ajustar permissões
RUN chown -R nicuser:nicuser /app && \
    chmod +x run-server.sh && \
    chmod +x deploy/docker-entrypoint.sh && \
    chmod +x schedule_job.sh && \
    chmod +x unschedule_job.sh

# Criar volumes para dados persistentes
VOLUME ["/app/pipeline-data", "/app/logs"]

# Expor apenas porta pública
EXPOSE 5000

# Usar usuário não-root
USER nicuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Ponto de entrada
ENTRYPOINT ["./deploy/docker-entrypoint.sh"]
CMD ["server"]