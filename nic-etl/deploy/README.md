# Deploy - NIC ETL Pipeline

Configuração Docker completa para deployment do NIC ETL Pipeline em diferentes ambientes.

## 🏗️ Arquitetura Docker

### Estrutura de Serviços
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Nginx        │    │   NIC ETL       │    │    QDrant       │
│   (Port 80)     │───▶│   (Port 5000)   │───▶│   (Port 6333)   │
│   Proxy Web     │    │   FastAPI +     │    │   Vector DB     │
│                 │    │   Jupyter       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                        │                        │
       │                        │                        │
   Público (HTTP)         Interno (Docker)         Interno (Docker)
```

### Exposição de Portas
- **Porta 5000**: Única porta exposta publicamente (FastAPI Proxy)
- **Porta 5001**: Jupyter Kernel Gateway (interno ao container)
- **Porta 6333**: QDrant (interno à rede Docker)

## 🚀 Quick Start

### 1. Deploy Rápido
```bash
# Build e start em modo development
./deploy/deploy.sh up

# Ou especificar ambiente
./deploy/deploy.sh up --env production --detach
```

### 2. Acesso à Aplicação
```bash
# Web Interface
http://localhost:5000

# API Documentation  
http://localhost:5000/api/v1

# Health Check
http://localhost:5000/api/v1/health
```

## 📋 Comandos Disponíveis

### Script de Deploy
```bash
# Construir imagem
./deploy/deploy.sh build

# Iniciar serviços
./deploy/deploy.sh up [--env ENV] [--detach]

# Parar serviços
./deploy/deploy.sh down

# Reiniciar
./deploy/deploy.sh restart

# Logs
./deploy/deploy.sh logs

# Status dos serviços
./deploy/deploy.sh status

# Shell interativo
./deploy/deploy.sh shell

# Executar pipeline
./deploy/deploy.sh pipeline

# Limpeza
./deploy/deploy.sh clean
```

### Docker Compose Direto
```bash
cd deploy/

# Iniciar (development)
ENVIRONMENT=development docker compose up

# Iniciar (production)
ENVIRONMENT=production docker compose up -d

# Parar
docker compose down

# Logs
docker compose logs -f nic-etl
```

## ⚙️ Configuração

### Ambientes
Crie arquivos de configuração para cada ambiente:

```bash
# .env.development
ENVIRONMENT=development
GITLAB_URL=http://gitlab.processa.info
GITLAB_TOKEN=development_token
QDRANT_URL=https://qdrant-dev.example.com
QDRANT_COLLECTION=nic_dev

# .env.staging  
ENVIRONMENT=staging
GITLAB_URL=http://gitlab.processa.info
GITLAB_TOKEN=staging_token
QDRANT_URL=https://qdrant-staging.example.com
QDRANT_COLLECTION=nic_staging

# .env.production
ENVIRONMENT=production
GITLAB_URL=http://gitlab.processa.info
GITLAB_TOKEN=production_token
QDRANT_URL=https://qdrant.example.com
QDRANT_COLLECTION=nic_production
```

### Volumes Persistentes
```yaml
volumes:
  - pipeline-data:/app/pipeline-data    # Dados do pipeline
  - logs:/app/logs                      # Logs da aplicação
  - static:/app/static:ro              # Assets estáticos
  - notebooks:/app/notebooks:ro        # Notebooks (read-only)
```

## 🔧 Estrutura de Arquivos

```
deploy/
├── Dockerfile              # Imagem principal do NIC ETL
├── docker-compose.yml      # Orquestração de serviços
├── docker-entrypoint.sh    # Script de inicialização
├── .dockerignore           # Arquivos ignorados no build
├── deploy.sh               # Script de automação
├── nginx.conf              # Configuração do proxy (produção)
├── README.md               # Esta documentação
└── data/                   # Dados persistentes (criado em runtime)
    ├── pipeline-data/      # Dados do pipeline ETL
    └── logs/               # Logs da aplicação
```

## 🛡️ Segurança

### Container Security
- **Usuário não-root**: Container roda com usuário `nicuser`
- **Portas internas**: Apenas porta 5000 exposta publicamente
- **Volumes read-only**: Código da aplicação é read-only
- **Health checks**: Monitoramento automático de saúde

### Network Security
- **Rede interna**: Comunicação entre serviços isolada
- **Proxy reverso**: Nginx filtra e gerencia requests
- **Rate limiting**: Proteção contra abuse de API
- **Security headers**: Headers de segurança aplicados

## 📊 Monitoramento

### Health Checks
```bash
# Container health
docker compose ps

# Application health
curl http://localhost:5000/api/v1/health

# Pipeline status
curl http://localhost:5000/api/v1/pipelines/gitlab-qdrant/runs/last
```

### Logs
```bash
# Logs da aplicação
docker compose logs -f nic-etl

# Logs do sistema
docker compose exec nic-etl tail -f /app/logs/app.log

# Logs do pipeline
docker compose exec nic-etl ls -la /app/pipeline-data/metadata/
```

## 🚀 Deploy em Produção

### 1. Preparação
```bash
# Criar arquivo de configuração de produção
cp .env.development .env.production
# Editar .env.production com credenciais reais

# Verificar recursos do servidor
df -h                    # Espaço em disco
free -h                  # Memória disponível
docker --version         # Docker instalado
```

### 2. Deploy com Nginx
```bash
# Ativar perfil de produção (inclui Nginx)
ENVIRONMENT=production docker compose --profile production up -d

# Verificar serviços
docker compose ps
curl http://localhost/api/v1/health
```

### 3. SSL/HTTPS (Opcional)
```bash
# Gerar certificados SSL
mkdir -p deploy/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout deploy/ssl/key.pem \
    -out deploy/ssl/cert.pem

# Reiniciar com SSL
docker compose restart nginx
```

## 🔄 Backup e Restauração

### Backup
```bash
# Backup dos dados
docker run --rm -v nic-etl_pipeline-data:/data -v $(pwd):/backup \
    alpine tar czf /backup/pipeline-data-backup.tar.gz /data

# Backup do QDrant
docker compose exec qdrant curl -X POST "http://localhost:6333/collections/nic_production/snapshots"
```

### Restauração
```bash
# Restaurar dados
docker run --rm -v nic-etl_pipeline-data:/data -v $(pwd):/backup \
    alpine tar xzf /backup/pipeline-data-backup.tar.gz -C /

# Reiniciar serviços
docker compose restart
```

## 🐛 Troubleshooting

### Problemas Comuns

**Container não inicia**
```bash
# Verificar logs
docker compose logs nic-etl

# Verificar configuração
docker compose config

# Testar build
docker compose build --no-cache nic-etl
```

**Porta em uso**
```bash
# Verificar portas
netstat -tlnp | grep :5000

# Parar serviços conflitantes
sudo systemctl stop nginx  # Se nginx local rodando
```

**Problemas de permissão**
```bash
# Ajustar permissões dos dados
sudo chown -R 1000:1000 deploy/data/

# Verificar permissões do script
chmod +x deploy/deploy.sh deploy/docker-entrypoint.sh
```

**Memória insuficiente**
```bash
# Monitorar uso
docker stats

# Ajustar limites no docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G  # Reduzir se necessário
```

### Debug Avançado
```bash
# Shell no container
./deploy/deploy.sh shell

# Executar comandos específicos
docker compose exec nic-etl python -c "import torch; print(torch.cuda.is_available())"

# Verificar variáveis de ambiente
docker compose exec nic-etl env | grep -E "(JUPYTER|PROXY|GITLAB|QDRANT)"

# Testar conectividade
docker compose exec nic-etl curl -f http://localhost:5001/api/status
```

## 📈 Performance

### Otimizações Recomendadas

**Recursos do Container**
- **CPU**: Mínimo 2 cores, recomendado 4+ cores
- **Memória**: Mínimo 4GB, recomendado 8GB+ 
- **Disco**: SSD com 50GB+ livres

**Configurações Docker**
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

**Sistema Operacional**
```bash
# Aumentar limites de arquivo
echo "fs.file-max = 65536" >> /etc/sysctl.conf

# Otimizar memória compartilhada
echo "kernel.shmmax = 68719476736" >> /etc/sysctl.conf

# Aplicar mudanças
sysctl -p
```