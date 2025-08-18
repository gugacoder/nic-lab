#!/bin/bash

# NIC ETL - Jupyter Kernel Gateway Launcher
# Executa os notebooks como API REST na porta 8000

set -e  # Exit on any error

echo "🚀 NIC ETL - Starting Jupyter Kernel Gateway"
echo "================================================"

# Verificar se o diretório de notebooks existe
if [ ! -d "notebooks" ]; then
    echo "❌ Error: notebooks directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Verificar se os notebooks principais existem
REQUIRED_NOTEBOOKS=(
    "notebooks/nic-etl-api.ipynb"
    "notebooks/nic-etl.ipynb" 
    "notebooks/nic-etl-report.ipynb"
)

for notebook in "${REQUIRED_NOTEBOOKS[@]}"; do
    if [ ! -f "$notebook" ]; then
        echo "❌ Error: Required notebook not found: $notebook"
        exit 1
    fi
done

# Verificar se existe arquivo de configuração
if [ ! -f "jupyter_kernel_gateway_config.py" ]; then
    echo "❌ Error: Configuration file not found: jupyter_kernel_gateway_config.py"
    exit 1
fi

# Carregar variáveis de ambiente se existir arquivo .env
if [ -f ".env" ]; then
    echo "📝 Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
elif [ -f ".env.development" ]; then
    echo "📝 Loading environment variables from .env.development"
    export $(grep -v '^#' .env.development | xargs)
else
    echo "⚠️  Warning: No .env file found, using default configuration"
fi

# Verificar se jupyter-kernel-gateway está instalado
if ! command -v jupyter-kernelgateway &> /dev/null; then
    echo "❌ Error: jupyter-kernel-gateway is not installed"
    echo "Please install it with: pip install jupyter_kernel_gateway"
    exit 1
fi

# Criar diretório de logs se não existir
mkdir -p logs

# Função para cleanup ao receber SIGINT (Ctrl+C)
cleanup() {
    echo ""
    echo "🛑 Shutting down Jupyter Kernel Gateway..."
    exit 0
}
trap cleanup SIGINT

# Exibir informações de configuração
echo "📋 Configuration:"
echo "  • Port: 8000"
echo "  • Host: 0.0.0.0 (all interfaces)"
echo "  • API Type: notebook_http"
echo "  • Working Directory: $(pwd)/notebooks"
echo "  • Config File: jupyter_kernel_gateway_config.py"
echo ""

echo "🌐 Available APIs:"
echo "  • Documentation:     GET  http://localhost:8000/nic/v1"
echo "  • Execute Pipeline:   POST http://localhost:8000/nic/v1/pipelines/gitlab-qdrant/run"
echo "  • Last Report:        GET  http://localhost:8000/nic/v1/pipelines/gitlab-qdrant/runs/last"
echo "  • Summary:            GET  http://localhost:8000/nic/v1/pipelines/gitlab-qdrant/runs/last/summary"
echo "  • Stages:             GET  http://localhost:8000/nic/v1/pipelines/gitlab-qdrant/runs/last/stages"
echo "  • Health Check:       GET  http://localhost:8000/nic/v1/pipelines/gitlab-qdrant/health"
echo ""

echo "🔧 Environment Variables:"
echo "  • GITLAB_URL: ${GITLAB_URL:-'not set'}"
echo "  • GITLAB_ACCESS_TOKEN: ${GITLAB_ACCESS_TOKEN:+***set***}"
echo "  • QDRANT_URL: ${QDRANT_URL:-'not set'}"
echo "  • QDRANT_API_KEY: ${QDRANT_API_KEY:+***set***}"
echo "  • ENVIRONMENT: ${ENVIRONMENT:-'development'}"
echo ""

echo "⏰ Starting server..."
echo "💡 Press Ctrl+C to stop the server"
echo "================================================"

# Iniciar o Jupyter Kernel Gateway com configuração personalizada
jupyter kernelgateway \
    --config=jupyter_kernel_gateway_config.py \
    --KernelGatewayApp.seed_uri=./notebooks/nic-etl-api.ipynb \
    2>&1 | tee logs/jupyter-kernel-gateway.log