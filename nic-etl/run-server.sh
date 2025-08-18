#!/bin/bash
set -euo pipefail

# Script para executar o NIC Lab com FastAPI Proxy + Jupyter Kernel Gateway
# O proxy serve arquivos estáticos e faz proxy de /api/* para o Kernel Gateway

echo "🧠 Iniciando NIC Lab..."

# Carregar .env se existir
if [ -f .env ]; then 
    echo "📄 Carregando configurações do .env"
    set -a
    source ./.env
    set +a
fi

# Verificações básicas
echo "🔍 Verificando dependências..."

if [ ! -d notebooks ]; then
    echo "❌ Erro: Diretório notebooks não encontrado"
    exit 1
fi

if [ ! -f src/proxy.py ]; then
    echo "❌ Erro: FastAPI proxy não encontrado em src/proxy.py"
    exit 1
fi

if [ ! -d static ]; then
    echo "❌ Erro: Diretório static não encontrado"
    exit 1
fi

# Verificar se as portas estão livres
check_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        echo "❌ Erro: Porta $port já está em uso"
        exit 1
    fi
}

check_port 5000  # FastAPI Proxy (porta pública)
check_port 5001  # Kernel Gateway (porta interna)

# Criar diretórios necessários
mkdir -p logs
mkdir -p static/assets/{css,js}

# Função para cleanup ao sair
cleanup() {
    echo ""
    echo "🛑 Parando serviços..."
    
    if [ ! -z "${KG_PID:-}" ]; then
        echo "🔌 Parando Jupyter Kernel Gateway (PID: $KG_PID)"
        kill $KG_PID 2>/dev/null || true
        wait $KG_PID 2>/dev/null || true
    fi
    
    echo "✅ Cleanup concluído"
    exit 0
}

# Configurar trap para cleanup
trap cleanup SIGINT SIGTERM EXIT

# 1) Iniciar Jupyter Kernel Gateway em background (porta interna 5001)
echo "🚀 Iniciando Jupyter Kernel Gateway na porta 5001..."

# Usar EXATAMENTE a mesma configuração do run-server.sh
cd notebooks
mkdir -p logs

# Executar com a mesma config, apenas overridando a porta para uso interno
nohup jupyter kernelgateway \
    --config=../jupyter_kernel_gateway_config.py \
    --debug \
    --KernelGatewayApp.port=5001 \
    --KernelGatewayApp.ip=127.0.0.1 \
    2>&1 | tee -a logs/kernel-gateway-proxy.log &

KG_PID=$!
cd ..

echo "📋 Kernel Gateway iniciado (PID: $KG_PID)"

# Aguardar Kernel Gateway ficar pronto
echo "⏳ Aguardando Kernel Gateway ficar disponível..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:5001/api/v1 > /dev/null 2>&1; then
        echo "✅ Kernel Gateway está respondendo"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ Timeout: Kernel Gateway não respondeu em 30 segundos"
        echo "📋 Logs do Kernel Gateway:"
        tail -20 logs/kernel-gateway.log
        exit 1
    fi
    
    sleep 1
done

# 2) Iniciar FastAPI Proxy (porta pública 5000)
echo "🌐 Iniciando FastAPI Proxy na porta 5000..."
echo "🔗 Acesse: http://localhost:5000"
echo ""
echo "🎯 Endpoints disponíveis:"
echo "   📱 Website:    http://localhost:5000/"
echo "   🔗 API:        http://localhost:5000/api/v1"
echo "   📊 Status:     http://localhost:5000/status"
echo "   📚 Docs:       http://localhost:5000/docs"
echo ""
echo "💡 Pressione Ctrl+C para parar todos os serviços"
echo "=" * 60

# Executar proxy em foreground (para manter o script ativo)
python src/proxy.py