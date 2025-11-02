#!/bin/bash
set -euo pipefail

# ASCII Art para NIC ETL
echo "
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███╗   ██╗██╗ ██████╗    ██╗      █████╗ ██████╗                        ║
║   ████╗  ██║██║██╔════╝    ██║     ██╔══██╗██╔══██╗                       ║
║   ██╔██╗ ██║██║██║         ██║     ███████║██████╔╝                       ║
║   ██║╚██╗██║██║██║         ██║     ██╔══██║██╔══██╗                       ║
║   ██║ ╚████║██║╚██████╗    ███████╗██║  ██║██████╔╝                       ║
║   ╚═╝  ╚═══╝╚═╝ ╚═════╝    ╚══════╝╚═╝  ╚═╝╚═════╝                        ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │  ◊ Núcleo de Inteligência e Conhecimento                            │  ║
║  │  ◊ Laboratório de Pesquisas em IA                                   │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║   ░░▒▒▓▓█ [ Knowledge Base ] █▓▓▒▒░░     ░░▒▒▓▓█ [ IA Powered ] █▓▓▒▒░░   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"

# Script para executar o NIC Lab com FastAPI Proxy + Jupyter Kernel Gateway
# O proxy serve arquivos estáticos e faz proxy de /api/* para o Kernel Gateway

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
    if nc -z 127.0.0.1 $port 2>/dev/null; then
        echo "❌ Erro: Porta $port já está em uso"
        echo "💡 Use 'pkill -f jupyter' para parar processos Jupyter"
        exit 1
    fi
}

check_port 5000  # FastAPI Proxy (porta pública)
check_port 5001  # Kernel Gateway ETL (porta interna)
check_port 5002  # Kernel Gateway RAG (porta interna)

# Criar diretórios necessários
mkdir -p logs
mkdir -p static/assets/{css,js}

# Função para cleanup ao sair
cleanup() {
    echo ""
    echo "🛑 Parando serviços..."
    
    if [ ! -z "${KG_ETL_PID:-}" ]; then
        echo "🔌 Parando Jupyter Kernel Gateway ETL (PID: $KG_ETL_PID)"
        kill $KG_ETL_PID 2>/dev/null || true
        wait $KG_ETL_PID 2>/dev/null || true
    fi
    
    if [ ! -z "${KG_RAG_PID:-}" ]; then
        echo "🔌 Parando Jupyter Kernel Gateway RAG (PID: $KG_RAG_PID)"
        kill $KG_RAG_PID 2>/dev/null || true
        wait $KG_RAG_PID 2>/dev/null || true
    fi
    
    echo "✅ Cleanup concluído"
    exit 0
}

# Configurar trap para cleanup
trap cleanup SIGINT SIGTERM EXIT

# 1) Iniciar Jupyter Kernel Gateway ETL em background (porta interna 5001)
echo "🚀 Iniciando Jupyter Kernel Gateway ETL na porta 5001..."

cd notebooks
mkdir -p logs

# Executar ETL Gateway na porta 5001
nohup jupyter kernelgateway \
    --config=../jupyter_kernel_gateway_config.py \
    --debug \
    2>&1 | tee -a logs/kernel-gateway-etl.log &

KG_ETL_PID=$!

echo "📋 Kernel Gateway ETL iniciado (PID: $KG_ETL_PID)"

# 2) Iniciar Jupyter Kernel Gateway RAG em background (porta interna 5002)
echo "🧠 Iniciando Jupyter Kernel Gateway RAG na porta 5002..."

# Executar RAG Gateway na porta 5002
nohup jupyter kernelgateway \
    --config=../jupyter_kernel_gateway_rag_config.py \
    --debug \
    2>&1 | tee -a logs/kernel-gateway-rag.log &

KG_RAG_PID=$!
cd ..

echo "📋 Kernel Gateway RAG iniciado (PID: $KG_RAG_PID)"

# 3) Aguardar ambos os Kernel Gateways ficarem prontos
echo "⏳ Aguardando Kernel Gateways ficarem disponíveis..."

# Aguardar ETL Gateway
for i in {1..30}; do
    if curl -s http://127.0.0.1:5001/api/v1 > /dev/null 2>&1; then
        echo "✅ Kernel Gateway ETL está respondendo"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ Timeout: Kernel Gateway ETL não respondeu em 30 segundos"
        echo "📋 Logs do ETL Gateway:"
        tail -20 notebooks/logs/kernel-gateway-etl.log
        exit 1
    fi
    
    sleep 1
done

# Aguardar RAG Gateway (comentado - não é necessário esperar)
for i in {1..30}; do
    if curl -s http://127.0.0.1:5002/api/v1/search/stats > /dev/null 2>&1; then
        echo "✅ Kernel Gateway RAG está respondendo"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ Timeout: Kernel Gateway RAG não respondeu em 30 segundos"
        echo "📋 Logs do RAG Gateway:"
        tail -20 notebooks/logs/kernel-gateway-rag.log
        exit 1
    fi
    
    sleep 1
done

echo "⏳ RAG Gateway iniciando em background..."

# 4) Iniciar FastAPI Proxy (porta pública 5000)
echo "🌐 Iniciando FastAPI Proxy na porta 5000..."
echo "🔗 Acesse: http://localhost:5000"
echo ""
echo "🎯 Endpoints disponíveis:"
echo "   📱 Website:    http://localhost:5000/"
echo "   🔗 ETL API:    http://localhost:5000/api/v1"
echo "   🧠 RAG API:    http://localhost:5000/rag/v1/search"
echo "   📊 Status:     http://localhost:5000/health"
echo "   📚 Docs:       http://localhost:5000/docs"
echo ""
echo "🧪 Testes rápidos:"
echo "   curl http://localhost:5000/health"
echo "   curl http://localhost:5000/api/v1"
echo "   curl http://localhost:5000/rag/v1/search/test"
echo ""
echo "💡 Pressione Ctrl+C para parar todos os serviços"
echo "=" * 60

# Executar proxy em foreground (para manter o script ativo)
python src/proxy.py