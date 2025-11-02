#!/bin/bash
set -euo pipefail

# Script para executar o pipeline ETL diretamente (para cron ou execução manual)
# Usa o mesmo sistema de trava do REST API, evitando conflitos

# Carregar .env se existir
if [ -f .env ]; then 
    set -a
    source ./.env
    set +a
fi

# Verificar se notebooks existe
if [ ! -d notebooks ]; then
    echo "❌ Erro: Diretório notebooks não encontrado"
    exit 1
fi

# Mudar para diretório notebooks
cd notebooks

# Verificar se wrapper existe
if [ ! -f src/run_notebook_wrapper.py ]; then
    echo "❌ Erro: Wrapper script não encontrado em src/run_notebook_wrapper.py"
    exit 1
fi

# Criar diretório de logs se não existir
mkdir -p logs

# Timestamp para log
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Iniciando execução do pipeline ETL via cron/manual..."

# Executar o pipeline usando o wrapper
# O wrapper gerencia a trava automaticamente
python src/run_notebook_wrapper.py etl.ipynb --log-file logs/etl-cron.log

# Verificar código de saída
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] ✅ Pipeline ETL executado com sucesso"
else
    echo "[$TIMESTAMP] ❌ Pipeline ETL falhou com código: $EXIT_CODE"
fi

exit $EXIT_CODE