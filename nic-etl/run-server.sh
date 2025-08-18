#!/bin/bash
set -euo pipefail

# opcional: carregar .env se existir
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# checagens mínimas
[ -d notebooks ] || { echo "erro: pasta notebooks ausente" | tee -a logs/rest-api.log; exit 1; }
[ -f jupyter_kernel_gateway_config.py ] || { echo "erro: config ausente" | tee -a logs/rest-api.log; exit 1; }

# mudar para diretório notebooks
cd ./notebooks

# criar diretório de logs se não existir  
mkdir -p logs

# executar jupyter kernel gateway com logs redirecionados usando tee
# capturar tanto stderr quanto stdout para pegar prints dos notebooks
exec jupyter kernelgateway --config=../jupyter_kernel_gateway_config.py --debug 2>&1 | tee -a logs/rest-api.log
