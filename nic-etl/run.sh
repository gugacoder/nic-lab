#!/bin/bash
set -euo pipefail

# opcional: carregar .env se existir
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# checagens mínimas
[ -d notebooks ] || { echo "erro: pasta notebooks ausente"; exit 1; }
[ -f jupyter_kernel_gateway_config.py ] || { echo "erro: config ausente"; exit 1; }

exec jupyter kernelgateway --config=jupyter_kernel_gateway_config.py
