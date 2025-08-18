#!/bin/bash

# Script para agendar o job ETL NIC no crontab
# Aplica o arquivo de configuração do crontab

echo "Agendando job ETL NIC..."

# Verifica se o arquivo crontab.txt existe
if [ ! -f "cron/crontab.txt" ]; then
    echo "Erro: Arquivo cron/crontab.txt não encontrado!"
    exit 1
fi

# Aplica o crontab
crontab cron/crontab.txt

if [ $? -eq 0 ]; then
    echo "Job ETL agendado com sucesso!"
    echo "O pipeline será executado de hora em hora."
    echo ""
    echo "Para verificar os jobs agendados, execute:"
    echo "crontab -l"
else
    echo "Erro ao agendar o job!"
    exit 1
fi