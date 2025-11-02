#!/bin/bash

# Script para desagendar o job ETL NIC do crontab
# Remove todos os jobs do crontab atual

echo "Desagendando job ETL NIC..."

# Verifica se existem jobs no crontab
if ! crontab -l >/dev/null 2>&1; then
    echo "Nenhum job encontrado no crontab."
    exit 0
fi

# Mostra os jobs atuais
echo "Jobs atuais no crontab:"
crontab -l

echo ""
read -p "Deseja realmente remover todos os jobs do crontab? (s/n): " confirm

if [ "$confirm" = "s" ] || [ "$confirm" = "S" ]; then
    # Remove o crontab
    crontab -r
    
    if [ $? -eq 0 ]; then
        echo "Todos os jobs foram removidos do crontab com sucesso!"
    else
        echo "Erro ao remover jobs do crontab!"
        exit 1
    fi
else
    echo "Operação cancelada."
fi