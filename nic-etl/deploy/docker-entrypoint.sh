#!/bin/bash
set -e

# Docker Entrypoint para NIC ETL Pipeline
# Este script gerencia a inicialização dos serviços no container

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função de log
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Função para aguardar serviço
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    log "Aguardando $service_name ($host:$port)..."
    
    while [ $attempt -le $max_attempts ]; do
        if timeout 1 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
            success "$service_name está disponível!"
            return 0
        fi
        
        log "Tentativa $attempt/$max_attempts - $service_name ainda não disponível..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    error "$service_name não ficou disponível após $max_attempts tentativas"
    return 1
}

# Função para verificar dependências
check_dependencies() {
    log "Verificando dependências..."
    
    # Verificar se diretórios existem
    for dir in pipeline-data logs static notebooks; do
        if [ ! -d "$dir" ]; then
            warn "Criando diretório $dir..."
            mkdir -p "$dir"
        fi
    done
    
    # Verificar permissões
    if [ ! -w pipeline-data ] || [ ! -w logs ]; then
        error "Permissões insuficientes nos diretórios de dados"
        exit 1
    fi
    
    # Verificar arquivo de configuração
    env_file=".env.${ENVIRONMENT:-development}"
    if [ ! -f "$env_file" ]; then
        warn "Arquivo $env_file não encontrado, usando configuração padrão"
        # Criar arquivo básico se não existir
        cat > "$env_file" << EOF
# Configuração padrão gerada automaticamente
ENVIRONMENT=${ENVIRONMENT:-development}
JUPYTER_PORT=5001
PROXY_PORT=5000
EOF
    fi
    
    success "Dependências verificadas"
}

# Função para inicializar aplicação
init_app() {
    log "Inicializando aplicação..."
    
    # Exportar variáveis de ambiente
    export JUPYTER_PORT=${JUPYTER_PORT:-5001}
    export PROXY_PORT=${PROXY_PORT:-5000}
    export PYTHONPATH=/app:$PYTHONPATH
    
    # Criar diretórios de dados se não existirem
    mkdir -p pipeline-data/{documents,processed,chunks,embeddings,metadata,checkpoints}
    
    # Configurar agendamento automático
    setup_scheduling
    
    success "Aplicação inicializada"
}

# Função para configurar agendamento
setup_scheduling() {
    log "Configurando sistema de agendamento..."
    
    # Iniciar cron service
    service cron start
    
    # Executar script de agendamento se existir
    if [ -f "./schedule_job.sh" ] && [ -x "./schedule_job.sh" ]; then
        log "Executando configuração de agendamento..."
        ./schedule_job.sh
        
        if [ $? -eq 0 ]; then
            success "Agendamento configurado via schedule_job.sh"
        else
            warn "Falha na execução do schedule_job.sh, tentando método alternativo..."
            setup_cron_fallback
        fi
    else
        log "schedule_job.sh não encontrado, usando configuração direta..."
        setup_cron_fallback
    fi
    
    # Verificar se crontab foi configurado
    if crontab -l >/dev/null 2>&1; then
        log "Tarefas agendadas ativas:"
        crontab -l | while read line; do
            if [[ ! "$line" =~ ^#.* ]] && [[ -n "$line" ]]; then
                log "  → $line"
            fi
        done
        success "Sistema de agendamento ativo"
    else
        warn "Nenhuma tarefa agendada encontrada"
    fi
}

# Função fallback para configuração direta do cron
setup_cron_fallback() {
    if [ -f "cron/crontab.txt" ]; then
        log "Aplicando crontab.txt diretamente..."
        crontab cron/crontab.txt
        success "Crontab aplicado diretamente"
    else
        warn "Arquivo cron/crontab.txt não encontrado"
    fi
}

# Função para executar servidor
run_server() {
    log "Iniciando NIC ETL Pipeline..."
    
    # Verificar se run-server.sh existe e é executável
    if [ ! -x "./run-server.sh" ]; then
        error "run-server.sh não encontrado ou não é executável"
        exit 1
    fi
    
    # Executar servidor
    exec ./run-server.sh
}

# Função para executar pipeline
run_pipeline() {
    log "Executando pipeline ETL..."
    
    # Verificar se Jupyter está rodando
    if ! pgrep -f "jupyter.*kernel.*gateway" > /dev/null; then
        log "Iniciando Jupyter Kernel Gateway..."
        jupyter kernelgateway --port=$JUPYTER_PORT --ip=0.0.0.0 &
        sleep 5
    fi
    
    # Executar notebook principal
    jupyter nbconvert --to notebook --execute notebooks/etl.ipynb --output-dir=pipeline-data/executions/
    
    success "Pipeline executado com sucesso"
}

# Função para gerenciar agendamento
manage_scheduling() {
    local action="${1:-status}"
    
    case "$action" in
        "setup"|"schedule")
            log "Configurando agendamento de tarefas..."
            setup_scheduling
            ;;
        "unschedule"|"remove")
            log "Removendo agendamento de tarefas..."
            if [ -f "./unschedule_job.sh" ] && [ -x "./unschedule_job.sh" ]; then
                ./unschedule_job.sh
                success "Agendamento removido via unschedule_job.sh"
            else
                crontab -r 2>/dev/null || true
                success "Crontab removido diretamente"
            fi
            ;;
        "status"|"list")
            log "Status do agendamento:"
            if crontab -l >/dev/null 2>&1; then
                crontab -l
            else
                warn "Nenhuma tarefa agendada"
            fi
            ;;
        *)
            error "Ação de agendamento inválida: $action"
            error "Use: setup, unschedule, ou status"
            exit 1
            ;;
    esac
}

# Função para mostrar ajuda
show_help() {
    cat << EOF
NIC ETL Pipeline - Docker Entrypoint

Uso: docker run nic-etl [COMANDO] [OPÇÕES]

Comandos disponíveis:
    server          Iniciar servidor web (padrão)
    pipeline        Executar pipeline ETL
    schedule [ação] Gerenciar agendamento (setup|unschedule|status)
    shell           Abrir shell interativo
    help            Mostrar esta ajuda

Variáveis de ambiente:
    ENVIRONMENT     Ambiente de execução (development, staging, production)
    JUPYTER_PORT    Porta do Jupyter Kernel Gateway (padrão: 5001)
    PROXY_PORT      Porta do proxy FastAPI (padrão: 5000)

Exemplos:
    docker run nic-etl server
    docker run nic-etl pipeline
    docker run nic-etl schedule setup
    docker run nic-etl schedule status
    docker run -it nic-etl shell
EOF
}

# Script principal
main() {
    log "=== NIC ETL Pipeline - Docker Container ==="
    log "Ambiente: ${ENVIRONMENT:-development}"
    log "Usuário: $(whoami)"
    log "Diretório: $(pwd)"
    
    # Verificar dependências
    check_dependencies
    
    # Inicializar aplicação
    init_app
    
    # Processar comando
    case "${1:-server}" in
        "server")
            log "Modo: Servidor Web"
            run_server
            ;;
        "pipeline")
            log "Modo: Execução de Pipeline"
            run_pipeline
            ;;
        "schedule")
            log "Modo: Gerenciamento de Agendamento"
            manage_scheduling "${2:-status}"
            ;;
        "shell")
            log "Modo: Shell Interativo"
            exec /bin/bash
            ;;
        "help"|"-h"|"--help")
            show_help
            exit 0
            ;;
        *)
            error "Comando desconhecido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Trap para cleanup
cleanup() {
    log "Recebido sinal de término, fazendo cleanup..."
    
    # Parar cron service
    service cron stop 2>/dev/null || true
    
    # Parar processos filhos
    pkill -P $$ 2>/dev/null || true
    
    # Aguardar processos terminarem
    wait
    
    log "Cleanup concluído"
    exit 0
}

# Registrar trap
trap cleanup SIGTERM SIGINT

# Executar função principal
main "$@"