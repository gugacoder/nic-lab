#!/bin/bash
set -e

# Script de Deploy para NIC ETL Pipeline
# Automatiza build, deploy e gerenciamento do container Docker

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Variáveis
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="nic-etl"
CONTAINER_NAME="nic-etl-pipeline"
DEFAULT_ENV="development"

# Funções de log
log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Função de ajuda
show_help() {
    cat << EOF
NIC ETL Pipeline - Script de Deploy

Uso: $0 [OPÇÕES] [COMANDO]

Comandos:
    build       Construir imagem Docker
    up          Iniciar serviços (build + run)
    down        Parar e remover containers
    restart     Reiniciar serviços
    logs        Mostrar logs dos containers
    status      Mostrar status dos serviços
    clean       Limpar imagens e volumes não utilizados
    shell       Abrir shell no container
    pipeline    Executar pipeline ETL
    schedule    Configurar agendamento de tarefas
    unschedule  Remover agendamento de tarefas

Opções:
    -e, --env ENV       Ambiente (development|staging|production)
    -d, --detach        Executar em background
    -f, --force         Forçar rebuild da imagem
    -v, --verbose       Saída detalhada
    -h, --help          Mostrar esta ajuda

Exemplos:
    $0 build
    $0 up --env production --detach
    $0 logs
    $0 schedule
    $0 unschedule
    $0 clean
    $0 shell
EOF
}

# Parsear argumentos
parse_args() {
    ENVIRONMENT="$DEFAULT_ENV"
    DETACH=""
    FORCE_BUILD=""
    VERBOSE=""
    COMMAND=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--detach)
                DETACH="-d"
                shift
                ;;
            -f|--force)
                FORCE_BUILD="--no-cache"
                shift
                ;;
            -v|--verbose)
                VERBOSE="--verbose"
                set -x
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            build|up|down|restart|logs|status|clean|shell|pipeline|schedule|unschedule)
                COMMAND="$1"
                shift
                ;;
            *)
                error "Opção desconhecida: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [ -z "$COMMAND" ]; then
        COMMAND="up"
    fi
}

# Verificar pré-requisitos
check_prerequisites() {
    log "Verificando pré-requisitos..."

    # Docker
    if ! command -v docker &> /dev/null; then
        error "Docker não está instalado"
        exit 1
    fi

    # Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose não está disponível"
        exit 1
    fi

    # Arquivo de ambiente
    env_file="$PROJECT_DIR/.env.$ENVIRONMENT"
    if [ ! -f "$env_file" ]; then
        warn "Arquivo $env_file não encontrado"
        log "Criando arquivo de configuração básico..."
        cat > "$env_file" << EOF
# Configuração para ambiente: $ENVIRONMENT
ENVIRONMENT=$ENVIRONMENT
JUPYTER_PORT=5001
PROXY_PORT=5000

# GitLab Configuration
GITLAB_URL=http://gitlab.processa.info
GITLAB_TOKEN=your_token_here
GITLAB_REPO=nic/documentacao/base-de-conhecimento
GITLAB_FOLDER=30-Aprovados

# QDrant Configuration
QDRANT_URL=https://qdrant.codrstudio.dev/
QDRANT_API_KEY=your_key_here
QDRANT_COLLECTION=nic_${ENVIRONMENT}
EOF
        warn "Edite o arquivo $env_file com suas configurações"
    fi

    success "Pré-requisitos verificados"
}

# Criar diretórios necessários
setup_directories() {
    log "Configurando diretórios..."
    
    cd "$SCRIPT_DIR"
    
    # Criar diretórios de dados
    mkdir -p data/{pipeline-data,logs}
    mkdir -p data/pipeline-data/{documents,processed,chunks,embeddings,metadata,checkpoints}
    
    # Ajustar permissões
    chmod -R 755 data/
    
    success "Diretórios configurados"
}

# Construir imagem
build_image() {
    log "Construindo imagem Docker..."
    
    cd "$SCRIPT_DIR"
    
    local build_args=""
    if [ "$VERBOSE" ]; then
        build_args="--progress=plain"
    fi
    
    docker compose build $FORCE_BUILD $build_args nic-etl
    
    success "Imagem construída: $IMAGE_NAME"
}

# Iniciar serviços
start_services() {
    log "Iniciando serviços..."
    
    cd "$SCRIPT_DIR"
    
    export ENVIRONMENT
    
    if [ "$DETACH" ]; then
        docker compose up $DETACH
        success "Serviços iniciados em background"
        log "Acesse a aplicação em: http://localhost:5000"
        log "Use '$0 logs' para acompanhar os logs"
    else
        log "Iniciando em modo interativo (Ctrl+C para parar)..."
        docker compose up
    fi
}

# Parar serviços
stop_services() {
    log "Parando serviços..."
    
    cd "$SCRIPT_DIR"
    
    docker compose down
    
    success "Serviços parados"
}

# Reiniciar serviços
restart_services() {
    log "Reiniciando serviços..."
    stop_services
    start_services
}

# Mostrar logs
show_logs() {
    log "Mostrando logs dos serviços..."
    
    cd "$SCRIPT_DIR"
    
    docker compose logs -f --tail=100
}

# Mostrar status
show_status() {
    log "Status dos serviços:"
    
    cd "$SCRIPT_DIR"
    
    docker compose ps
    
    echo ""
    log "Imagens Docker:"
    docker images | grep -E "(nic-etl|qdrant)" || echo "Nenhuma imagem encontrada"
    
    echo ""
    log "Volumes Docker:"
    docker volume ls | grep nic || echo "Nenhum volume encontrado"
}

# Limpeza
cleanup() {
    log "Limpando recursos não utilizados..."
    
    # Parar containers
    docker compose down 2>/dev/null || true
    
    # Remover imagens não utilizadas
    docker image prune -f
    
    # Remover volumes órfãos
    docker volume prune -f
    
    # Remover networks não utilizadas
    docker network prune -f
    
    success "Limpeza concluída"
}

# Shell interativo
open_shell() {
    log "Abrindo shell no container..."
    
    cd "$SCRIPT_DIR"
    
    if ! docker compose ps | grep -q "nic-etl-pipeline.*Up"; then
        log "Container não está rodando, iniciando..."
        docker compose up -d nic-etl
        sleep 5
    fi
    
    docker compose exec nic-etl /bin/bash
}

# Executar pipeline
run_pipeline() {
    log "Executando pipeline ETL..."
    
    cd "$SCRIPT_DIR"
    
    if ! docker compose ps | grep -q "nic-etl-pipeline.*Up"; then
        log "Container não está rodando, iniciando..."
        docker compose up -d nic-etl
        sleep 10
    fi
    
    docker compose exec nic-etl bash -c "cd /app && python -m jupyter nbconvert --to notebook --execute notebooks/etl.ipynb"
    
    success "Pipeline executado"
}

# Configurar agendamento
setup_scheduling() {
    log "Configurando agendamento de tarefas..."
    
    cd "$SCRIPT_DIR"
    
    if ! docker compose ps | grep -q "nic-etl-pipeline.*Up"; then
        log "Container não está rodando, iniciando..."
        docker compose up -d nic-etl
        sleep 5
    fi
    
    docker compose exec nic-etl schedule setup
    
    success "Agendamento configurado"
}

# Remover agendamento
remove_scheduling() {
    log "Removendo agendamento de tarefas..."
    
    cd "$SCRIPT_DIR"
    
    if ! docker compose ps | grep -q "nic-etl-pipeline.*Up"; then
        log "Container não está rodando, iniciando..."
        docker compose up -d nic-etl
        sleep 5
    fi
    
    docker compose exec nic-etl schedule unschedule
    
    success "Agendamento removido"
}

# Função principal
main() {
    log "=== NIC ETL Pipeline - Deploy Script ==="
    log "Ambiente: $ENVIRONMENT"
    log "Comando: $COMMAND"
    
    # Verificar pré-requisitos
    check_prerequisites
    
    # Configurar diretórios
    setup_directories
    
    # Executar comando
    case "$COMMAND" in
        "build")
            build_image
            ;;
        "up")
            build_image
            start_services
            ;;
        "down")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "clean")
            cleanup
            ;;
        "shell")
            open_shell
            ;;
        "pipeline")
            run_pipeline
            ;;
        "schedule")
            setup_scheduling
            ;;
        "unschedule")
            remove_scheduling
            ;;
        *)
            error "Comando desconhecido: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Parsear argumentos e executar
parse_args "$@"
main