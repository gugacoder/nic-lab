#!/bin/bash
# Production Deployment Script for NIC ETL Pipeline

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

echo "🚀 NIC ETL Pipeline Deployment Script"
echo "======================================"
echo "Environment: $ENVIRONMENT"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Validate environment
case $ENVIRONMENT in
    production|staging|development)
        echo "✅ Valid environment: $ENVIRONMENT"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "Usage: $0 [production|staging|development]"
        exit 1
        ;;
esac

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi
echo "✅ Docker is available"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed or not in PATH"
    exit 1
fi
echo "✅ Docker Compose is available"

# Check environment file
ENV_FILE="$SCRIPT_DIR/environments/.env.$ENVIRONMENT"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    exit 1
fi
echo "✅ Environment file found: $ENV_FILE"

# Copy environment file
cp "$ENV_FILE" "$SCRIPT_DIR/.env"
echo "✅ Environment configuration loaded"

# Validate required secrets
echo "🔐 Validating secrets..."

# Check GitLab token
if grep -q "your_.*_gitlab_token_here" "$ENV_FILE"; then
    echo "⚠️  WARNING: GitLab token not configured in $ENV_FILE"
    echo "   Please update GITLAB_ACCESS_TOKEN with a valid token"
fi

# Check Qdrant API key
if grep -q "your_.*_qdrant_api_key_here" "$ENV_FILE"; then
    echo "⚠️  WARNING: Qdrant API key not configured in $ENV_FILE"
    echo "   Please update QDRANT_API_KEY with a valid key"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/cache"
mkdir -p "$SCRIPT_DIR/data"
echo "✅ Directories created"

# Build and deploy
echo "🔨 Building and deploying containers..."

cd "$SCRIPT_DIR"

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose pull

# Build application image
echo "🏗️  Building application image..."
docker-compose build nic-etl-pipeline

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "💓 Performing health checks..."

# Check if containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Some containers failed to start"
    docker-compose logs
    exit 1
fi

# Check health endpoint
echo "🔍 Checking health endpoint..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Health check passed"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Health check failed after 10 attempts"
        docker-compose logs nic-etl-pipeline
        exit 1
    fi
    echo "⏳ Waiting for health check... ($i/10)"
    sleep 10
done

# Display service information
echo ""
echo "🎉 Deployment completed successfully!"
echo "====================================="
echo ""
echo "📊 Service URLs:"
echo "  Jupyter Lab:    http://localhost:8888"
echo "  Health Check:   http://localhost:8000/health"
echo "  Status Check:   http://localhost:8000/status"
echo "  Metrics:        http://localhost:8000/metrics"
echo "  Prometheus:     http://localhost:9090"
echo "  Grafana:        http://localhost:3000"
echo ""
echo "🔧 Management Commands:"
echo "  View logs:      docker-compose logs -f"
echo "  Stop services:  docker-compose down"
echo "  Restart:        docker-compose restart"
echo "  Update:         $0 $ENVIRONMENT"
echo ""
echo "📝 Configuration:"
echo "  Environment:    $ENVIRONMENT"
echo "  Config file:    $ENV_FILE"
echo "  Log directory:  $SCRIPT_DIR/logs"
echo "  Cache directory: $SCRIPT_DIR/cache"
echo ""

# Show container status
echo "📋 Container Status:"
docker-compose ps

echo ""
echo "✅ NIC ETL Pipeline is now running!"