#!/bin/bash
# Production startup script for NIC ETL Pipeline

set -e

echo "🚀 Starting NIC ETL Pipeline in ${ENVIRONMENT:-production} mode..."

# Validate required environment variables
if [ -z "$GITLAB_ACCESS_TOKEN" ]; then
    echo "❌ ERROR: GITLAB_ACCESS_TOKEN is required"
    exit 1
fi

if [ -z "$QDRANT_API_KEY" ]; then
    echo "❌ ERROR: QDRANT_API_KEY is required"
    exit 1
fi

# Create log directory
mkdir -p /app/logs

# Start health check endpoint in background
echo "🔍 Starting health check endpoint..."
python health_check.py &
HEALTH_PID=$!

# Run configuration validation
echo "⚙️  Validating configuration..."
python -c "
from modules.configuration_management import create_configuration_manager
try:
    config_manager = create_configuration_manager(environment='${ENVIRONMENT:-production}')
    print('✅ Configuration validation passed')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    exit(1)
"

# Run basic system tests
echo "🧪 Running system health checks..."
python run_tests.py --no-coverage --quiet

# Start Jupyter Lab
echo "📓 Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token="" --NotebookApp.password="" \
    --NotebookApp.disable_check_xsrf=True &
JUPYTER_PID=$!

echo "✅ NIC ETL Pipeline started successfully!"
echo "📊 Jupyter Lab: http://localhost:8888"
echo "💓 Health Check: http://localhost:8000/health"
echo "📈 Prometheus: http://localhost:9090"
echo "📊 Grafana: http://localhost:3000"

# Wait for either process to exit
wait $HEALTH_PID $JUPYTER_PID