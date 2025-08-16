# Production Dockerfile for NIC ETL Pipeline
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
COPY test_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r test_requirements.txt

# Copy application code
COPY modules/ ./modules/
COPY tests/ ./tests/
COPY *.py ./
COPY *.ipynb ./
COPY *.md ./
COPY PRPs/ ./PRPs/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data /app/cache/checkpoints

# Set environment variables
ENV PYTHONPATH=/app:/app/modules
ENV ENVIRONMENT=production
ENV JUPYTER_ENABLE_LAB=yes

# Create non-root user for security
RUN useradd -m -u 1000 nicetl && \
    chown -R nicetl:nicetl /app
USER nicetl

# Health check endpoint
COPY deployment/health_check.py ./
EXPOSE 8000 8888

# Startup script
COPY deployment/start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]