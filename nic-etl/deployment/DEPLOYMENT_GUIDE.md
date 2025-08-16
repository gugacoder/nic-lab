# NIC ETL Pipeline - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the NIC ETL Pipeline in production, staging, and development environments using Docker and Docker Compose.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM recommended
- **Storage**: 50GB+ free space (for models and data)
- **Network**: Internet access for model downloads and external services

### Software Dependencies

1. **Docker Engine** (v20.10+)
2. **Docker Compose** (v2.0+)
3. **Git** (for repository access)

### Installation

#### Ubuntu/Debian
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

#### macOS
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker compose version
```

## Environment Configuration

### 1. Clone Repository

```bash
git clone <repository-url>
cd nic-etl
```

### 2. Configure Environment Variables

Choose your target environment and configure the corresponding file:

#### Production Environment
```bash
# Edit production configuration
nano deployment/environments/.env.production
```

Required configuration:
- `GITLAB_ACCESS_TOKEN`: Your GitLab personal access token
- `QDRANT_API_KEY`: Your Qdrant API key
- `GRAFANA_PASSWORD`: Secure password for Grafana
- `JUPYTER_TOKEN`: Secure token for Jupyter access

#### Staging Environment
```bash
# Edit staging configuration
nano deployment/environments/.env.staging
```

#### Development Environment
```bash
# Copy from production and modify
cp deployment/environments/.env.production deployment/environments/.env.development
nano deployment/environments/.env.development
```

### 3. Security Configuration

**Important**: Never commit real secrets to version control!

```bash
# Set secure passwords
GRAFANA_PASSWORD=$(openssl rand -base64 32)
JUPYTER_TOKEN=$(openssl rand -base64 32)

# Update environment file
sed -i "s/your_secure_grafana_password_here/$GRAFANA_PASSWORD/" deployment/environments/.env.production
sed -i "s/your_secure_jupyter_token_here/$JUPYTER_TOKEN/" deployment/environments/.env.production
```

## Deployment

### Quick Deployment

```bash
# Deploy to production
./deployment/deploy.sh production

# Deploy to staging
./deployment/deploy.sh staging

# Deploy to development
./deployment/deploy.sh development
```

### Manual Deployment

If you prefer manual control:

```bash
cd deployment

# Copy environment configuration
cp environments/.env.production .env

# Create directories
mkdir -p logs cache data

# Build and start services
docker-compose build
docker-compose up -d

# Check status
docker-compose ps
```

### Deployment Validation

The deployment script automatically validates the deployment. You can also manually check:

```bash
# Check container status
docker-compose ps

# Check health endpoint
curl http://localhost:8000/health

# Check application logs
docker-compose logs nic-etl-pipeline

# Check all services
docker-compose logs
```

## Service Access

After successful deployment, the following services are available:

| Service | URL | Description |
|---------|-----|-------------|
| **Jupyter Lab** | http://localhost:8888 | Interactive notebook interface |
| **Health Check** | http://localhost:8000/health | Application health status |
| **Status API** | http://localhost:8000/status | Detailed system status |
| **Metrics** | http://localhost:8000/metrics | Prometheus metrics |
| **Prometheus** | http://localhost:9090 | Metrics collection |
| **Grafana** | http://localhost:3000 | Monitoring dashboards |

### Default Credentials

- **Grafana**: admin / (configured password)
- **Jupyter**: No password (token-based, see logs for token)

## Configuration Management

### Environment-Specific Settings

| Setting | Development | Staging | Production |
|---------|-------------|---------|------------|
| Log Level | DEBUG | INFO | WARNING |
| Concurrent Docs | 2 | 3 | 5 |
| Batch Size | 16 | 24 | 32 |
| Memory Limit | 4GB | 6GB | 8GB |

### Runtime Configuration

Configuration can be updated without rebuilding:

```bash
# Update environment file
nano deployment/.env

# Restart services
docker-compose restart
```

## Monitoring and Maintenance

### Health Monitoring

The system provides comprehensive health monitoring:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status | jq .

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Log Management

```bash
# View real-time logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f nic-etl-pipeline

# View log files
tail -f deployment/logs/nic_etl_production.log
```

### Performance Monitoring

Access Grafana dashboard at http://localhost:3000:

1. Login with admin credentials
2. Navigate to NIC ETL Dashboard
3. Monitor key metrics:
   - Document processing throughput
   - Memory and CPU usage
   - Error rates
   - Service availability

### Backup and Recovery

#### Data Backup

```bash
# Backup cache and data
tar -czf nic-etl-backup-$(date +%Y%m%d).tar.gz \
    deployment/cache \
    deployment/data \
    deployment/logs

# Backup configuration
cp deployment/.env nic-etl-config-backup-$(date +%Y%m%d).env
```

#### Recovery

```bash
# Restore from backup
tar -xzf nic-etl-backup-YYYYMMDD.tar.gz

# Restart services
docker-compose restart
```

## Scaling and Performance

### Horizontal Scaling

For high-volume processing:

```bash
# Scale pipeline workers
docker-compose up -d --scale nic-etl-pipeline=3
```

### Performance Tuning

Edit environment configuration:

```bash
# Increase concurrent documents
MAX_CONCURRENT_DOCS=10

# Increase batch sizes
EMBEDDING_BATCH_SIZE=64

# Increase memory limit
MAX_MEMORY_GB=16.0
```

### Resource Monitoring

```bash
# Monitor resource usage
docker stats

# Monitor disk usage
df -h

# Monitor memory usage
free -h
```

## Troubleshooting

### Common Issues

#### 1. Container Startup Failures

```bash
# Check container logs
docker-compose logs nic-etl-pipeline

# Check system resources
docker system df
docker system prune  # Clean up if needed
```

#### 2. Health Check Failures

```bash
# Debug health endpoint
curl -v http://localhost:8000/health

# Check internal connectivity
docker-compose exec nic-etl-pipeline curl localhost:8000/health
```

#### 3. External Service Connectivity

```bash
# Test GitLab connectivity
docker-compose exec nic-etl-pipeline curl -I $GITLAB_URL

# Test Qdrant connectivity
docker-compose exec nic-etl-pipeline curl -I $QDRANT_URL
```

#### 4. Performance Issues

```bash
# Check resource usage
docker stats nic-etl-pipeline

# Analyze logs for bottlenecks
docker-compose logs nic-etl-pipeline | grep -E "(slow|timeout|memory)"

# Reduce concurrent processing
# Edit .env: MAX_CONCURRENT_DOCS=2
docker-compose restart
```

### Recovery Procedures

#### Full System Recovery

```bash
# Stop all services
docker-compose down

# Clean up containers and volumes
docker-compose down -v

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d
```

#### Configuration Reset

```bash
# Reset to default configuration
cp deployment/environments/.env.production deployment/.env

# Restart services
docker-compose restart
```

## Security Considerations

### Network Security

- Use firewall rules to restrict access to service ports
- Configure reverse proxy (nginx) for production
- Enable HTTPS with SSL certificates

### Access Control

- Change default passwords
- Use strong authentication tokens
- Implement IP-based access restrictions

### Data Security

- Encrypt sensitive environment variables
- Use Docker secrets for production secrets
- Regular security updates

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Check service health status
- Monitor resource usage
- Review error logs

#### Weekly
- Update monitoring dashboards
- Backup configuration and data
- Review performance metrics

#### Monthly
- Update Docker images
- Security patches
- Performance optimization review

### Update Procedure

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
./deployment/deploy.sh production
```

## Support and Documentation

### Additional Resources

- **System Architecture**: See `FINAL_EXECUTION_REPORT.md`
- **API Documentation**: Available in Jupyter notebooks
- **Troubleshooting**: Check health endpoints and logs
- **Performance Tuning**: Monitor Grafana dashboards

### Getting Help

1. Check application logs: `docker-compose logs`
2. Verify configuration: `curl localhost:8000/status`
3. Review health metrics: `curl localhost:8000/health`
4. Monitor system resources: `docker stats`

---

**Document Version**: 1.0  
**Last Updated**: August 16, 2025  
**Environment**: Production Ready