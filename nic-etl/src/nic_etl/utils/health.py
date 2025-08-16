#!/usr/bin/env python3
"""
Health Check Endpoint for NIC ETL Pipeline
Provides system health monitoring and status reporting
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self._handle_health_check()
        elif self.path == '/status':
            self._handle_status_check()
        elif self.path == '/metrics':
            self._handle_metrics()
        else:
            self._send_404()
    
    def _handle_health_check(self):
        """Basic health check endpoint"""
        try:
            health_status = self._get_health_status()
            
            if health_status['status'] == 'healthy':
                self._send_json_response(health_status, 200)
            else:
                self._send_json_response(health_status, 503)
                
        except Exception as e:
            error_response = {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            self._send_json_response(error_response, 500)
    
    def _handle_status_check(self):
        """Detailed status check endpoint"""
        try:
            status = self._get_detailed_status()
            self._send_json_response(status, 200)
        except Exception as e:
            error_response = {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            self._send_json_response(error_response, 500)
    
    def _handle_metrics(self):
        """Prometheus-style metrics endpoint"""
        try:
            metrics = self._get_prometheus_metrics()
            self._send_response(metrics, 200, 'text/plain')
        except Exception as e:
            self._send_response(f"# Error generating metrics: {e}", 500, 'text/plain')
    
    def _get_health_status(self):
        """Get basic health status"""
        checks = {
            'modules': self._check_modules(),
            'configuration': self._check_configuration(),
            'disk_space': self._check_disk_space(),
            'memory': self._check_memory()
        }
        
        all_healthy = all(checks.values())
        
        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'checks': checks,
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        }
    
    def _get_detailed_status(self):
        """Get detailed system status"""
        return {
            'application': {
                'name': 'NIC ETL Pipeline',
                'version': '1.0.0',
                'environment': self._get_environment(),
                'uptime': self._get_uptime()
            },
            'modules': self._get_module_status(),
            'configuration': self._get_config_status(),
            'resources': self._get_resource_status(),
            'external_services': self._get_external_service_status(),
            'recent_activity': self._get_recent_activity(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_prometheus_metrics(self):
        """Generate Prometheus-style metrics"""
        metrics = []
        
        # Application metrics
        metrics.append(f"nic_etl_up 1")
        metrics.append(f"nic_etl_version_info{{version=\"1.0.0\"}} 1")
        
        # Module status metrics
        module_status = self._get_module_status()
        for module, status in module_status.items():
            value = 1 if status.get('available', False) else 0
            metrics.append(f"nic_etl_module_available{{module=\"{module}\"}} {value}")
        
        # Resource metrics
        resources = self._get_resource_status()
        if 'disk_usage_percent' in resources:
            metrics.append(f"nic_etl_disk_usage_percent {resources['disk_usage_percent']}")
        if 'memory_usage_percent' in resources:
            metrics.append(f"nic_etl_memory_usage_percent {resources['memory_usage_percent']}")
        
        # Add timestamp
        timestamp = int(time.time() * 1000)
        for i, metric in enumerate(metrics):
            metrics[i] = f"{metric} {timestamp}"
        
        return "\n".join(metrics) + "\n"
    
    def _check_modules(self):
        """Check if core modules are importable"""
        try:
            # Try importing key modules
            from configuration_management import create_configuration_manager
            from pipeline_orchestration import PipelineOrchestrator
            return True
        except Exception:
            return False
    
    def _check_configuration(self):
        """Check configuration validity"""
        try:
            from configuration_management import create_configuration_manager
            config_manager = create_configuration_manager()
            return True
        except Exception:
            return False
    
    def _check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/app")
            free_percent = (free / total) * 100
            return free_percent > 10  # At least 10% free space
        except Exception:
            return True  # Assume OK if check fails
    
    def _check_memory(self):
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Less than 90% memory usage
        except Exception:
            return True  # Assume OK if check fails
    
    def _get_environment(self):
        """Get current environment"""
        import os
        return os.getenv('ENVIRONMENT', 'development')
    
    def _get_uptime(self):
        """Get application uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                return f"{uptime_seconds:.0f}s"
        except Exception:
            return "unknown"
    
    def _get_module_status(self):
        """Get status of all modules"""
        modules = {
            'configuration_management': False,
            'error_handling': False,
            'metadata_management': False,
            'gitlab_integration': False,
            'document_ingestion': False,
            'docling_processing': False,
            'text_chunking': False,
            'embedding_generation': False,
            'qdrant_integration': False,
            'pipeline_orchestration': False
        }
        
        for module_name in modules:
            try:
                __import__(module_name)
                modules[module_name] = {'available': True, 'status': 'loaded'}
            except Exception as e:
                modules[module_name] = {'available': False, 'error': str(e)}
        
        return modules
    
    def _get_config_status(self):
        """Get configuration status"""
        try:
            from configuration_management import create_configuration_manager
            config_manager = create_configuration_manager()
            
            return {
                'valid': True,
                'environment': config_manager.environment,
                'modules_configured': len(config_manager.create_module_configs())
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _get_resource_status(self):
        """Get system resource status"""
        status = {}
        
        try:
            import shutil
            total, used, free = shutil.disk_usage("/app")
            status['disk_total_gb'] = total / (1024**3)
            status['disk_used_gb'] = used / (1024**3)
            status['disk_free_gb'] = free / (1024**3)
            status['disk_usage_percent'] = (used / total) * 100
        except Exception:
            pass
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            status['memory_total_gb'] = memory.total / (1024**3)
            status['memory_used_gb'] = memory.used / (1024**3)
            status['memory_usage_percent'] = memory.percent
        except Exception:
            pass
        
        return status
    
    def _get_external_service_status(self):
        """Get external service connectivity status"""
        services = {}
        
        # Check GitLab connectivity (mock in development)
        services['gitlab'] = {
            'url': 'http://gitlab.processa.info',
            'status': 'unknown',
            'last_check': datetime.utcnow().isoformat()
        }
        
        # Check Qdrant connectivity (mock in development)
        services['qdrant'] = {
            'url': 'https://qdrant.codrstudio.dev/',
            'status': 'unknown',
            'last_check': datetime.utcnow().isoformat()
        }
        
        return services
    
    def _get_recent_activity(self):
        """Get recent system activity"""
        activity = []
        
        # Check for recent log files
        logs_dir = Path('/app/logs')
        if logs_dir.exists():
            log_files = list(logs_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                activity.append({
                    'type': 'log_activity',
                    'file': str(latest_log),
                    'modified': datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()
                })
        
        # Check for recent checkpoints
        checkpoint_dir = Path('/app/cache/checkpoints')
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.pkl'))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda f: f.stat().st_mtime)
                activity.append({
                    'type': 'checkpoint_created',
                    'file': str(latest_checkpoint),
                    'created': datetime.fromtimestamp(latest_checkpoint.stat().st_mtime).isoformat()
                })
        
        return activity[-5:]  # Return last 5 activities
    
    def _send_json_response(self, data, status_code):
        """Send JSON response"""
        json_data = json.dumps(data, indent=2)
        self._send_response(json_data, status_code, 'application/json')
    
    def _send_response(self, content, status_code, content_type):
        """Send HTTP response"""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _send_404(self):
        """Send 404 response"""
        self._send_json_response({'error': 'Not Found'}, 404)
    
    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        return

def main():
    """Start health check server"""
    port = 8000
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    
    print(f"💓 Health check server starting on port {port}")
    print(f"🔍 Health endpoint: http://localhost:{port}/health")
    print(f"📊 Status endpoint: http://localhost:{port}/status") 
    print(f"📈 Metrics endpoint: http://localhost:{port}/metrics")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n💓 Health check server stopped")
        server.shutdown()

if __name__ == "__main__":
    main()