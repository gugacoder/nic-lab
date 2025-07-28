"""
Health Check HTTP Endpoint

This module provides a simple HTTP server for health check endpoints
that runs alongside the Streamlit application.
"""

import asyncio
import json
import logging
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
from typing import Dict, Any

from health import get_health_checker, quick_health_check
from config.settings import get_settings

logger = logging.getLogger(__name__)


class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints"""
    
    def do_GET(self):
        """Handle GET requests for health endpoints"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/health':
                self._handle_health_check()
            elif parsed_path.path == '/health/detailed':
                self._handle_detailed_health_check()
            elif parsed_path.path == '/health/quick':
                self._handle_quick_health_check()
            else:
                self._handle_not_found()
                
        except Exception as e:
            logger.error(f"Health endpoint error: {e}")
            self._send_error_response(str(e))
    
    def _handle_health_check(self):
        """Handle basic health check endpoint"""
        try:
            # Run async health check in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            health_data = loop.run_until_complete(quick_health_check())
            loop.close()
            
            self._send_json_response(health_data)
            
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_detailed_health_check(self):
        """Handle detailed health check endpoint"""
        try:
            # Run async health check in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            checker = get_health_checker()
            health_data = loop.run_until_complete(checker.check_application_health())
            loop.close()
            
            self._send_json_response(health_data)
            
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_quick_health_check(self):
        """Handle quick health check endpoint"""
        try:
            settings = get_settings()
            
            simple_health = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": settings.version,
                "uptime": "running"
            }
            
            self._send_json_response(simple_health)
            
        except Exception as e:
            self._send_error_response(str(e))
    
    def _handle_not_found(self):
        """Handle 404 not found"""
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        error_response = {
            "error": "Not found",
            "available_endpoints": [
                "/health",
                "/health/quick", 
                "/health/detailed"
            ]
        }
        
        self.wfile.write(json.dumps(error_response).encode())
    
    def _send_json_response(self, data: Dict[str, Any]):
        """Send JSON response"""
        status_code = 200
        
        # Set HTTP status based on health status
        if isinstance(data, dict) and "status" in data:
            health_status = data["status"]
            if health_status in ["error", "unhealthy"]:
                status_code = 503  # Service Unavailable
            elif health_status in ["degraded", "warning"]:
                status_code = 200  # OK but with warnings
        
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode())
    
    def _send_error_response(self, error_message: str):
        """Send error response"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        error_response = {
            "status": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override log_message to use our logger"""
        logger.debug(f"Health endpoint: {format % args}")


class HealthServer:
    """Health check HTTP server that runs alongside Streamlit"""
    
    def __init__(self, port: int = 8001, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.server = None
        self.thread = None
    
    def start(self):
        """Start the health check server in a background thread"""
        try:
            self.server = HTTPServer((self.host, self.port), HealthRequestHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            
            logger.info(f"Health check server started on {self.host}:{self.port}")
            logger.info(f"Health endpoints available:")
            logger.info(f"  - http://{self.host}:{self.port}/health")
            logger.info(f"  - http://{self.host}:{self.port}/health/quick")
            logger.info(f"  - http://{self.host}:{self.port}/health/detailed")
            
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
    
    def stop(self):
        """Stop the health check server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=1)
            logger.info("Health check server stopped")


# Global health server instance
_health_server: HealthServer = None


def start_health_server(port: int = None) -> HealthServer:
    """Start the global health server"""
    global _health_server
    
    if _health_server is None:
        settings = get_settings()
        # Use port 8001 by default, or main port + 1
        default_port = port or (settings.streamlit.server_port + 1)
        
        _health_server = HealthServer(port=default_port)
        _health_server.start()
    
    return _health_server


def stop_health_server():
    """Stop the global health server"""
    global _health_server
    if _health_server:
        _health_server.stop()
        _health_server = None


if __name__ == "__main__":
    # Test health server
    import time
    
    print("Starting health check server...")
    server = start_health_server(8001)
    
    print("Health server running. Test endpoints:")
    print("  curl http://localhost:8001/health")
    print("  curl http://localhost:8001/health/quick")
    print("  curl http://localhost:8001/health/detailed")
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping health server...")
        stop_health_server()
        print("Health server stopped")