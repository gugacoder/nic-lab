"""
Health Check Endpoint for NIC Chat System

This module provides health check functionality that can be accessed via HTTP
to monitor the application's status and component health.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any
import traceback

from config.settings import get_settings

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check service for monitoring application components"""
    
    def __init__(self):
        self.settings = get_settings()
        self.start_time = datetime.now()
    
    async def check_application_health(self) -> Dict[str, Any]:
        """Comprehensive application health check
        
        Returns:
            Dictionary containing health status and component details
        """
        health_data = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "version": self.settings.version,
            "environment": self.settings.environment,
            "components": {},
            "summary": {}
        }
        
        try:
            # Check configuration
            config_status = self._check_configuration()
            health_data["components"]["configuration"] = config_status
            
            # Check GitLab connectivity
            gitlab_status = await self._check_gitlab_health()
            health_data["components"]["gitlab"] = gitlab_status
            
            # Check Groq API
            groq_status = await self._check_groq_health()
            health_data["components"]["groq"] = groq_status
            
            # Check AI pipeline (if available)
            ai_status = await self._check_ai_pipeline_health()
            health_data["components"]["ai_pipeline"] = ai_status
            
            # Determine overall status
            health_data["status"] = self._calculate_overall_status(health_data["components"])
            
            # Create summary
            health_data["summary"] = self._create_health_summary(health_data["components"])
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_data["status"] = "error"
            health_data["error"] = str(e)
            
        return health_data
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check application configuration health"""
        try:
            missing_config = self.settings.validate_required_settings()
            
            return {
                "status": "healthy" if not missing_config else "degraded",
                "missing_config": missing_config,
                "app_name": self.settings.app_name,
                "debug_mode": self.settings.is_debug(),
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def _check_gitlab_health(self) -> Dict[str, Any]:
        """Check GitLab integration health"""
        try:
            if not self.settings.gitlab.url or not self.settings.gitlab.private_token:
                return {
                    "status": "not_configured",
                    "message": "GitLab credentials not configured",
                    "last_checked": datetime.now().isoformat()
                }
            
            # Try to import and test GitLab client
            try:
                from integrations.gitlab_client import get_gitlab_client
                
                client = get_gitlab_client()
                success, message = client.test_connection()
                
                return {
                    "status": "healthy" if success else "unhealthy",
                    "message": message,
                    "url": self.settings.gitlab.url,
                    "last_checked": datetime.now().isoformat()
                }
            except ImportError as e:
                return {
                    "status": "unavailable",
                    "message": f"GitLab client not available: {e}",
                    "last_checked": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def _check_groq_health(self) -> Dict[str, Any]:
        """Check Groq API health"""
        try:
            if not self.settings.groq.api_key:
                return {
                    "status": "not_configured",
                    "message": "Groq API key not configured",
                    "last_checked": datetime.now().isoformat()
                }
            
            # Try to import and test Groq client
            try:
                from ai.groq_client import GroqClient
                
                client = GroqClient()
                health_check = await client.health_check()
                
                return {
                    **health_check,
                    "last_checked": datetime.now().isoformat()
                }
            except ImportError as e:
                return {
                    "status": "unavailable",
                    "message": f"Groq client not available: {e}",
                    "last_checked": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    async def _check_ai_pipeline_health(self) -> Dict[str, Any]:
        """Check AI RAG pipeline health"""
        try:
            # Try to import RAG pipeline components
            try:
                from ai.rag_pipeline import RAGPipeline
                
                pipeline = RAGPipeline()
                health_data = await pipeline.health_check()
                
                return {
                    **health_data,
                    "last_checked": datetime.now().isoformat()
                }
            except ImportError as e:
                return {
                    "status": "unavailable",
                    "message": f"AI pipeline not available: {e}",
                    "last_checked": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }
    
    def _calculate_overall_status(self, components: Dict[str, Dict]) -> str:
        """Calculate overall application status based on component health"""
        statuses = [comp.get("status", "unknown") for comp in components.values()]
        
        if any(status == "error" for status in statuses):
            return "error"
        elif any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        elif any(status == "degraded" for status in statuses):
            return "degraded"
        elif any(status == "not_configured" for status in statuses):
            return "warning"
        elif all(status in ["healthy", "unavailable"] for status in statuses):
            return "healthy"
        else:
            return "unknown"
    
    def _create_health_summary(self, components: Dict[str, Dict]) -> Dict[str, Any]:
        """Create a health summary"""
        summary = {
            "total_components": len(components),
            "healthy": 0,
            "unhealthy": 0,
            "degraded": 0,
            "not_configured": 0,
            "unavailable": 0,
            "errors": []
        }
        
        for name, component in components.items():
            status = component.get("status", "unknown")
            
            if status == "healthy":
                summary["healthy"] += 1
            elif status == "unhealthy":
                summary["unhealthy"] += 1
            elif status == "degraded":
                summary["degraded"] += 1
            elif status == "not_configured":
                summary["not_configured"] += 1
            elif status == "unavailable":
                summary["unavailable"] += 1
            
            # Collect errors
            if "error" in component:
                summary["errors"].append(f"{name}: {component['error']}")
        
        return summary


# Global health checker instance
_health_checker: HealthChecker = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# Simple health check function for quick testing
async def quick_health_check() -> Dict[str, Any]:
    """Quick health check without detailed component analysis"""
    try:
        settings = get_settings()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.version,
            "environment": settings.environment,
            "message": "Application is running"
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    # Test health check functionality
    import sys
    
    async def test_health():
        print("Testing health check system...")
        
        # Quick health check
        quick_result = await quick_health_check()
        print(f"Quick health check: {quick_result['status']}")
        
        # Full health check
        checker = get_health_checker()
        full_result = await checker.check_application_health()
        
        print(f"\nFull Health Check Results:")
        print(f"Overall Status: {full_result['status']}")
        print(f"Components: {len(full_result['components'])}")
        
        for name, component in full_result["components"].items():
            status = component.get("status", "unknown")
            print(f"  {name}: {status}")
            if "error" in component:
                print(f"    Error: {component['error']}")
        
        if full_result["summary"]["errors"]:
            print(f"\nErrors found: {len(full_result['summary']['errors'])}")
            for error in full_result["summary"]["errors"]:
                print(f"  - {error}")
    
    # Run test
    asyncio.run(test_health())