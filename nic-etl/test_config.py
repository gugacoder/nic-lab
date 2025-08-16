#!/usr/bin/env python3
"""
Test Configuration System for NIC ETL Pipeline
This module tests and validates the configuration management system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class GitLabConfig:
    """GitLab configuration"""
    url: str = "http://gitlab.processa.info"
    token: str = ""
    repository: str = "nic/documentacao/base-de-conhecimento"
    branch: str = "main"
    folder_path: str = "30-Aprovados"
    timeout: int = 30
    max_retries: int = 3

@dataclass
class QdrantConfig:
    """Qdrant configuration"""
    url: str = "https://qdrant.codrstudio.dev/"
    api_key: str = ""
    collection_name: str = "nic"
    vector_size: int = 1024
    distance_metric: str = "COSINE"
    timeout: int = 30
    batch_size: int = 100

@dataclass
class DoclingConfig:
    """Docling configuration"""
    enable_ocr: bool = True
    ocr_languages: list = field(default_factory=lambda: ["pt", "en"])
    confidence_threshold: float = 0.75
    output_format: str = "json"
    max_pages: int = 100

@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    tokenizer: str = "BAAI/bge-m3"
    max_tokens: int = 500
    overlap_tokens: int = 100
    min_chunk_size: int = 50

@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    model_name: str = "BAAI/bge-m3"
    model_dimensions: int = 1024
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = True

@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    directory: str = "./cache"
    ttl: int = 86400  # 24 hours
    max_size_mb: int = 1000

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_workers: int = 4
    max_concurrent_documents: int = 5
    request_timeout: int = 60
    retry_delay: int = 5
    max_retries: int = 3

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    max_concurrent_documents: int = 5
    checkpoint_frequency: int = 10
    enable_monitoring: bool = True
    enable_alerts: bool = True
    log_level: str = "INFO"

@dataclass
class NICETLConfig:
    """Complete NIC ETL configuration"""
    environment: str = "development"
    gitlab: GitLabConfig = field(default_factory=GitLabConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self, include_secrets: bool = False) -> str:
        """Convert configuration to JSON"""
        config_dict = self.to_dict()
        
        if not include_secrets:
            # Remove sensitive information
            if 'gitlab' in config_dict:
                config_dict['gitlab']['token'] = "***"
            if 'qdrant' in config_dict:
                config_dict['qdrant']['api_key'] = "***"
        
        return json.dumps(config_dict, indent=2)

class ConfigurationManager:
    """Configuration manager for NIC ETL Pipeline"""
    
    def __init__(self, environment: str = "development", env_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.environment = Environment(environment)
        self.config = NICETLConfig(environment=environment)
        
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Load configuration from environment
        self._load_from_env()
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # GitLab configuration
        self.config.gitlab.url = os.getenv("GITLAB_URL", self.config.gitlab.url)
        self.config.gitlab.token = os.getenv("GITLAB_TOKEN", self.config.gitlab.token)
        self.config.gitlab.repository = os.getenv("GITLAB_REPOSITORY", self.config.gitlab.repository)
        self.config.gitlab.branch = os.getenv("GITLAB_BRANCH", self.config.gitlab.branch)
        self.config.gitlab.folder_path = os.getenv("GITLAB_FOLDER_PATH", self.config.gitlab.folder_path)
        
        # Qdrant configuration
        self.config.qdrant.url = os.getenv("QDRANT_URL", self.config.qdrant.url)
        self.config.qdrant.api_key = os.getenv("QDRANT_API_KEY", self.config.qdrant.api_key)
        self.config.qdrant.collection_name = os.getenv("QDRANT_COLLECTION", self.config.qdrant.collection_name)
        
        # Embedding configuration
        self.config.embedding.model_name = os.getenv("EMBEDDING_MODEL", self.config.embedding.model_name)
        self.config.embedding.device = os.getenv("EMBEDDING_DEVICE", self.config.embedding.device)
        
        # Performance configuration
        if os.getenv("MAX_WORKERS"):
            self.config.performance.max_workers = int(os.getenv("MAX_WORKERS"))
        if os.getenv("MAX_CONCURRENT_DOCUMENTS"):
            self.config.pipeline.max_concurrent_documents = int(os.getenv("MAX_CONCURRENT_DOCUMENTS"))
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.PRODUCTION:
            self.config.pipeline.log_level = "WARNING"
            self.config.performance.max_workers = 8
            self.config.pipeline.max_concurrent_documents = 10
            self.config.cache.enabled = True
            self.config.pipeline.enable_alerts = True
        
        elif self.environment == Environment.STAGING:
            self.config.pipeline.log_level = "INFO"
            self.config.performance.max_workers = 4
            self.config.pipeline.max_concurrent_documents = 5
            self.config.cache.enabled = True
            self.config.pipeline.enable_alerts = False
        
        else:  # DEVELOPMENT
            self.config.pipeline.log_level = "DEBUG"
            self.config.performance.max_workers = 2
            self.config.pipeline.max_concurrent_documents = 2
            self.config.cache.enabled = False
            self.config.pipeline.enable_alerts = False
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific module"""
        if hasattr(self.config, module_name):
            return asdict(getattr(self.config, module_name))
        return {}
    
    def validate_configuration(self) -> tuple[bool, list[str]]:
        """Validate configuration integrity"""
        errors = []
        
        # Check required credentials
        if not self.config.gitlab.token:
            errors.append("GitLab token is missing (GITLAB_TOKEN)")
        
        if not self.config.qdrant.api_key:
            errors.append("Qdrant API key is missing (QDRANT_API_KEY)")
        
        # Check URLs
        if not self.config.gitlab.url:
            errors.append("GitLab URL is missing")
        
        if not self.config.qdrant.url:
            errors.append("Qdrant URL is missing")
        
        # Check paths
        cache_dir = Path(self.config.cache.directory)
        if self.config.cache.enabled and not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create cache directory: {e}")
        
        return len(errors) == 0, errors
    
    def export_configuration(self, include_secrets: bool = False) -> str:
        """Export configuration as JSON"""
        return self.config.to_json(include_secrets=include_secrets)
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information"""
        is_valid, errors = self.validate_configuration()
        
        return {
            "environment": self.environment.value,
            "configuration_valid": is_valid,
            "errors": errors,
            "gitlab_configured": bool(self.config.gitlab.token),
            "qdrant_configured": bool(self.config.qdrant.api_key),
            "cache_enabled": self.config.cache.enabled,
            "monitoring_enabled": self.config.pipeline.enable_monitoring
        }

def create_configuration_manager(environment: str = "development") -> ConfigurationManager:
    """Factory function to create configuration manager"""
    return ConfigurationManager(environment=environment)

def main():
    """Main function for testing configuration"""
    print("🔧 NIC ETL Configuration Test")
    print("=" * 50)
    
    # Test different environments
    for env in ["development", "staging", "production"]:
        print(f"\n📋 Testing {env.upper()} environment:")
        
        manager = create_configuration_manager(environment=env)
        
        # Validate configuration
        is_valid, errors = manager.validate_configuration()
        
        if is_valid:
            print(f"   ✅ Configuration is valid")
        else:
            print(f"   ❌ Configuration has errors:")
            for error in errors:
                print(f"      - {error}")
        
        # Show health check
        health = manager.get_health_check()
        print(f"\n   📊 Health Check:")
        for key, value in health.items():
            if key != "errors":
                print(f"      {key}: {value}")
        
        # Export configuration (without secrets)
        print(f"\n   📄 Configuration Summary:")
        config_json = json.loads(manager.export_configuration(include_secrets=False))
        print(f"      GitLab URL: {config_json['gitlab']['url']}")
        print(f"      GitLab Folder: {config_json['gitlab']['folder_path']}")
        print(f"      Qdrant URL: {config_json['qdrant']['url']}")
        print(f"      Qdrant Collection: {config_json['qdrant']['collection_name']}")
        print(f"      Embedding Model: {config_json['embedding']['model_name']}")
        print(f"      Max Workers: {config_json['performance']['max_workers']}")
        print(f"      Cache Enabled: {config_json['cache']['enabled']}")
    
    print("\n" + "=" * 50)
    print("✅ Configuration test completed")
    print("\nTo use in your code:")
    print("  from test_config import create_configuration_manager")
    print("  config = create_configuration_manager('production')")

if __name__ == "__main__":
    main()