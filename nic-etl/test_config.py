#!/usr/bin/env python3
"""Test configuration management system"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv
import json
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigurationManager:
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self.environment = self._detect_environment()
        
        # Load environment variables
        self._load_environment_variables()
        
        # Initialize configuration
        self.config = self._build_configuration()
        
        # Validate configuration
        self._validate_configuration()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_name = os.getenv('NIC_ENVIRONMENT', 'development').lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            print(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_environment_variables(self):
        """Load environment variables from .env file"""
        env_path = Path(self.env_file)
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded configuration from {env_path}")
        else:
            print(f"No .env file found at {env_path}, using environment variables and defaults")
    
    def _build_configuration(self) -> Dict[str, Any]:
        """Build complete configuration from multiple sources"""
        config = {
            'environment': self.environment.value,
            
            # GitLab Configuration
            'gitlab': {
                'url': os.getenv('GITLAB_URL', 'http://gitlab.processa.info'),
                'token': os.getenv('GITLAB_TOKEN', 'glpat-zycwWRydKE53SHxxpfbN'),
                'project': os.getenv('GITLAB_PROJECT', 'nic/documentacao/base-de-conhecimento'),
                'branch': os.getenv('GITLAB_BRANCH', 'main'),
                'folder': os.getenv('GITLAB_FOLDER', '30-Aprovados')
            },
            
            # Qdrant Configuration
            'qdrant': {
                'url': os.getenv('QDRANT_URL', 'https://qdrant.codrstudio.dev/'),
                'api_key': os.getenv('QDRANT_API_KEY', '93f0c9d6b9a53758f2376decf318b3ae300e9bdb50be2d0e9c893ee4469fd857'),
                'collection': os.getenv('QDRANT_COLLECTION', 'nic')
            },
            
            # Processing Configuration
            'docling': {
                'enable_ocr': os.getenv('DOCLING_ENABLE_OCR', 'true').lower() == 'true',
                'ocr_languages': os.getenv('DOCLING_OCR_LANGUAGES', 'pt,en').split(','),
                'confidence_threshold': float(os.getenv('DOCLING_CONFIDENCE_THRESHOLD', '0.75'))
            },
            
            # Chunking Configuration
            'chunking': {
                'size': int(os.getenv('CHUNK_SIZE', '500')),
                'overlap': int(os.getenv('CHUNK_OVERLAP', '100')),
                'min_size': int(os.getenv('CHUNK_MIN_SIZE', '100'))
            },
            
            # Embedding Configuration
            'embedding': {
                'model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'),
                'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '32')),
                'device': os.getenv('EMBEDDING_DEVICE', 'cpu'),
                'normalize': os.getenv('EMBEDDING_NORMALIZE', 'true').lower() == 'true'
            },
            
            # Cache Configuration
            'cache': {
                'dir': Path(os.getenv('CACHE_DIR', './cache')),
                'state_file': Path(os.getenv('CACHE_STATE_FILE', './cache/pipeline_state.json')),
                'max_size_gb': float(os.getenv('CACHE_MAX_SIZE_GB', '10')),
                'cleanup_interval_hours': int(os.getenv('CACHE_CLEANUP_INTERVAL', '24'))
            },
            
            # Logging Configuration
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': os.getenv('LOG_FILE', './logs/nic_etl.log'),
                'max_size_mb': int(os.getenv('LOG_MAX_SIZE_MB', '100')),
                'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5'))
            },
            
            # Performance Configuration
            'performance': {
                'max_workers': int(os.getenv('MAX_WORKERS', '4')),
                'timeout_seconds': int(os.getenv('TIMEOUT_SECONDS', '300')),
                'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3')),
                'batch_size': int(os.getenv('BATCH_SIZE', '100'))
            },
            
            # Feature Flags
            'features': {
                'enable_caching': os.getenv('FEATURE_ENABLE_CACHING', 'true').lower() == 'true',
                'enable_parallel_processing': os.getenv('FEATURE_PARALLEL_PROCESSING', 'true').lower() == 'true',
                'enable_quality_checks': os.getenv('FEATURE_QUALITY_CHECKS', 'true').lower() == 'true',
                'enable_metrics': os.getenv('FEATURE_METRICS', 'true').lower() == 'true'
            }
        }
        
        # Environment-specific overrides
        config = self._apply_environment_overrides(config)
        
        return config
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides"""
        
        if self.environment == Environment.DEVELOPMENT:
            # Development overrides
            config['logging']['level'] = 'DEBUG'
            config['embedding']['batch_size'] = 8  # Smaller batches for development
            config['performance']['max_workers'] = 2
            config['cache']['max_size_gb'] = 2
            
        elif self.environment == Environment.STAGING:
            # Staging overrides
            config['logging']['level'] = 'INFO'
            config['embedding']['batch_size'] = 16
            config['performance']['max_workers'] = 4
            config['features']['enable_quality_checks'] = True
            
        elif self.environment == Environment.PRODUCTION:
            # Production overrides
            config['logging']['level'] = 'WARNING'
            config['embedding']['batch_size'] = 32
            config['performance']['max_workers'] = 8
            config['performance']['timeout_seconds'] = 600
            config['features']['enable_metrics'] = True
            
            # Production security settings
            if not config['gitlab']['token'] or config['gitlab']['token'].startswith('glpat-'):
                print("WARNING: Using development token in production")
        
        return config
    
    def _validate_configuration(self):
        """Validate configuration values"""
        validation_errors = []
        
        # Validate required fields
        required_fields = [
            ('gitlab.url', self.config['gitlab']['url']),
            ('gitlab.token', self.config['gitlab']['token']),
            ('qdrant.url', self.config['qdrant']['url']),
            ('qdrant.api_key', self.config['qdrant']['api_key'])
        ]
        
        for field_name, value in required_fields:
            if not value or value.strip() == '':
                validation_errors.append(f"Required field '{field_name}' is empty")
        
        # Validate numeric ranges
        if self.config['chunking']['size'] <= 0:
            validation_errors.append("Chunk size must be positive")
        
        if self.config['chunking']['overlap'] >= self.config['chunking']['size']:
            validation_errors.append("Chunk overlap must be less than chunk size")
        
        if self.config['embedding']['batch_size'] <= 0:
            validation_errors.append("Embedding batch size must be positive")
        
        # Validate paths
        cache_dir = self.config['cache']['dir']
        if not cache_dir.parent.exists():
            validation_errors.append(f"Cache directory parent does not exist: {cache_dir.parent}")
        
        # Report validation errors
        if validation_errors:
            print("Configuration validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            raise ValueError(f"Configuration validation failed: {len(validation_errors)} errors")
        
        print(f"Configuration validation passed for {self.environment.value} environment")

def test_configuration_loading():
    """Test configuration loading from environment"""
    manager = ConfigurationManager()
    
    assert manager.environment in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
    assert manager.config['gitlab']['url'] is not None
    assert manager.config['qdrant']['collection'] == 'nic'
    print("✓ Configuration loading test passed")

def test_environment_overrides():
    """Test environment-specific configuration overrides"""
    # Set environment
    os.environ['NIC_ENVIRONMENT'] = 'production'
    
    manager = ConfigurationManager()
    
    assert manager.config['logging']['level'] == 'WARNING'
    assert manager.config['embedding']['batch_size'] == 32
    print("✓ Environment overrides test passed")

def test_configuration_validation():
    """Test configuration validation"""
    manager = ConfigurationManager()
    
    # Should not raise exception for valid config
    manager._validate_configuration()
    print("✓ Configuration validation test passed")

if __name__ == "__main__":
    print("Running configuration management tests...\n")
    
    try:
        test_configuration_loading()
        test_environment_overrides() 
        test_configuration_validation()
        
        print("\n✅ All tests passed!")
        
        # Initialize and display configuration
        print("\n" + "="*50)
        config_manager = ConfigurationManager()
        CONFIG = config_manager.config
        
        # Print configuration summary
        print(f"\n=== NIC ETL Configuration ===")
        print(f"Environment: {CONFIG['environment']}")
        print(f"GitLab: {CONFIG['gitlab']['url']}")
        print(f"Qdrant: {CONFIG['qdrant']['url']}")
        print(f"Cache: {CONFIG['cache']['dir']}")
        print(f"Embedding Model: {CONFIG['embedding']['model']}")
        print(f"Chunk Size: {CONFIG['chunking']['size']} tokens")
        print(f"Features: {', '.join([k for k, v in CONFIG['features'].items() if v])}")
        print(f"================================\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)