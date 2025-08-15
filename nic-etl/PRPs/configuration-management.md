# Configuration Management - PRP

## ROLE
**DevOps Engineer with Configuration Management Expertise**

Specialist in environment configuration, secrets management, and deployment automation. Expert in implementing flexible configuration systems that support multiple environments while maintaining security and maintainability.

## OBJECTIVE
**Implement Environment-Aware Configuration System**

Create a comprehensive configuration management system within Jupyter Notebook cells that:
* Supports multiple environments (development, staging, production)
* Manages secure credential storage using .env files
* Provides configuration validation and defaults
* Enables runtime configuration updates
* Maintains configuration version control
* Supports feature flags and conditional settings

## MOTIVATION
**Flexible and Secure Environment Management**

Proper configuration management enables seamless deployment across environments while maintaining security best practices. This system ensures that the same notebook can operate in different environments with appropriate configurations without code changes.

## CONTEXT
**Multi-Environment Jupyter Notebook Deployment**

Environment specifications:
* Deployment targets: Development, staging, production
* Configuration sources: Environment variables, .env files, defaults
* Security requirements: Encrypted secrets, no hardcoded credentials
* Flexibility: Runtime configuration updates
* Constraints: Jupyter Notebook implementation

## IMPLEMENTATION BLUEPRINT

### Code Structure
```python
# Cell 1: Environment Configuration and Constants
import os
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
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_runtime_config(self, updates: Dict[str, Any]):
        """Update configuration at runtime"""
        for key, value in updates.items():
            self.set(key, value)
        
        # Re-validate after updates
        self._validate_configuration()
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration (optionally excluding secrets)"""
        config_copy = self.config.copy()
        
        if not include_secrets:
            # Mask sensitive values
            sensitive_keys = [
                'gitlab.token',
                'qdrant.api_key'
            ]
            
            for key in sensitive_keys:
                value = self.get(key)
                if value:
                    self.set(key, value[:8] + '***' if len(value) > 8 else '***')
        
        return config_copy
    
    def create_env_template(self, output_path: str = '.env.template'):
        """Create .env template file with all configuration options"""
        template_content = '''# NIC ETL Pipeline Configuration
# Copy this file to .env and update values as needed

# Environment (development, staging, production)
NIC_ENVIRONMENT=development

# GitLab Configuration
GITLAB_URL=http://gitlab.processa.info
GITLAB_TOKEN=your_gitlab_token_here
GITLAB_PROJECT=nic/documentacao/base-de-conhecimento
GITLAB_BRANCH=main
GITLAB_FOLDER=30-Aprovados

# Qdrant Configuration
QDRANT_URL=https://qdrant.codrstudio.dev/
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=nic

# Processing Configuration
DOCLING_ENABLE_OCR=true
DOCLING_OCR_LANGUAGES=pt,en
DOCLING_CONFIDENCE_THRESHOLD=0.75

# Chunking Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100
CHUNK_MIN_SIZE=100

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu
EMBEDDING_NORMALIZE=true

# Cache Configuration
CACHE_DIR=./cache
CACHE_STATE_FILE=./cache/pipeline_state.json
CACHE_MAX_SIZE_GB=10
CACHE_CLEANUP_INTERVAL=24

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/nic_etl.log
LOG_MAX_SIZE_MB=100
LOG_BACKUP_COUNT=5

# Performance Configuration
MAX_WORKERS=4
TIMEOUT_SECONDS=300
RETRY_ATTEMPTS=3
BATCH_SIZE=100

# Feature Flags
FEATURE_ENABLE_CACHING=true
FEATURE_PARALLEL_PROCESSING=true
FEATURE_QUALITY_CHECKS=true
FEATURE_METRICS=true
'''
        
        with open(output_path, 'w') as f:
            f.write(template_content)
        
        print(f"Environment template created at {output_path}")

# Initialize global configuration
config_manager = ConfigurationManager()
CONFIG = config_manager.config

# Export commonly used constants
GITLAB_URL = CONFIG['gitlab']['url']
GITLAB_TOKEN = CONFIG['gitlab']['token']
GITLAB_PROJECT = CONFIG['gitlab']['project']
GITLAB_BRANCH = CONFIG['gitlab']['branch']
GITLAB_FOLDER = CONFIG['gitlab']['folder']

QDRANT_URL = CONFIG['qdrant']['url']
QDRANT_API_KEY = CONFIG['qdrant']['api_key']
QDRANT_COLLECTION = CONFIG['qdrant']['collection']

CACHE_DIR = CONFIG['cache']['dir']
STATE_FILE = CONFIG['cache']['state_file']

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
```

## VALIDATION LOOP

### Unit Testing
```python
def test_configuration_loading():
    """Test configuration loading from environment"""
    manager = ConfigurationManager()
    
    assert manager.environment in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
    assert manager.config['gitlab']['url'] is not None
    assert manager.config['qdrant']['collection'] == 'nic'

def test_environment_overrides():
    """Test environment-specific configuration overrides"""
    # Set environment
    os.environ['NIC_ENVIRONMENT'] = 'production'
    
    manager = ConfigurationManager()
    
    assert manager.config['logging']['level'] == 'WARNING'
    assert manager.config['embedding']['batch_size'] == 32

def test_configuration_validation():
    """Test configuration validation"""
    manager = ConfigurationManager()
    
    # Should not raise exception for valid config
    manager._validate_configuration()
    
    # Test invalid config
    manager.config['chunking']['size'] = -1
    
    with pytest.raises(ValueError):
        manager._validate_configuration()
```

## ADDITIONAL NOTES

### Security Considerations
* **Secret Management**: Never commit .env files with real credentials
* **Environment Isolation**: Separate configurations for each environment
* **Access Control**: Restrict access to configuration files
* **Audit Trail**: Log configuration changes

### Maintenance Requirements
* **Documentation**: Keep configuration options documented
* **Validation**: Regular validation of configuration schemas
* **Migration**: Support for configuration migrations
* **Monitoring**: Monitor configuration drift across environments