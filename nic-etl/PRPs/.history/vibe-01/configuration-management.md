# Configuration Management - PRP

## ROLE
**DevOps Configuration Engineer with Security and Environment Management expertise**

Specialized in configuration architecture, environment management, and secure credential handling. Responsible for implementing centralized configuration management that supports multiple environments, validates configuration integrity, and provides secure access to sensitive parameters across the entire ETL pipeline system.

## OBJECTIVE
**Centralized, Secure Configuration Management System**

Deliver a production-ready Python module that:
- Provides centralized configuration management for all pipeline modules
- Supports environment-specific configurations (development, staging, production)
- Implements secure handling of sensitive credentials and API keys
- Validates configuration schemas and provides clear error messages for misconfigurations
- Supports configuration inheritance, merging, and environment variable overrides
- Enables configuration reloading without system restart
- Provides factory patterns for module configuration instantiation
- Implements configuration change detection and audit logging

## MOTIVATION
**Unified Configuration Foundation for Modular Architecture**

Centralized configuration management is essential for maintaining consistency, security, and operational efficiency across all pipeline modules. By implementing a robust configuration system with environment support, validation, and secure credential management, this module enables reliable deployment across different environments while maintaining security best practices and operational transparency.

## CONTEXT
**Multi-Environment Configuration Architecture**

- **Environment Support**: Development, staging, production configurations
- **Configuration Sources**: .env files, environment variables, default values
- **Security Requirements**: Secure credential storage, no hardcoded secrets
- **Module Integration**: Configuration factory for all pipeline modules
- **Validation**: Schema-based validation with clear error reporting
- **Flexibility**: Support for configuration overrides and runtime changes

## IMPLEMENTATION BLUEPRINT
**Comprehensive Configuration Management Module**

### Architecture Overview
```python
# Module Structure: modules/configuration_management.py
class ConfigurationManager:
    """Centralized configuration management with environment support"""
    
    def __init__(self, env_file: Optional[str] = None, environment: str = "development")
    def load_configuration(self) -> PipelineConfiguration
    def get_module_config(self, module_name: str) -> Dict[str, Any]
    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult
    def reload_configuration(self) -> bool
    def create_module_configs(self) -> ModuleConfigurations
    def get_secret(self, key: str) -> Optional[str]
```

### Code Structure
**File Organization**: `modules/configuration_management.py`
```python
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import json
import hashlib
from dotenv import load_dotenv
import yaml
from cryptography.fernet import Fernet
import base64

@dataclass
class GitLabConfig:
    """GitLab integration configuration"""
    url: str = "http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git"
    access_token: str = ""
    project_path: str = "nic/documentacao/base-de-conhecimento"
    branch: str = "main"
    folder_path: str = "30-Aprovados"
    timeout: float = 30.0
    retry_attempts: int = 3
    supported_extensions: List[str] = field(default_factory=lambda: ['.txt', '.md', '.pdf', '.docx', '.jpg', '.png'])

@dataclass
class DoclingConfig:
    """Docling processing configuration"""
    ocr_engine: str = "easyocr"
    confidence_threshold: float = 0.8
    max_file_size_mb: int = 100
    enable_table_extraction: bool = True
    enable_figure_extraction: bool = True
    output_format: str = "json"
    quality_gates_enabled: bool = True
    cache_processed_documents: bool = True

@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    target_chunk_size: int = 500
    overlap_size: int = 100
    max_chunk_size: int = 600
    min_chunk_size: int = 50
    model_name: str = "BAAI/bge-m3"
    boundary_strategy: str = "paragraph"
    preserve_structure: bool = True
    respect_semantic_boundaries: bool = True

@dataclass
class EmbeddingConfig:
    """Embedding generation configuration"""
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    device: str = "cpu"
    cache_model: bool = True
    max_memory_gb: float = 4.0
    num_threads: int = 4
    deterministic: bool = True
    warmup_iterations: int = 3

@dataclass
class QdrantConfig:
    """Qdrant vector database configuration"""
    url: str = "https://qdrant.codrstudio.dev/"
    api_key: str = ""
    collection_name: str = "nic"
    vector_size: int = 1024
    distance_metric: str = "COSINE"
    timeout: float = 30.0
    retry_attempts: int = 3
    batch_size: int = 100
    enable_payload_validation: bool = True
    optimize_collection: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console_logging: bool = True
    enable_file_logging: bool = True

@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration"""
    max_concurrent_documents: int = 5
    max_memory_usage_gb: float = 8.0
    checkpoint_interval: int = 100
    enable_progress_tracking: bool = True
    retry_failed_documents: bool = True
    max_pipeline_retries: int = 3

@dataclass
class PipelineConfiguration:
    """Complete pipeline configuration"""
    environment: str = "development"
    gitlab: GitLabConfig = field(default_factory=GitLabConfig)
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

@dataclass
class ValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_secrets: List[str] = field(default_factory=list)

class ConfigurationManager:
    """Production-ready configuration management with environment support"""
    
    # Default configuration file paths
    DEFAULT_ENV_FILE = ".env"
    CONFIG_SCHEMA_VERSION = "1.0"
    
    # Required secret fields that must be provided
    REQUIRED_SECRETS = {
        'gitlab.access_token': 'GITLAB_ACCESS_TOKEN',
        'qdrant.api_key': 'QDRANT_API_KEY'
    }
    
    # Environment-specific defaults
    ENVIRONMENT_DEFAULTS = {
        'development': {
            'logging.level': 'DEBUG',
            'logging.enable_console_logging': True,
            'docling.quality_gates_enabled': False,
            'pipeline.max_concurrent_documents': 2
        },
        'staging': {
            'logging.level': 'INFO',
            'logging.enable_file_logging': True,
            'docling.quality_gates_enabled': True,
            'pipeline.max_concurrent_documents': 3
        },
        'production': {
            'logging.level': 'WARNING',
            'logging.enable_file_logging': True,
            'logging.file_path': '/var/log/nic-etl/pipeline.log',
            'docling.quality_gates_enabled': True,
            'pipeline.max_concurrent_documents': 5
        }
    }
    
    def __init__(self, env_file: Optional[str] = None, environment: str = "development"):
        self.environment = environment
        self.env_file = env_file or self.DEFAULT_ENV_FILE
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.config_hash = None
        self.last_loaded = None
        
        # Load configuration
        self.config = self.load_configuration()
        
    def load_configuration(self) -> PipelineConfiguration:
        """Load configuration from multiple sources with priority order"""
        
        # 1. Start with default configuration
        config = PipelineConfiguration(environment=self.environment)
        
        # 2. Load environment file if it exists
        self._load_env_file()
        
        # 3. Apply environment-specific defaults
        self._apply_environment_defaults(config)
        
        # 4. Override with environment variables
        self._apply_environment_variables(config)
        
        # 5. Load secrets securely
        self._load_secrets(config)
        
        # 6. Validate configuration
        validation_result = self.validate_configuration(config)
        if not validation_result.is_valid:
            error_msg = f"Configuration validation failed: {'; '.join(validation_result.errors)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning: {warning}")
        
        # 7. Update tracking metadata
        config_dict = asdict(config)
        self.config_hash = hashlib.sha256(str(config_dict).encode()).hexdigest()
        self.last_loaded = datetime.utcnow()
        
        self.logger.info(f"Configuration loaded successfully for environment: {self.environment}")
        return config
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        
        env_path = Path(self.env_file)
        if env_path.exists():
            load_dotenv(env_path)
            self.logger.info(f"Loaded environment file: {env_path}")
        else:
            self.logger.info(f"Environment file not found: {env_path}, using environment variables only")
    
    def _apply_environment_defaults(self, config: PipelineConfiguration):
        """Apply environment-specific default values"""
        
        env_defaults = self.ENVIRONMENT_DEFAULTS.get(self.environment, {})
        
        for key, value in env_defaults.items():
            self._set_nested_attr(config, key, value)
    
    def _apply_environment_variables(self, config: PipelineConfiguration):
        """Apply environment variable overrides"""
        
        # GitLab configuration
        config.gitlab.url = os.getenv('GITLAB_URL', config.gitlab.url)
        config.gitlab.project_path = os.getenv('GITLAB_PROJECT_PATH', config.gitlab.project_path)
        config.gitlab.branch = os.getenv('GITLAB_BRANCH', config.gitlab.branch)
        config.gitlab.folder_path = os.getenv('GITLAB_FOLDER_PATH', config.gitlab.folder_path)
        
        # Qdrant configuration
        config.qdrant.url = os.getenv('QDRANT_URL', config.qdrant.url)
        config.qdrant.collection_name = os.getenv('QDRANT_COLLECTION', config.qdrant.collection_name)
        
        # Embedding configuration
        config.embedding.model_name = os.getenv('EMBEDDING_MODEL', config.embedding.model_name)
        config.embedding.device = os.getenv('EMBEDDING_DEVICE', config.embedding.device)
        config.embedding.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', config.embedding.batch_size))
        
        # Chunking configuration
        config.chunking.target_chunk_size = int(os.getenv('CHUNK_SIZE', config.chunking.target_chunk_size))
        config.chunking.overlap_size = int(os.getenv('CHUNK_OVERLAP', config.chunking.overlap_size))
        
        # Pipeline configuration
        config.pipeline.max_concurrent_documents = int(os.getenv('MAX_CONCURRENT_DOCS', config.pipeline.max_concurrent_documents))
        config.pipeline.max_memory_usage_gb = float(os.getenv('MAX_MEMORY_GB', config.pipeline.max_memory_usage_gb))
        
        # Logging configuration
        config.logging.level = os.getenv('LOG_LEVEL', config.logging.level)
        config.logging.file_path = os.getenv('LOG_FILE_PATH', config.logging.file_path)
    
    def _load_secrets(self, config: PipelineConfiguration):
        """Load sensitive configuration values securely"""
        
        # Load secrets from environment variables
        config.gitlab.access_token = self.get_secret('GITLAB_ACCESS_TOKEN') or ""
        config.qdrant.api_key = self.get_secret('QDRANT_API_KEY') or ""
        
        # Check for encryption key for additional secret decryption
        encryption_key = os.getenv('CONFIG_ENCRYPTION_KEY')
        if encryption_key:
            # Support for encrypted secrets (optional feature)
            self._load_encrypted_secrets(config, encryption_key)
    
    def _load_encrypted_secrets(self, config: PipelineConfiguration, encryption_key: str):
        """Load encrypted secrets if encryption is configured"""
        
        try:
            fernet = Fernet(encryption_key.encode())
            
            # Example: load encrypted GitLab token if available
            encrypted_GITLAB_ACCESS_TOKEN = os.getenv('GITLAB_ACCESS_TOKEN_ENCRYPTED')
            if encrypted_GITLAB_ACCESS_TOKEN:
                decrypted_token = fernet.decrypt(encrypted_GITLAB_ACCESS_TOKEN.encode()).decode()
                config.gitlab.access_token = decrypted_token
                
        except Exception as e:
            self.logger.warning(f"Failed to decrypt secrets: {e}")
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value from environment variables"""
        
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
        return None
    
    def validate_configuration(self, config: PipelineConfiguration) -> ValidationResult:
        """Comprehensive configuration validation"""
        
        errors = []
        warnings = []
        missing_secrets = []
        
        # Validate required secrets
        for config_path, env_var in self.REQUIRED_SECRETS.items():
            value = self._get_nested_attr(config, config_path)
            if not value or not value.strip():
                missing_secrets.append(env_var)
                errors.append(f"Required secret missing: {env_var} (config: {config_path})")
        
        # Validate GitLab configuration
        if not config.gitlab.url:
            errors.append("GitLab URL is required")
        
        if not config.gitlab.project_path:
            errors.append("GitLab project path is required")
        
        # Validate Qdrant configuration
        if not config.qdrant.url:
            errors.append("Qdrant URL is required")
        
        if config.qdrant.vector_size != 1024:
            errors.append("Qdrant vector size must be 1024 for BAAI/bge-m3 compatibility")
        
        # Validate chunking configuration
        if config.chunking.target_chunk_size <= 0:
            errors.append("Chunk size must be positive")
        
        if config.chunking.overlap_size >= config.chunking.target_chunk_size:
            errors.append("Overlap size must be less than chunk size")
        
        # Validate embedding configuration
        if config.embedding.batch_size <= 0:
            errors.append("Embedding batch size must be positive")
        
        if config.embedding.max_memory_gb <= 0:
            warnings.append("Max memory limit should be set to prevent OOM errors")
        
        # Validate pipeline configuration
        if config.pipeline.max_concurrent_documents <= 0:
            errors.append("Max concurrent documents must be positive")
        
        # Validate logging configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_log_levels:
            errors.append(f"Invalid log level: {config.logging.level}. Must be one of: {valid_log_levels}")
        
        # Environment-specific validations
        if self.environment == 'production':
            if config.logging.level == 'DEBUG':
                warnings.append("DEBUG logging is not recommended for production")
            
            if not config.logging.enable_file_logging:
                warnings.append("File logging should be enabled in production")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            missing_secrets=missing_secrets
        )
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for specific module"""
        
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        module_configs = {
            'gitlab': asdict(self.config.gitlab),
            'docling': asdict(self.config.docling),
            'chunking': asdict(self.config.chunking),
            'embedding': asdict(self.config.embedding),
            'qdrant': asdict(self.config.qdrant),
            'logging': asdict(self.config.logging),
            'pipeline': asdict(self.config.pipeline)
        }
        
        if module_name not in module_configs:
            raise ValueError(f"Unknown module: {module_name}. Available: {list(module_configs.keys())}")
        
        return module_configs[module_name]
    
    def create_module_configs(self) -> Dict[str, Any]:
        """Create all module configurations for factory pattern"""
        
        return {
            'gitlab': self.get_module_config('gitlab'),
            'docling': self.get_module_config('docling'),
            'chunking': self.get_module_config('chunking'),
            'embedding': self.get_module_config('embedding'),
            'qdrant': self.get_module_config('qdrant'),
            'logging': self.get_module_config('logging'),
            'pipeline': self.get_module_config('pipeline')
        }
    
    def reload_configuration(self) -> bool:
        """Reload configuration and detect changes"""
        
        try:
            old_hash = self.config_hash
            new_config = self.load_configuration()
            
            if self.config_hash != old_hash:
                self.config = new_config
                self.logger.info("Configuration reloaded with changes detected")
                return True
            else:
                self.logger.debug("Configuration reloaded, no changes detected")
                return False
                
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            return False
    
    def export_configuration(self, format: str = "json", include_secrets: bool = False) -> str:
        """Export configuration for debugging or documentation"""
        
        config_dict = asdict(self.config)
        
        if not include_secrets:
            # Mask sensitive fields
            config_dict['gitlab']['access_token'] = "***MASKED***" if config_dict['gitlab']['access_token'] else ""
            config_dict['qdrant']['api_key'] = "***MASKED***" if config_dict['qdrant']['api_key'] else ""
        
        if format.lower() == "json":
            return json.dumps(config_dict, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'")
    
    def create_env_template(self) -> str:
        """Create .env template file with all configurable variables"""
        
        template_lines = [
            "# NIC ETL Pipeline Configuration",
            "# Copy this file to .env and fill in your values",
            "",
            "# Environment",
            "ENVIRONMENT=development",
            "",
            "# GitLab Configuration",
            "GITLAB_URL=http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git",
            "GITLAB_ACCESS_TOKEN=your_GITLAB_ACCESS_TOKEN_here",
            "GITLAB_PROJECT_PATH=nic/documentacao/base-de-conhecimento",
            "GITLAB_BRANCH=main",
            "GITLAB_FOLDER_PATH=30-Aprovados",
            "",
            "# Qdrant Configuration",
            "QDRANT_URL=https://qdrant.codrstudio.dev/",
            "QDRANT_API_KEY=your_qdrant_api_key_here",
            "QDRANT_COLLECTION=nic",
            "",
            "# Embedding Configuration",
            "EMBEDDING_MODEL=BAAI/bge-m3",
            "EMBEDDING_DEVICE=cpu",
            "EMBEDDING_BATCH_SIZE=32",
            "",
            "# Chunking Configuration",
            "CHUNK_SIZE=500",
            "CHUNK_OVERLAP=100",
            "",
            "# Pipeline Configuration",
            "MAX_CONCURRENT_DOCS=5",
            "MAX_MEMORY_GB=8.0",
            "",
            "# Logging Configuration",
            "LOG_LEVEL=INFO",
            "LOG_FILE_PATH=",
            "",
            "# Optional: Encryption key for encrypted secrets",
            "# CONFIG_ENCRYPTION_KEY=your_encryption_key_here"
        ]
        
        return "\n".join(template_lines)
    
    def _set_nested_attr(self, obj, attr_path: str, value: Any):
        """Set nested attribute using dot notation"""
        
        parts = attr_path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _get_nested_attr(self, obj, attr_path: str) -> Any:
        """Get nested attribute using dot notation"""
        
        parts = attr_path.split('.')
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

# Factory function for configuration management
def create_configuration_manager(env_file: Optional[str] = None, 
                                environment: Optional[str] = None) -> ConfigurationManager:
    """Factory function for configuration manager creation"""
    
    # Determine environment from environment variable if not specified
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    return ConfigurationManager(env_file=env_file, environment=environment)

# Context manager for configuration
class ConfigurationContext:
    """Context manager for configuration with automatic cleanup"""
    
    def __init__(self, env_file: Optional[str] = None, environment: Optional[str] = None):
        self.env_file = env_file
        self.environment = environment
        self.config_manager = None
    
    def __enter__(self) -> ConfigurationManager:
        self.config_manager = create_configuration_manager(self.env_file, self.environment)
        return self.config_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass
```

### Error Handling
**Configuration Management Error Handling**
```python
class ConfigurationError(Exception):
    """Base exception for configuration errors"""
    pass

class ConfigurationValidationError(ConfigurationError):
    """Configuration validation failures"""
    pass

class SecretLoadingError(ConfigurationError):
    """Secret loading and decryption errors"""
    pass

class EnvironmentError(ConfigurationError):
    """Environment-specific configuration errors"""
    pass

# Configuration validation decorator
def require_valid_config(func):
    """Decorator to ensure valid configuration before execution"""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'config') or not self.config:
            raise ConfigurationError("Configuration not loaded")
        return func(self, *args, **kwargs)
    return wrapper
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_configuration_management.py
import pytest
import os
import tempfile
from pathlib import Path
from modules.configuration_management import ConfigurationManager, PipelineConfiguration

class TestConfigurationManager:
    
    @pytest.fixture
    def temp_env_file(self):
        """Create temporary .env file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("GITLAB_ACCESS_TOKEN=test_token\n")
            f.write("QDRANT_API_KEY=test_api_key\n")
            f.write("LOG_LEVEL=DEBUG\n")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_default_configuration_loading(self):
        """Test loading default configuration"""
        # Mock environment variables
        os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token'
        os.environ['QDRANT_API_KEY'] = 'test_api_key'
        
        config_manager = ConfigurationManager(environment='development')
        
        assert config_manager.config is not None
        assert config_manager.config.environment == 'development'
        assert config_manager.config.gitlab.access_token == 'test_token'
        assert config_manager.config.qdrant.api_key == 'test_api_key'
        
        # Cleanup
        del os.environ['GITLAB_ACCESS_TOKEN']
        del os.environ['QDRANT_API_KEY']
    
    def test_env_file_loading(self, temp_env_file):
        """Test loading configuration from .env file"""
        config_manager = ConfigurationManager(env_file=temp_env_file, environment='development')
        
        assert config_manager.config.gitlab.access_token == 'test_token'
        assert config_manager.config.qdrant.api_key == 'test_api_key'
        assert config_manager.config.logging.level == 'DEBUG'
    
    def test_environment_specific_defaults(self):
        """Test environment-specific default values"""
        # Mock required secrets
        os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token'
        os.environ['QDRANT_API_KEY'] = 'test_api_key'
        
        # Test production environment
        prod_config = ConfigurationManager(environment='production')
        assert prod_config.config.logging.level == 'WARNING'
        assert prod_config.config.pipeline.max_concurrent_documents == 5
        
        # Test development environment
        dev_config = ConfigurationManager(environment='development')
        assert dev_config.config.logging.level == 'DEBUG'
        assert dev_config.config.pipeline.max_concurrent_documents == 2
        
        # Cleanup
        del os.environ['GITLAB_ACCESS_TOKEN']
        del os.environ['QDRANT_API_KEY']
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config_manager = ConfigurationManager(environment='development')
        
        # Test valid configuration
        valid_config = PipelineConfiguration()
        valid_config.gitlab.access_token = 'test_token'
        valid_config.qdrant.api_key = 'test_api_key'
        
        result = config_manager.validate_configuration(valid_config)
        assert result.is_valid is True
        
        # Test invalid configuration (missing secrets)
        invalid_config = PipelineConfiguration()
        result = config_manager.validate_configuration(invalid_config)
        assert result.is_valid is False
        assert len(result.missing_secrets) > 0
    
    def test_module_config_retrieval(self):
        """Test module-specific configuration retrieval"""
        os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token'
        os.environ['QDRANT_API_KEY'] = 'test_api_key'
        
        config_manager = ConfigurationManager(environment='development')
        
        gitlab_config = config_manager.get_module_config('gitlab')
        assert 'url' in gitlab_config
        assert 'access_token' in gitlab_config
        assert gitlab_config['access_token'] == 'test_token'
        
        qdrant_config = config_manager.get_module_config('qdrant')
        assert 'url' in qdrant_config
        assert 'api_key' in qdrant_config
        assert qdrant_config['vector_size'] == 1024
        
        # Cleanup
        del os.environ['GITLAB_ACCESS_TOKEN']
        del os.environ['QDRANT_API_KEY']
    
    def test_configuration_export(self):
        """Test configuration export functionality"""
        os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token'
        os.environ['QDRANT_API_KEY'] = 'test_api_key'
        
        config_manager = ConfigurationManager(environment='development')
        
        # Test JSON export without secrets
        json_export = config_manager.export_configuration(format='json', include_secrets=False)
        assert 'MASKED' in json_export
        assert 'test_token' not in json_export
        
        # Test JSON export with secrets
        json_export_with_secrets = config_manager.export_configuration(format='json', include_secrets=True)
        assert 'test_token' in json_export_with_secrets
        
        # Cleanup
        del os.environ['GITLAB_ACCESS_TOKEN']
        del os.environ['QDRANT_API_KEY']
```

### Integration Testing
```python
# tests/integration/test_config_integration.py
@pytest.mark.integration
def test_module_integration_with_config():
    """Test that all modules can be initialized with configuration"""
    
    # Setup test environment
    os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token'
    os.environ['QDRANT_API_KEY'] = 'test_api_key'
    
    config_manager = ConfigurationManager(environment='development')
    module_configs = config_manager.create_module_configs()
    
    # Test that all expected module configs are present
    expected_modules = ['gitlab', 'docling', 'chunking', 'embedding', 'qdrant', 'logging', 'pipeline']
    for module in expected_modules:
        assert module in module_configs
        assert isinstance(module_configs[module], dict)
    
    # Cleanup
    del os.environ['GITLAB_ACCESS_TOKEN']
    del os.environ['QDRANT_API_KEY']
```

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **Secret Management**: Never log or expose sensitive configuration values
- **Environment Isolation**: Strict separation between development and production configurations
- **Encryption Support**: Optional encryption for highly sensitive secrets
- **Access Control**: Limit configuration file access with appropriate permissions
- **Audit Logging**: Track configuration changes and access patterns

### Performance Optimization
- **Configuration Caching**: Cache loaded configurations to minimize I/O operations
- **Lazy Loading**: Load configuration sections only when needed
- **Change Detection**: Efficient change detection using configuration hashing
- **Memory Efficiency**: Optimize configuration objects for memory usage

### Maintenance Requirements
- **Configuration Validation**: Regular validation of production configurations
- **Secret Rotation**: Support for credential rotation without service restart
- **Documentation**: Maintain comprehensive configuration documentation
- **Version Control**: Track configuration schema versions and compatibility
- **Monitoring**: Monitor configuration health and validation status