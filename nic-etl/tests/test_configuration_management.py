import pytest
import os
import tempfile
from pathlib import Path
import sys

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from configuration_management import ConfigurationManager, PipelineConfiguration, ValidationResult

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
    
    @pytest.fixture
    def clean_environment(self):
        """Clean environment variables before test"""
        # Store original values
        original_values = {}
        test_vars = ['GITLAB_ACCESS_TOKEN', 'QDRANT_API_KEY', 'LOG_LEVEL', 'ENVIRONMENT']
        
        for var in test_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
        
        yield
        
        # Restore original values
        for var, value in original_values.items():
            os.environ[var] = value
    
    def test_default_configuration_loading(self, clean_environment):
        """Test loading default configuration"""
        # Mock environment variables
        os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token'
        os.environ['QDRANT_API_KEY'] = 'test_api_key'
        
        config_manager = ConfigurationManager(environment='development')
        
        assert config_manager.config is not None
        assert config_manager.config.environment == 'development'
        assert config_manager.config.gitlab.access_token == 'test_token'
        assert config_manager.config.qdrant.api_key == 'test_api_key'
    
    def test_env_file_loading(self, temp_env_file):
        """Test loading configuration from .env file"""
        config_manager = ConfigurationManager(env_file=temp_env_file, environment='development')
        
        assert config_manager.config.gitlab.access_token == 'test_token'
        assert config_manager.config.qdrant.api_key == 'test_api_key'
        assert config_manager.config.logging.level == 'DEBUG'
    
    def test_environment_specific_defaults(self, clean_environment):
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
    
    def test_configuration_validation(self, clean_environment):
        """Test configuration validation"""
        config_manager = ConfigurationManager.__new__(ConfigurationManager)
        config_manager.environment = 'development'
        config_manager.logger = pytest.LoggingMock()
        
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
    
    def test_module_config_retrieval(self, clean_environment):
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
    
    def test_configuration_export(self, clean_environment):
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
    
    def test_env_template_creation(self):
        """Test .env template creation"""
        config_manager = ConfigurationManager.__new__(ConfigurationManager)
        template = config_manager.create_env_template()
        
        assert 'GITLAB_ACCESS_TOKEN' in template
        assert 'QDRANT_API_KEY' in template
        assert 'ENVIRONMENT' in template
        assert 'CHUNK_SIZE' in template

# Mock for pytest logging
class LoggingMock:
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def debug(self, msg): pass

pytest.LoggingMock = LoggingMock