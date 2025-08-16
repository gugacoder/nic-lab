#!/usr/bin/env python3
"""
Basic test script for configuration management system
"""
import sys
import os
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from configuration_management import ConfigurationManager, create_configuration_manager

def test_basic_configuration():
    """Test basic configuration loading"""
    print("Testing Configuration Management System...")
    
    try:
        # Set minimal required environment variables for testing
        os.environ['GITLAB_ACCESS_TOKEN'] = 'test_token_12345'
        os.environ['QDRANT_API_KEY'] = 'test_api_key_67890'
        
        # Test configuration manager creation
        config_manager = create_configuration_manager(environment='development')
        
        print(f"✓ Configuration loaded for environment: {config_manager.config.environment}")
        print(f"✓ GitLab URL: {config_manager.config.gitlab.url}")
        print(f"✓ Qdrant URL: {config_manager.config.qdrant.url}")
        print(f"✓ Chunk size: {config_manager.config.chunking.target_chunk_size}")
        print(f"✓ Vector size: {config_manager.config.qdrant.vector_size}")
        
        # Test module config retrieval
        gitlab_config = config_manager.get_module_config('gitlab')
        print(f"✓ GitLab module config retrieved: {len(gitlab_config)} keys")
        
        # Test configuration export (without secrets)
        config_json = config_manager.export_configuration(include_secrets=False)
        print(f"✓ Configuration exported to JSON: {len(config_json)} characters")
        
        # Test env template creation
        env_template = config_manager.create_env_template()
        print(f"✓ Environment template created: {len(env_template.split())} lines")
        
        print("\n✓ All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    finally:
        # Cleanup test environment variables
        if 'GITLAB_ACCESS_TOKEN' in os.environ:
            del os.environ['GITLAB_ACCESS_TOKEN']
        if 'QDRANT_API_KEY' in os.environ:
            del os.environ['QDRANT_API_KEY']

if __name__ == "__main__":
    success = test_basic_configuration()
    sys.exit(0 if success else 1)