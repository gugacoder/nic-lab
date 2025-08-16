#!/usr/bin/env python3
"""
Basic test script for GitLab integration system
"""
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from gitlab_integration import (
    GitLabConnector, FileMetadata, CommitMetadata,
    create_gitlab_connector, GitLabIntegrationError,
    AuthenticationError, AccessError, NetworkError
)

def test_basic_gitlab_integration():
    """Test basic GitLab integration functionality"""
    print("Testing GitLab Integration System...")
    
    try:
        # Test 1: Connector creation
        config = {
            'gitlab_url': 'http://gitlab.processa.info',
            'access_token': 'test-token-12345',
            'project_path': 'nic/documentacao/base-de-conhecimento'
        }
        
        connector = create_gitlab_connector(config)
        print("✓ GitLab connector created successfully")
        
        # Test 2: Configuration validation
        assert connector.gitlab_url == 'http://gitlab.processa.info'
        assert connector.access_token == 'test-token-12345'
        assert connector.project_path == 'nic/documentacao/base-de-conhecimento'
        print("✓ Connector configuration validated")
        
        # Test 3: Token setup
        assert connector.access_token == 'test-token-12345'
        print("✓ Authentication token configured")
        
        # Test 4: API URL construction
        expected_api_base = 'http://gitlab.processa.info/api/v4'
        assert connector.api_base == expected_api_base
        print("✓ API base URL constructed correctly")
        
        # Test 5: Supported extensions
        expected_extensions = {'.txt', '.md', '.pdf', '.docx', '.jpg', '.jpeg', '.png'}
        assert connector.SUPPORTED_EXTENSIONS == expected_extensions
        print("✓ Supported file extensions defined")
        
        # Test 6: Mock authentication success
        with patch.object(connector, '_make_request') as mock_request:
            # Mock user response
            mock_request.side_effect = [
                (200, {'id': 123, 'username': 'testuser'}),  # user info
                (200, {'id': 456, 'path_with_namespace': 'nic/docs'})  # project info
            ]
            
            result = connector.authenticate()
            assert result is True
            assert connector.project_id == 456
            print("✓ Mock authentication successful")
        
        # Test 7: Mock commit info retrieval
        with patch.object(connector, '_make_request') as mock_request:
            connector.project_id = 456  # Simulate authenticated state
            
            mock_request.return_value = (200, [{
                'id': 'abc123def456',
                'author_name': 'Test Author',
                'created_at': '2023-01-01T12:00:00Z',
                'message': 'Test commit message'
            }])
            
            commit_info = connector.get_commit_info('main')
            assert isinstance(commit_info, CommitMetadata)
            assert commit_info.sha == 'abc123def456'
            assert commit_info.author == 'Test Author'
            print("✓ Mock commit info retrieval successful")
        
        # Test 8: Mock file listing
        with patch.object(connector, '_make_request') as mock_request:
            connector.project_id = 456  # Simulate authenticated state
            
            # Mock commit response then tree response
            mock_request.side_effect = [
                (200, [{
                    'id': 'commit123',
                    'author_name': 'Author',
                    'created_at': '2023-01-01T12:00:00Z',
                    'message': 'Commit msg'
                }]),
                (200, [
                    {
                        'type': 'blob',
                        'path': '30-Aprovados/document1.pdf',
                        'size': 1024
                    },
                    {
                        'type': 'blob',
                        'path': '30-Aprovados/document2.txt',
                        'size': 512
                    }
                ])
            ]
            
            files = connector.list_files(folder_path='30-Aprovados')
            assert len(files) == 2
            assert all(isinstance(f, FileMetadata) for f in files)
            print(f"✓ Mock file listing successful: {len(files)} files found")
        
        # Test 9: Mock file download
        with patch.object(connector, '_make_request') as mock_request:
            connector.project_id = 456  # Simulate authenticated state
            
            mock_request.return_value = (200, {
                'content': 'VGVzdCBmaWxlIGNvbnRlbnQ=',  # Base64 for "Test file content"
                'encoding': 'base64'
            })
            
            content = connector.download_file('30-Aprovados/test.txt')
            assert content == b'Test file content'
            print("✓ Mock file download successful")
        
        # Test 10: Error handling validation
        connector_unauth = create_gitlab_connector(config)
        
        try:
            connector_unauth.list_files()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not established" in str(e)
            print("✓ Unauthenticated access properly blocked")
        
        # Test 11: Factory function validation
        try:
            create_gitlab_connector({'gitlab_url': 'test'})  # Missing required fields
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Missing required configuration keys" in str(e)
            print("✓ Factory function validation works")
        
        # Test 12: Access validation structure
        validation = {
            'authentication': False,
            'project_access': False,
            'repository_read': False,
            'target_folder_access': False
        }
        
        # Test that validation structure is correct
        test_validation = connector.validate_access()
        assert isinstance(test_validation, dict)
        assert all(key in test_validation for key in validation.keys())
        print("✓ Access validation structure correct")
        
        print("\n✓ All GitLab integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ GitLab integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_gitlab_integration()
    sys.exit(0 if success else 1)