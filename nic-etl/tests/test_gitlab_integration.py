import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from gitlab_integration import (
    GitLabConnector, FileMetadata, CommitMetadata,
    create_gitlab_connector, GitLabIntegrationError,
    AuthenticationError, AccessError, NetworkError
)

class TestGitLabConnector:
    
    @pytest.fixture
    def mock_config(self):
        return {
            'gitlab_url': 'http://gitlab.processa.info',
            'access_token': 'test-token-12345',
            'project_path': 'nic/documentacao/base-de-conhecimento'
        }
    
    @pytest.fixture
    def connector(self, mock_config):
        return GitLabConnector(**mock_config)
    
    def test_connector_initialization(self, connector):
        """Test GitLab connector initialization"""
        assert connector.gitlab_url == 'http://gitlab.processa.info'
        assert connector.access_token == 'test-token-12345'
        assert connector.project_path == 'nic/documentacao/base-de-conhecimento'
        assert connector.project_id is None
        assert 'Bearer test-token-12345' in connector.headers['Authorization']
    
    @patch('gitlab_integration.requests.request')
    def test_authentication_success(self, mock_request, connector):
        """Test successful GitLab authentication"""
        # Mock user info response
        user_response = Mock()
        user_response.status_code = 200
        user_response.json.return_value = {'id': 123, 'username': 'testuser'}
        
        # Mock project info response
        project_response = Mock()
        project_response.status_code = 200
        project_response.json.return_value = {'id': 456, 'path_with_namespace': 'nic/docs'}
        
        mock_request.side_effect = [user_response, project_response]
        
        result = connector.authenticate()
        
        assert result is True
        assert connector.project_id == 456
        assert mock_request.call_count == 2
    
    @patch('gitlab_integration.requests.request')
    def test_authentication_invalid_token(self, mock_request, connector):
        """Test authentication failure with invalid token"""
        # Mock unauthorized response
        response = Mock()
        response.status_code = 401
        mock_request.return_value = response
        
        result = connector.authenticate()
        
        assert result is False
        assert connector.project_id is None
    
    @patch('gitlab_integration.requests.request')
    def test_authentication_project_not_found(self, mock_request, connector):
        """Test authentication failure when project not found"""
        # Mock user info success, project not found
        user_response = Mock()
        user_response.status_code = 200
        user_response.json.return_value = {'id': 123}
        
        project_response = Mock()
        project_response.status_code = 404
        
        mock_request.side_effect = [user_response, project_response]
        
        result = connector.authenticate()
        
        assert result is False
        assert connector.project_id is None
    
    @patch('gitlab_integration.requests.request')
    def test_get_commit_info_success(self, mock_request, connector):
        """Test successful commit info retrieval"""
        connector.project_id = 456  # Simulate authenticated state
        
        # Mock commits response
        response = Mock()
        response.status_code = 200
        response.json.return_value = [{
            'id': 'abc123def456',
            'author_name': 'Test Author',
            'created_at': '2023-01-01T12:00:00Z',
            'message': 'Test commit message'
        }]
        mock_request.return_value = response
        
        commit_info = connector.get_commit_info('main')
        
        assert isinstance(commit_info, CommitMetadata)
        assert commit_info.sha == 'abc123def456'
        assert commit_info.author == 'Test Author'
        assert commit_info.message == 'Test commit message'
        assert commit_info.is_latest is True
    
    @patch('gitlab_integration.requests.request')
    def test_list_files_success(self, mock_request, connector):
        """Test successful file listing"""
        connector.project_id = 456  # Simulate authenticated state
        
        # Mock commit response
        commit_response = Mock()
        commit_response.status_code = 200
        commit_response.json.return_value = [{
            'id': 'commit123',
            'author_name': 'Author',
            'created_at': '2023-01-01T12:00:00Z',
            'message': 'Commit msg'
        }]
        
        # Mock tree response
        tree_response = Mock()
        tree_response.status_code = 200
        tree_response.json.return_value = [
            {
                'type': 'blob',
                'path': '30-Aprovados/document1.pdf',
                'size': 1024
            },
            {
                'type': 'blob',
                'path': '30-Aprovados/document2.txt',
                'size': 512
            },
            {
                'type': 'blob',
                'path': '30-Aprovados/ignored.exe',  # Should be filtered out
                'size': 256
            },
            {
                'type': 'tree',  # Directory, should be ignored
                'path': '30-Aprovados/subfolder'
            }
        ]
        
        mock_request.side_effect = [commit_response, tree_response]
        
        files = connector.list_files(folder_path='30-Aprovados', extensions=['pdf', 'txt'])
        
        assert len(files) == 2
        assert all(isinstance(f, FileMetadata) for f in files)
        assert files[0].path == '30-Aprovados/document1.pdf'
        assert files[0].extension == 'pdf'
        assert files[1].path == '30-Aprovados/document2.txt'
        assert files[1].extension == 'txt'
    
    @patch('gitlab_integration.requests.request')
    def test_download_file_success(self, mock_request, connector):
        """Test successful file download"""
        connector.project_id = 456  # Simulate authenticated state
        
        # Mock file content response
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            'content': 'VGVzdCBmaWxlIGNvbnRlbnQ=',  # Base64 for "Test file content"
            'encoding': 'base64'
        }
        mock_request.return_value = response
        
        content = connector.download_file('30-Aprovados/test.txt')
        
        assert content == b'Test file content'
    
    @patch('gitlab_integration.requests.request')
    def test_download_file_not_found(self, mock_request, connector):
        """Test file download when file not found"""
        connector.project_id = 456  # Simulate authenticated state
        
        # Mock 404 response
        response = Mock()
        response.status_code = 404
        mock_request.return_value = response
        
        with pytest.raises(AccessError, match="File not found"):
            connector.download_file('30-Aprovados/nonexistent.txt')
    
    def test_factory_function_success(self, mock_config):
        """Test successful connector creation via factory function"""
        connector = create_gitlab_connector(mock_config)
        
        assert isinstance(connector, GitLabConnector)
        assert connector.gitlab_url == mock_config['gitlab_url']
        assert connector.access_token == mock_config['access_token']
        assert connector.project_path == mock_config['project_path']
    
    def test_factory_function_missing_config(self):
        """Test factory function with missing configuration"""
        incomplete_config = {
            'gitlab_url': 'http://gitlab.test',
            # Missing access_token and project_path
        }
        
        with pytest.raises(ValueError, match="Missing required configuration keys"):
            create_gitlab_connector(incomplete_config)
    
    def test_unauthenticated_operations(self, connector):
        """Test that operations fail when not authenticated"""
        # Connector not authenticated (project_id is None)
        
        with pytest.raises(RuntimeError, match="GitLab connection not established"):
            connector.list_files()
        
        with pytest.raises(RuntimeError, match="GitLab connection not established"):
            connector.download_file('test.txt')
        
        with pytest.raises(RuntimeError, match="GitLab connection not established"):
            connector.get_commit_info()
    
    @patch('gitlab_integration.requests.request')
    def test_validation_access_complete(self, mock_request, connector):
        """Test complete access validation"""
        # Mock all successful responses
        user_response = Mock()
        user_response.status_code = 200
        user_response.json.return_value = {'id': 123}
        
        project_response = Mock()
        project_response.status_code = 200
        project_response.json.return_value = {'id': 456}
        
        commit_response = Mock()
        commit_response.status_code = 200
        commit_response.json.return_value = [{'id': 'commit123', 'author_name': 'Author', 'created_at': '2023-01-01T12:00:00Z', 'message': 'Test'}]
        
        tree_response = Mock()
        tree_response.status_code = 200
        tree_response.json.return_value = []
        
        mock_request.side_effect = [user_response, project_response, commit_response, tree_response]
        
        validation = connector.validate_access()
        
        assert validation['authentication'] is True
        assert validation['project_access'] is True
        assert validation['repository_read'] is True
        assert validation['target_folder_access'] is True
    
    def test_supported_extensions(self, connector):
        """Test supported file extensions"""
        expected_extensions = {'.txt', '.md', '.pdf', '.docx', '.jpg', '.jpeg', '.png'}
        assert connector.SUPPORTED_EXTENSIONS == expected_extensions
    
    @patch('gitlab_integration.requests.request')
    def test_network_error_handling(self, mock_request, connector):
        """Test network error handling with retries"""
        import requests
        
        # Simulate network timeout
        mock_request.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(NetworkError, match="Request timeout"):
            connector.authenticate()
    
    def test_url_normalization(self):
        """Test GitLab URL normalization"""
        config = {
            'gitlab_url': 'http://gitlab.test/',  # URL with trailing slash
            'access_token': 'token',
            'project_path': 'test/project'
        }
        
        connector = GitLabConnector(**config)
        assert connector.gitlab_url == 'http://gitlab.test'  # Trailing slash removed