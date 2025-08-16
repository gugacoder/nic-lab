# GitLab Integration - PRP

## ROLE
**Backend Integration Engineer with GitLab API expertise**

Specialized in GitLab API integration, secure authentication patterns, and distributed version control systems. Responsible for implementing robust, secure, and efficient GitLab repository access patterns with comprehensive error handling and rate limiting compliance.

## OBJECTIVE
**Secure GitLab Repository Integration Module**

Deliver a production-ready Python module that:
- Authenticates securely with GitLab using access tokens
- Connects to specific GitLab repositories with branch and folder targeting
- Retrieves all supported document formats (TXT, MD, PDF, DOCX, JPG, PNG)
- Implements proper error handling and retry mechanisms
- Provides file metadata including commit information and processing lineage
- Maintains audit trails for compliance and debugging

## MOTIVATION
**Foundation for Reliable Document Ingestion**

This module serves as the critical entry point for the entire NIC ETL pipeline, ensuring reliable and secure access to approved documents stored in GitLab. Robust GitLab integration prevents pipeline failures, maintains data lineage, and provides the foundation for idempotent processing workflows essential for production environments.

## CONTEXT
**GitLab Repository Access Architecture**

- **Target Repository**: `http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git`
- **Authentication**: Access token-based (`glpat-zycwWRydKE53SHxxpfbN`)
- **Target Branch**: `main`
- **Target Folder**: `30-Aprovados`
- **Supported Formats**: TXT, MD, PDF, DOCX, JPG, PNG
- **Integration Pattern**: Modular Python module called from Jupyter Notebook
- **Security Requirements**: Token protection, audit logging, secure communication

## IMPLEMENTATION BLUEPRINT
**Comprehensive GitLab Integration Module**

### Architecture Overview
```python
# Module Structure: modules/gitlab_integration.py
class GitLabConnector:
    """Secure GitLab repository connector with error handling and rate limiting"""
    
    def __init__(self, gitlab_url: str, access_token: str, project_path: str)
    def authenticate(self) -> bool
    def list_files(self, branch: str, folder_path: str, extensions: List[str]) -> List[FileMetadata]
    def download_file(self, file_path: str, branch: str) -> FileContent
    def get_commit_info(self, branch: str) -> CommitMetadata
    def validate_access(self) -> ValidationResult
```

### Code Structure
**File Organization**: `modules/gitlab_integration.py`
```python
import gitlab
import requests
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from pathlib import Path

@dataclass
class FileMetadata:
    """Complete file metadata from GitLab"""
    path: str
    name: str
    extension: str
    size: int
    commit_sha: str
    commit_date: datetime
    author: str
    gitlab_url: str
    raw_url: str

@dataclass
class CommitMetadata:
    """Git commit information for lineage tracking"""
    sha: str
    author: str
    date: datetime
    message: str
    is_latest: bool

class GitLabConnector:
    """Production-ready GitLab integration with comprehensive error handling"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.jpg', '.jpeg', '.png'}
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    def __init__(self, gitlab_url: str, access_token: str, project_path: str):
        self.gitlab_url = gitlab_url.rstrip('/')
        self.access_token = access_token
        self.project_path = project_path
        self.client = None
        self.project = None
        self.logger = logging.getLogger(__name__)
        
    def authenticate(self) -> bool:
        """Establish authenticated GitLab connection with validation"""
        try:
            self.client = gitlab.Gitlab(self.gitlab_url, private_token=self.access_token)
            self.client.auth()
            self.project = self.client.projects.get(self.project_path)
            
            # Validate access permissions
            self.project.repository_tree()
            self.logger.info(f"Successfully authenticated to GitLab project: {self.project_path}")
            return True
            
        except gitlab.exceptions.GitlabAuthenticationError:
            self.logger.error("GitLab authentication failed - invalid token")
            return False
        except gitlab.exceptions.GitlabGetError as e:
            self.logger.error(f"Project access failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"GitLab connection error: {e}")
            return False
    
    def list_files(self, branch: str = 'main', folder_path: str = '30-Aprovados', 
                   extensions: Optional[List[str]] = None) -> List[FileMetadata]:
        """List all files in target folder with metadata"""
        if not self.project:
            raise RuntimeError("GitLab connection not established")
            
        extensions = extensions or list(self.SUPPORTED_EXTENSIONS)
        ext_set = {ext.lower().lstrip('.') for ext in extensions}
        
        files = []
        try:
            # Get latest commit info for lineage
            commit_info = self.get_commit_info(branch)
            
            # Recursive file listing in target folder
            tree_items = self.project.repository_tree(path=folder_path, 
                                                    ref=branch, 
                                                    recursive=True, 
                                                    all=True)
            
            for item in tree_items:
                if item['type'] == 'blob':  # File (not directory)
                    file_path = Path(item['path'])
                    file_ext = file_path.suffix.lower().lstrip('.')
                    
                    if file_ext in ext_set:
                        file_metadata = FileMetadata(
                            path=item['path'],
                            name=file_path.name,
                            extension=file_ext,
                            size=0,  # Will be populated during download
                            commit_sha=commit_info.sha,
                            commit_date=commit_info.date,
                            author=commit_info.author,
                            gitlab_url=f"{self.gitlab_url}/{self.project_path}",
                            raw_url=f"{self.gitlab_url}/{self.project_path}/-/raw/{branch}/{item['path']}"
                        )
                        files.append(file_metadata)
            
            self.logger.info(f"Found {len(files)} files in {folder_path}")
            return files
            
        except gitlab.exceptions.GitlabGetError as e:
            self.logger.error(f"Failed to list files in {folder_path}: {e}")
            raise
    
    def download_file(self, file_path: str, branch: str = 'main') -> bytes:
        """Download file content with retry mechanism"""
        if not self.project:
            raise RuntimeError("GitLab connection not established")
        
        for attempt in range(self.MAX_RETRIES):
            try:
                file_content = self.project.files.get(file_path=file_path, ref=branch)
                content = file_content.decode()
                
                self.logger.debug(f"Successfully downloaded: {file_path}")
                return content
                
            except gitlab.exceptions.GitlabGetError as e:
                if attempt < self.MAX_RETRIES - 1:
                    self.logger.warning(f"Download attempt {attempt + 1} failed for {file_path}: {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    self.logger.error(f"Failed to download {file_path} after {self.MAX_RETRIES} attempts: {e}")
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {file_path}: {e}")
                raise
    
    def get_commit_info(self, branch: str = 'main') -> CommitMetadata:
        """Get latest commit information for lineage tracking"""
        if not self.project:
            raise RuntimeError("GitLab connection not established")
        
        try:
            commits = self.project.commits.list(ref_name=branch, per_page=1)
            if not commits:
                raise ValueError(f"No commits found for branch: {branch}")
            
            latest_commit = commits[0]
            return CommitMetadata(
                sha=latest_commit.id,
                author=latest_commit.author_name,
                date=datetime.fromisoformat(latest_commit.created_at.replace('Z', '+00:00')),
                message=latest_commit.message,
                is_latest=True
            )
            
        except gitlab.exceptions.GitlabGetError as e:
            self.logger.error(f"Failed to get commit info for branch {branch}: {e}")
            raise

def create_gitlab_connector(config: Dict[str, str]) -> GitLabConnector:
    """Factory function for GitLab connector creation"""
    required_keys = ['gitlab_url', 'access_token', 'project_path']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return GitLabConnector(
        gitlab_url=config['gitlab_url'],
        access_token=config['access_token'],
        project_path=config['project_path']
    )
```

### Error Handling
**Comprehensive Exception Management**
```python
class GitLabIntegrationError(Exception):
    """Base exception for GitLab integration errors"""
    pass

class AuthenticationError(GitLabIntegrationError):
    """Authentication failures"""
    pass

class AccessError(GitLabIntegrationError):
    """Repository or file access errors"""
    pass

class NetworkError(GitLabIntegrationError):
    """Network connectivity issues"""
    pass

# Error handling patterns with exponential backoff
def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Decorator for retry logic with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise NetworkError(f"Network error after {max_retries} attempts: {e}")
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_gitlab_integration.py
import pytest
from unittest.mock import Mock, patch
from modules.gitlab_integration import GitLabConnector, FileMetadata

class TestGitLabConnector:
    
    @pytest.fixture
    def mock_config(self):
        return {
            'gitlab_url': 'http://gitlab.processa.info',
            'access_token': 'test-token',
            'project_path': 'nic/documentacao/base-de-conhecimento'
        }
    
    @patch('gitlab.Gitlab')
    def test_authentication_success(self, mock_gitlab, mock_config):
        """Test successful GitLab authentication"""
        connector = GitLabConnector(**mock_config)
        mock_client = Mock()
        mock_gitlab.return_value = mock_client
        
        result = connector.authenticate()
        
        assert result is True
        mock_client.auth.assert_called_once()
    
    def test_authentication_failure(self, mock_config):
        """Test authentication failure handling"""
        connector = GitLabConnector(**mock_config)
        
        with patch('gitlab.Gitlab') as mock_gitlab:
            mock_gitlab.side_effect = gitlab.exceptions.GitlabAuthenticationError
            
            result = connector.authenticate()
            assert result is False
    
    def test_file_listing_filtering(self, mock_config):
        """Test file extension filtering"""
        connector = GitLabConnector(**mock_config)
        # Mock project and repository tree response
        mock_tree_response = [
            {'type': 'blob', 'path': '30-Aprovados/doc1.pdf'},
            {'type': 'blob', 'path': '30-Aprovados/doc2.txt'},
            {'type': 'blob', 'path': '30-Aprovados/image.exe'},  # Should be filtered out
        ]
        
        with patch.object(connector, 'project') as mock_project:
            mock_project.repository_tree.return_value = mock_tree_response
            
            files = connector.list_files(extensions=['pdf', 'txt'])
            
            assert len(files) == 2
            assert all(f.extension in ['pdf', 'txt'] for f in files)
```

### Integration Testing
```python
# tests/integration/test_gitlab_live.py
@pytest.mark.integration
def test_live_gitlab_connection():
    """Integration test with live GitLab instance (requires valid credentials)"""
    config = {
        'gitlab_url': os.getenv('GITLAB_URL'),
        'access_token': os.getenv('GITLAB_TOKEN'),
        'project_path': os.getenv('GITLAB_PROJECT')
    }
    
    if not all(config.values()):
        pytest.skip("GitLab credentials not provided")
    
    connector = GitLabConnector(**config)
    auth_result = connector.authenticate()
    
    assert auth_result is True
    
    files = connector.list_files(folder_path='30-Aprovados')
    assert isinstance(files, list)
    assert all(isinstance(f, FileMetadata) for f in files)
```

### Performance Testing
- **Rate Limiting Compliance**: Implement rate limiting to respect GitLab API limits (10 requests/second default)
- **Large Repository Handling**: Test with repositories containing 1000+ files
- **Network Resilience**: Test timeout handling and retry mechanisms
- **Memory Efficiency**: Ensure file listing doesn't load entire repository into memory

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **Token Security**: Access tokens stored in environment variables, never hardcoded
- **HTTPS Enforcement**: All GitLab communications over HTTPS only
- **Input Validation**: Sanitize file paths and branch names to prevent injection attacks
- **Audit Logging**: Log all access attempts, successful downloads, and failures
- **Token Rotation**: Support for token rotation without service interruption

### Performance Optimization
- **Connection Pooling**: Reuse GitLab client connections across multiple operations
- **Batch Operations**: Group file operations to minimize API calls
- **Caching Strategy**: Cache file listings for short periods to reduce API load
- **Parallel Downloads**: Support concurrent file downloads with rate limiting
- **Streaming Downloads**: Stream large files to minimize memory usage

### Maintenance Requirements
- **GitLab API Compatibility**: Regular testing against GitLab API updates
- **Token Management**: Automated token validation and renewal alerts
- **Error Monitoring**: Integration with logging infrastructure for production monitoring
- **Documentation**: Comprehensive API documentation and troubleshooting guides
- **Version Compatibility**: Support for multiple GitLab versions (13.x, 14.x, 15.x+)