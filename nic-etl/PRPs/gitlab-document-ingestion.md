# GitLab Document Ingestion - PRP

## ROLE
**Backend Python Developer with GitLab API expertise**

Responsible for implementing secure GitLab API integration to collect documents from specific repositories, branches, and folders. Must have experience with Python GitLab libraries, authentication handling, and file processing workflows.

## OBJECTIVE
**Implement automated document collection from GitLab repositories**

Create a robust system that:
- Connects to GitLab API using personal access tokens
- Navigates to specific branch and folder structures
- Downloads all files from target directories
- Implements retry mechanisms for failed requests
- Provides progress tracking and error reporting
- Supports multiple file formats (PDF, DOCX, images)

Success criteria: Successfully retrieve 100% of files from specified GitLab paths with <1% failure rate.

## MOTIVATION
**Enable automated knowledge base ingestion for AI-powered document processing**

This feature serves as the foundation for the NIC ETL pipeline, enabling automated collection of organizational knowledge from GitLab repositories. By automating document ingestion, the system reduces manual effort, ensures consistency, and enables real-time knowledge base updates as new documents are added to repositories.

## CONTEXT
**NIC ETL Pipeline - Document Collection Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment  
- GitLab API v4 integration
- Target GitLab instance: http://gitlab.processa.info/
- Authentication: Personal Access Token (glpat-zycwWRydKE53SHxxpfbN)
- Target repository: nic/documentacao/base-de-conhecimento.git
- Target branch: main
- Target folder: 30-Aprovados

Integration requirements:
- Must output structured file metadata for downstream processing
- Must handle various file formats (PDF, DOCX, images)
- Must integrate with document normalization pipeline
- Must provide file lineage tracking (repo, branch, commit, path)

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
GitLab API → Authentication → Repository Navigation → File Collection → Local Storage → Metadata Generation
```

### Code Structure
```python
# File organization
src/
├── ingestion/
│   ├── __init__.py
│   ├── gitlab_client.py          # GitLab API wrapper
│   ├── file_collector.py         # File collection logic
│   ├── metadata_extractor.py     # File metadata extraction
│   └── storage_manager.py        # Local file storage
├── config/
│   └── gitlab_config.py          # Configuration management
└── notebooks/
    └── 01_gitlab_ingestion.ipynb # Main execution notebook
```

### GitLab Client Implementation
```python
from gitlab import Gitlab
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

class GitLabDocumentCollector:
    def __init__(self, gitlab_url: str, access_token: str):
        self.gitlab_url = gitlab_url
        self.access_token = access_token
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Establish connection to GitLab instance"""
        try:
            self.client = Gitlab(self.gitlab_url, private_token=self.access_token)
            self.client.auth()
            return True
        except Exception as e:
            self.logger.error(f"GitLab connection failed: {e}")
            return False
    
    def collect_documents(self, project_path: str, branch: str, folder: str) -> List[Dict[str, Any]]:
        """Collect all documents from specified project/branch/folder"""
        try:
            project = self.client.projects.get(project_path, lazy=True)
            files = project.repository_tree(ref=branch, path=folder, recursive=True, all=True)
            
            collected_files = []
            for file_info in files:
                if file_info['type'] == 'blob':  # Only files, not directories
                    file_data = self._download_file(project, file_info, branch)
                    if file_data:
                        collected_files.append(file_data)
            
            return collected_files
        except Exception as e:
            self.logger.error(f"Document collection failed: {e}")
            return []
    
    def _download_file(self, project, file_info: Dict, branch: str) -> Dict[str, Any]:
        """Download individual file and extract metadata"""
        try:
            file_content = project.files.get(file_path=file_info['path'], ref=branch)
            
            return {
                'filename': file_info['name'],
                'path': file_info['path'],
                'content': file_content.decode(),
                'size': len(file_content.decode()),
                'commit_id': branch,
                'last_modified': file_info.get('modified', None),
                'file_type': Path(file_info['name']).suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"Failed to download file {file_info['path']}: {e}")
            return None
```

### Database Design
```python
# File metadata schema for tracking
file_metadata_schema = {
    'file_id': 'string',           # Unique identifier
    'original_path': 'string',     # GitLab file path
    'filename': 'string',          # Original filename
    'file_type': 'string',         # File extension
    'size_bytes': 'integer',       # File size
    'gitlab_url': 'string',        # Source GitLab URL
    'repository': 'string',        # Repository path
    'branch': 'string',            # Source branch
    'commit_id': 'string',         # Commit hash
    'collected_at': 'datetime',    # Collection timestamp
    'local_path': 'string',        # Local storage path
    'processing_status': 'string'  # Processing pipeline status
}
```

### API Specifications
```python
# Configuration interface
class GitLabConfig:
    GITLAB_URL = "http://gitlab.processa.info/"
    ACCESS_TOKEN = "glpat-zycwWRydKE53SHxxpfbN"  # Use environment variable in production
    REPOSITORY_PATH = "nic/documentacao/base-de-conhecimento"
    BRANCH = "main"
    TARGET_FOLDER = "30-Aprovados"
    
    # File type filters
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md', '.png', '.jpg', '.jpeg']
    MAX_FILE_SIZE_MB = 100
```

### Error Handling
```python
class GitLabIngestionError(Exception):
    pass

class ConnectionError(GitLabIngestionError):
    pass

class AuthenticationError(GitLabIngestionError):
    pass

class FileDownloadError(GitLabIngestionError):
    pass

# Retry mechanism with exponential backoff
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from unittest.mock import Mock, patch
from src.ingestion.gitlab_client import GitLabDocumentCollector

class TestGitLabDocumentCollector:
    def test_connection_success(self):
        collector = GitLabDocumentCollector("http://test.com", "token")
        with patch('gitlab.Gitlab') as mock_gitlab:
            mock_gitlab.return_value.auth.return_value = True
            assert collector.connect() == True
    
    def test_document_collection(self):
        collector = GitLabDocumentCollector("http://test.com", "token")
        # Mock GitLab API responses
        with patch.object(collector, '_download_file') as mock_download:
            mock_download.return_value = {'filename': 'test.pdf'}
            files = collector.collect_documents("test/repo", "main", "folder")
            assert len(files) > 0
    
    def test_authentication_failure(self):
        collector = GitLabDocumentCollector("http://test.com", "invalid_token")
        with patch('gitlab.Gitlab') as mock_gitlab:
            mock_gitlab.return_value.auth.side_effect = Exception("Auth failed")
            assert collector.connect() == False
```

### Integration Testing
```python
def test_end_to_end_collection():
    """Test complete document collection workflow"""
    collector = GitLabDocumentCollector(
        GitLabConfig.GITLAB_URL, 
        GitLabConfig.ACCESS_TOKEN
    )
    
    assert collector.connect() == True
    
    files = collector.collect_documents(
        GitLabConfig.REPOSITORY_PATH,
        GitLabConfig.BRANCH,
        GitLabConfig.TARGET_FOLDER
    )
    
    assert len(files) > 0
    assert all('filename' in file for file in files)
    assert all('content' in file for file in files)
```

### Performance Testing
- Collection of 1000+ files should complete within 10 minutes
- Memory usage should remain under 1GB during collection
- API rate limiting compliance (no more than 600 requests/minute)

### Security Testing
- Token validation and secure storage
- Input sanitization for GitLab paths
- File content validation before storage

## ADDITIONAL NOTES

### Security Considerations
- Store access tokens in environment variables, not code
- Implement token rotation mechanism
- Validate file types and sizes before download
- Sanitize file paths to prevent directory traversal attacks

### Performance Optimization
- Implement concurrent file downloads with rate limiting
- Use streaming downloads for large files
- Implement local caching to avoid re-downloading unchanged files
- Add progress tracking for large collections

### Maintenance Requirements
- Monitor GitLab API rate limits and implement backoff strategies
- Log all collection activities for audit purposes
- Implement health checks for GitLab connectivity
- Create alerts for collection failures or significant changes in document count