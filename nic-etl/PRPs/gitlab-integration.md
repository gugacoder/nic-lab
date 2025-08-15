# GitLab Integration - PRP

## ROLE
**Senior Data Engineer with GitLab API Expertise**

Responsible for designing and implementing secure, efficient GitLab repository integration within a Jupyter Notebook environment. Must possess deep understanding of GitLab API v4, authentication mechanisms, file system operations, and Python-based repository management. Expert in handling private repositories, branch management, and secure credential storage in notebook environments.

## OBJECTIVE
**Establish Secure GitLab Repository Connection and File Access**

Deliver a robust GitLab integration module within Jupyter Notebook cells that:
* Authenticates securely using GitLab personal access tokens
* Connects to the specified private repository at `http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git`
* Navigates to the `main` branch and accesses the `30-Aprovados` folder
* Lists and retrieves all supported document formats (TXT, MD, PDF, DOCX, JPG, PNG)
* Implements efficient file caching to minimize API calls
* Provides deterministic file access with version tracking

## MOTIVATION
**Foundation for Document Processing Pipeline**

This GitLab integration serves as the critical entry point for the entire NIC ETL pipeline. Official documents stored in the GitLab repository represent the authoritative source of knowledge that must be processed and indexed. Without reliable, secure access to these documents, the entire ETL pipeline cannot function. This integration ensures controlled, auditable access to sensitive organizational documents while maintaining security compliance and operational efficiency.

## CONTEXT
**Jupyter Notebook Environment with Production Constraints**

The implementation operates within a Jupyter Notebook environment with specific constraints:
* All code must reside within notebook cells (no external .py files)
* Configuration through environment variables loaded via python-dotenv
* GitLab instance: Self-hosted at `gitlab.processa.info`
* Authentication: Personal Access Token (PAT) provided
* Repository: Private repository requiring authenticated access
* Target folder: `30-Aprovados` on `main` branch
* File formats: Mixed document types including binary (PDF, DOCX, images) and text (TXT, MD)
* Network: May require proxy configuration for corporate environments
* Caching: Local file system caching to reduce API load

## IMPLEMENTATION BLUEPRINT
**Complete GitLab Integration Architecture**

### Architecture Overview
```
Cell 1: Configuration
├── Load environment variables
├── Define GitLab constants
└── Set cache directory paths

Cell 3: GitLab Functions
├── GitLabClient class
│   ├── __init__(url, token, project_path)
│   ├── authenticate()
│   ├── get_project_info()
│   ├── list_repository_files()
│   ├── download_file()
│   └── cache_management()
├── File filtering by extension
└── Batch download capabilities
```

### Code Structure
```python
# Cell 1: Configuration
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# GitLab Configuration
GITLAB_URL = os.getenv("GITLAB_URL", "http://gitlab.processa.info")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "glpat-zycwWRydKE53SHxxpfbN")
GITLAB_PROJECT = os.getenv("GITLAB_PROJECT", "nic/documentacao/base-de-conhecimento")
GITLAB_BRANCH = os.getenv("GITLAB_BRANCH", "main")
GITLAB_FOLDER = os.getenv("GITLAB_FOLDER", "30-Aprovados")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(exist_ok=True)

# Cell 3: GitLab Integration
import requests
import base64
import hashlib
from typing import List, Dict, Optional
from urllib.parse import quote
import json
from datetime import datetime

class GitLabClient:
    def __init__(self, url: str, token: str, project_path: str):
        self.base_url = url
        self.token = token
        self.project_path = project_path
        self.api_url = f"{url}/api/v4"
        self.headers = {"PRIVATE-TOKEN": token}
        self.project_id = None
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def authenticate(self) -> bool:
        """Validate token and retrieve project ID"""
        encoded_path = quote(self.project_path, safe='')
        response = self.session.get(f"{self.api_url}/projects/{encoded_path}")
        if response.status_code == 200:
            self.project_id = response.json()['id']
            return True
        raise Exception(f"Authentication failed: {response.status_code}")
    
    def list_repository_files(self, path: str, branch: str = "main") -> List[Dict]:
        """List all files in specified directory"""
        params = {
            'path': path,
            'ref': branch,
            'recursive': True,
            'per_page': 100
        }
        
        files = []
        page = 1
        while True:
            params['page'] = page
            response = self.session.get(
                f"{self.api_url}/projects/{self.project_id}/repository/tree",
                params=params
            )
            if response.status_code != 200:
                break
            
            batch = response.json()
            if not batch:
                break
                
            files.extend(batch)
            page += 1
            
        return self._filter_supported_files(files)
    
    def download_file(self, file_path: str, branch: str = "main") -> bytes:
        """Download file content from repository"""
        cache_key = self._generate_cache_key(file_path, branch)
        cached_file = CACHE_DIR / cache_key
        
        if cached_file.exists():
            return cached_file.read_bytes()
        
        encoded_path = quote(file_path, safe='')
        response = self.session.get(
            f"{self.api_url}/projects/{self.project_id}/repository/files/{encoded_path}/raw",
            params={'ref': branch}
        )
        
        if response.status_code == 200:
            content = response.content
            cached_file.write_bytes(content)
            return content
        raise Exception(f"Failed to download {file_path}: {response.status_code}")
    
    def _filter_supported_files(self, files: List[Dict]) -> List[Dict]:
        """Filter files by supported extensions"""
        supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.jpg', '.png'}
        return [
            f for f in files 
            if f['type'] == 'blob' and 
            Path(f['name']).suffix.lower() in supported_extensions
        ]
    
    def _generate_cache_key(self, file_path: str, branch: str) -> str:
        """Generate deterministic cache key for file"""
        key_string = f"{self.project_path}:{branch}:{file_path}"
        return hashlib.md5(key_string.encode()).hexdigest()
```

### Error Handling
```python
class GitLabConnectionError(Exception):
    """Raised when connection to GitLab fails"""
    pass

class GitLabAuthenticationError(Exception):
    """Raised when authentication fails"""
    pass

class GitLabFileNotFoundError(Exception):
    """Raised when requested file doesn't exist"""
    pass

def safe_download_with_retry(client: GitLabClient, file_path: str, max_retries: int = 3):
    """Download file with exponential backoff retry"""
    import time
    
    for attempt in range(max_retries):
        try:
            return client.download_file(file_path)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise GitLabConnectionError(f"Failed after {max_retries} attempts: {str(e)}")
            wait_time = 2 ** attempt
            time.sleep(wait_time)
```

## VALIDATION LOOP
**Comprehensive GitLab Integration Testing**

### Unit Testing
```python
def test_gitlab_authentication():
    """Test successful authentication with valid token"""
    client = GitLabClient(GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT)
    assert client.authenticate() == True
    assert client.project_id is not None

def test_file_listing():
    """Test repository file listing"""
    client = GitLabClient(GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT)
    client.authenticate()
    files = client.list_repository_files(GITLAB_FOLDER)
    assert len(files) > 0
    assert all('name' in f for f in files)

def test_file_download():
    """Test file download and caching"""
    client = GitLabClient(GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT)
    client.authenticate()
    files = client.list_repository_files(GITLAB_FOLDER)
    if files:
        content = client.download_file(f"{GITLAB_FOLDER}/{files[0]['name']}")
        assert content is not None
        assert len(content) > 0
```

### Integration Testing
```python
def test_full_repository_scan():
    """Test complete repository scanning workflow"""
    client = GitLabClient(GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT)
    client.authenticate()
    
    files = client.list_repository_files(GITLAB_FOLDER)
    downloaded = []
    
    for file_info in files[:5]:  # Test first 5 files
        file_path = f"{GITLAB_FOLDER}/{file_info['name']}"
        content = client.download_file(file_path)
        downloaded.append({
            'path': file_path,
            'size': len(content),
            'cached': (CACHE_DIR / client._generate_cache_key(file_path, GITLAB_BRANCH)).exists()
        })
    
    assert all(d['cached'] for d in downloaded)
```

### Performance Testing
* API call optimization: Ensure batch operations where possible
* Cache hit ratio: Target > 90% cache hits after initial download
* Connection pooling: Verify session reuse for multiple requests
* Timeout handling: Test with network delays and interruptions

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
* **Token Protection**: Never hardcode tokens in notebook cells; always use environment variables
* **Secure Storage**: Store tokens in `.env` file with restricted permissions (chmod 600)
* **Token Rotation**: Implement token expiration monitoring and rotation reminders
* **Audit Logging**: Log all file access attempts with timestamps and user context
* **Network Security**: Support HTTPS/TLS for all GitLab communications
* **Proxy Support**: Handle corporate proxy configurations via environment variables

### Performance Optimization
* **Connection Pooling**: Reuse HTTP sessions for multiple requests
* **Parallel Downloads**: Implement concurrent file downloads for large batches
* **Smart Caching**: Cache based on file hash and modification time
* **Compression**: Support gzip compression for API responses
* **Rate Limiting**: Implement exponential backoff for rate limit handling
* **Pagination**: Efficiently handle large directory listings with pagination

### Maintenance Requirements
* **Logging Setup**: Comprehensive logging with levels (DEBUG, INFO, WARNING, ERROR)
* **Monitoring**: Track API usage, cache hit rates, and download times
* **Version Compatibility**: Test against GitLab API v4 changes
* **Cache Management**: Implement cache expiration and cleanup strategies
* **Documentation**: Maintain inline documentation for all functions
* **Error Recovery**: Graceful degradation when GitLab is unavailable