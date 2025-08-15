# GitLab Repository Integration - PRP

## ROLE
**Senior Backend Engineer with GitLab API expertise**

Implement secure GitLab repository integration for automated document retrieval and synchronization within the NIC ETL pipeline. This role requires expertise in GitLab API authentication, file system operations, Git repository management, and error handling patterns.

## OBJECTIVE
**Complete GitLab repository integration enabling automated document ingestion**

Deliver a production-ready module that:
- Authenticates securely with GitLab using provided access tokens
- Connects to the specified NIC documentation repository 
- Retrieves all supported document formats from the target branch/folder
- Implements efficient file caching and synchronization mechanisms
- Provides comprehensive error handling and retry logic
- Maintains audit trails of repository access and file retrieval operations

Success criteria: Successfully retrieve all documents from `30-Aprovados` folder with 99.9% reliability, support incremental updates, and handle network failures gracefully.

## MOTIVATION
**Foundation for knowledge base population and document lineage tracking**

This integration serves as the critical entry point for the NIC ETL pipeline, enabling automated ingestion of officially approved documents from the centralized GitLab repository. The implementation ensures that all downstream processing (OCR, chunking, embedding) operates on the most current approved documentation, maintaining data freshness and regulatory compliance.

The module establishes the foundation for document provenance tracking, enabling full lineage from source repository to final vector database storage, essential for audit requirements and change management processes.

## CONTEXT
**Jupyter Notebook environment with GitLab Enterprise instance integration**

**Repository Configuration:**
- GitLab URL: `http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git`
- Target Branch: `main`
- Target Folder: `30-Aprovados`
- Access Token: `glpat-zycwWRydKE53SHxxpfbN`

**Supported File Formats:** TXT, MD, PDF, DOCX, JPG, PNG

**Technical Environment:**
- Jupyter Notebook execution context
- Python 3.8+ runtime
- Network access to GitLab Enterprise instance
- Local file system for document caching
- Integration with downstream Docling processing pipeline

**Constraints:**
- Private repository requiring token authentication
- Network reliability considerations for enterprise environment
- File size limitations for large documents
- Jupyter notebook memory management for batch operations

## IMPLEMENTATION BLUEPRINT
**Comprehensive GitLab integration architecture**

### Architecture Overview
```python
# Core Components Architecture
GitLabConnector
├── Authentication Manager (token validation, renewal)
├── Repository Client (python-gitlab integration)
├── File Retrieval Engine (batch download, filtering)
├── Cache Manager (local storage, sync detection)
├── Metadata Extractor (file info, commit tracking)
└── Error Handler (retries, fallbacks, logging)

# Data Flow
GitLab API → File Download → Local Cache → Metadata Extraction → Pipeline Handoff
```

### Code Structure
```python
# File Organization
src/
├── gitlab_integration/
│   ├── __init__.py
│   ├── connector.py          # Main GitLabConnector class
│   ├── auth_manager.py       # Token management and validation
│   ├── file_retriever.py     # Document download and caching
│   ├── metadata_extractor.py # File metadata and lineage
│   └── utils.py              # Helper functions and constants
└── config/
    └── gitlab_config.py      # Configuration constants

# Key Classes
class GitLabConnector:
    def authenticate(self) -> bool
    def list_files(self, folder_path: str) -> List[FileInfo]
    def download_file(self, file_path: str) -> bytes
    def sync_folder(self, folder_path: str) -> SyncResult
    def get_file_metadata(self, file_path: str) -> FileMetadata

class FileRetriever:
    def download_batch(self, file_list: List[str]) -> DownloadResult
    def cache_file(self, file_path: str, content: bytes) -> bool
    def check_cache_validity(self, file_path: str) -> bool

class MetadataExtractor:
    def extract_git_metadata(self, file_path: str) -> GitMetadata
    def get_commit_info(self, file_path: str) -> CommitInfo
    def build_lineage_data(self, file_path: str) -> LineageInfo
```

### Database Design
```python
# File Cache Schema (SQLite)
CREATE TABLE file_cache (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    local_path TEXT NOT NULL,
    last_modified TIMESTAMP,
    commit_sha TEXT,
    file_size INTEGER,
    content_hash TEXT,
    download_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_valid BOOLEAN DEFAULT TRUE
);

CREATE TABLE sync_history (
    id INTEGER PRIMARY KEY,
    sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    files_processed INTEGER,
    files_updated INTEGER,
    files_added INTEGER,
    files_deleted INTEGER,
    duration_seconds FLOAT,
    status TEXT
);

# Metadata Structure
@dataclass
class FileMetadata:
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    last_modified: datetime
    commit_sha: str
    commit_message: str
    author_name: str
    author_email: str
    branch: str
    repository_url: str
    download_timestamp: datetime
    content_hash: str
```

### API Specifications
```python
# Main Integration Interface
class GitLabConnector:
    def __init__(self, 
                 gitlab_url: str,
                 access_token: str,
                 project_path: str,
                 branch: str = "main",
                 cache_dir: str = "./cache"):
        """Initialize GitLab connector with authentication."""
        
    def connect(self) -> bool:
        """Establish connection and validate credentials."""
        
    def sync_documents(self, 
                      folder_path: str,
                      file_extensions: List[str] = None,
                      force_refresh: bool = False) -> SyncResult:
        """Sync all documents from specified folder."""
        
    def get_document_stream(self, folder_path: str) -> Iterator[DocumentInfo]:
        """Stream documents for memory-efficient processing."""
        
    def get_file_content(self, file_path: str) -> bytes:
        """Retrieve file content with caching."""
        
    def check_for_updates(self, since: datetime = None) -> List[str]:
        """Check for file updates since timestamp."""

# Response Models
@dataclass
class SyncResult:
    total_files: int
    processed_files: int
    updated_files: int
    failed_files: int
    duration: float
    errors: List[str]
    
@dataclass
class DocumentInfo:
    file_path: str
    content: bytes
    metadata: FileMetadata
    cached_path: str
```

### User Interface Requirements
```python
# Jupyter Notebook Interface
def setup_gitlab_integration():
    """Interactive setup for GitLab connection in Jupyter."""
    
def display_sync_progress(sync_result: SyncResult):
    """Rich display of synchronization results."""
    
def show_file_browser(folder_path: str):
    """Interactive file browser for repository exploration."""
    
def validate_connection():
    """Connection test with diagnostic information."""

# Progress Tracking
from tqdm.notebook import tqdm
from IPython.display import display, HTML, clear_output

def download_with_progress(file_list: List[str]) -> Iterator[Tuple[str, bytes]]:
    """Download files with Jupyter-friendly progress bar."""
```

### Error Handling
```python
# Exception Hierarchy
class GitLabIntegrationError(Exception):
    """Base exception for GitLab integration errors."""
    
class AuthenticationError(GitLabIntegrationError):
    """Authentication or authorization failures."""
    
class NetworkError(GitLabIntegrationError):
    """Network connectivity issues."""
    
class FileNotFoundError(GitLabIntegrationError):
    """Requested file or folder not found."""
    
class CacheError(GitLabIntegrationError):
    """Local cache corruption or access issues."""

# Retry Logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((NetworkError, ConnectionError))
)
def download_file_with_retry(file_path: str) -> bytes:
    """Download file with exponential backoff retry."""

# Error Recovery
def handle_partial_sync_failure(failed_files: List[str]) -> RecoveryResult:
    """Recover from partial synchronization failures."""
    
def diagnose_connection_issues() -> DiagnosticReport:
    """Comprehensive connection diagnostics."""
```

## VALIDATION LOOP
**Comprehensive testing strategy for GitLab integration**

### Unit Testing
```python
# Test Coverage Requirements: 95% minimum
import pytest
from unittest.mock import Mock, patch
import responses

class TestGitLabConnector:
    def test_authentication_success(self):
        """Verify successful authentication with valid token."""
        
    def test_authentication_failure(self):
        """Handle invalid token gracefully."""
        
    def test_file_download(self):
        """Download single file successfully."""
        
    def test_batch_download(self):
        """Download multiple files efficiently."""
        
    def test_cache_functionality(self):
        """Cache management and validation."""
        
    def test_metadata_extraction(self):
        """Extract complete file metadata."""
        
    def test_error_handling(self):
        """Error scenarios and recovery."""

# Mock Strategies
@pytest.fixture
def mock_gitlab_client():
    """Mock python-gitlab client for testing."""
    
@responses.activate
def test_api_responses():
    """Mock GitLab API responses."""
```

### Integration Testing
```python
# Component Integration Tests
def test_end_to_end_sync():
    """Complete synchronization workflow."""
    
def test_incremental_updates():
    """Incremental sync functionality."""
    
def test_large_file_handling():
    """Handle files exceeding memory limits."""
    
def test_network_interruption_recovery():
    """Recover from network failures."""
    
# Test Environment Setup
@pytest.fixture(scope="session")
def test_gitlab_instance():
    """Set up test GitLab instance or mock."""
```

### Performance Testing
```python
# Performance Benchmarks
def benchmark_download_speed():
    """Measure download throughput."""
    target_speed = 10_MB_per_second
    
def benchmark_batch_processing():
    """Measure batch download efficiency."""
    target_concurrent = 5_files_parallel
    
def benchmark_cache_performance():
    """Cache hit rates and access speed."""
    target_cache_hit_rate = 85_percent
    
# Memory Usage Tests
def test_memory_efficiency():
    """Ensure memory usage stays below limits."""
    max_memory_usage = 512_MB  # Jupyter constraint
```

### Security Testing
```python
# Security Validation
def test_token_security():
    """Verify token is never logged or exposed."""
    
def test_file_path_validation():
    """Prevent directory traversal attacks."""
    
def test_ssl_verification():
    """Ensure SSL certificates are validated."""
    
def test_data_sanitization():
    """Sanitize all user inputs and file paths."""
```

## ADDITIONAL NOTES
**Security, performance, and operational considerations**

### Security Considerations
- **Token Management**: Store access tokens securely, never in source code or logs
- **SSL/TLS Verification**: Always verify SSL certificates for GitLab connections
- **Input Validation**: Sanitize all file paths and repository parameters
- **Audit Logging**: Log all repository access for security auditing
- **Permission Validation**: Verify token has minimum required permissions
- **Rate Limiting**: Implement client-side rate limiting to prevent API abuse

### Performance Optimization
- **Concurrent Downloads**: Implement parallel file downloads with connection pooling
- **Intelligent Caching**: Use content hashes and last-modified timestamps for cache invalidation
- **Memory Management**: Stream large files to disk instead of loading into memory
- **Incremental Sync**: Only download changed files using Git commit tracking
- **Compression**: Support compressed file transfers where available
- **Connection Reuse**: Maintain persistent connections for batch operations

### Maintenance Requirements
- **Monitoring**: Implement health checks for GitLab connectivity
- **Logging**: Comprehensive logging with configurable levels
- **Documentation**: Auto-generate API documentation from docstrings
- **Version Compatibility**: Support GitLab API version migrations
- **Configuration Management**: Externalize all configuration parameters
- **Backup Strategy**: Implement cache backup and restore procedures
- **Update Procedures**: Define procedures for token rotation and endpoint changes