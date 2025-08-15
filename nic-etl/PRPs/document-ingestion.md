# Document Ingestion - PRP

## ROLE
**Data Engineer with Document Processing Expertise**

Specialist in multi-format document handling, file system operations, and data ingestion pipelines. Expert in working with diverse document formats including PDFs, Word documents, images, and text files. Proficient in implementing robust file validation, format detection, and batch processing systems within Jupyter Notebook environments.

## OBJECTIVE
**Build Comprehensive Document Ingestion System**

Create a document ingestion module within Jupyter Notebook cells that:
* Processes multiple document formats (TXT, MD, PDF, DOCX, JPG, PNG) from GitLab repository
* Implements intelligent format detection and validation
* Manages document queuing and batch processing
* Tracks document processing state and metadata
* Ensures idempotent processing to prevent duplicates
* Provides document versioning and change detection

## MOTIVATION
**Reliable Foundation for Document Processing Pipeline**

The document ingestion system serves as the critical bridge between raw repository files and the sophisticated processing pipeline. It ensures that all documents, regardless of format or size, are properly validated, cataloged, and prepared for downstream processing. This system prevents data loss, handles format variations gracefully, and maintains processing integrity across the entire ETL pipeline.

## CONTEXT
**Multi-Format Document Environment with Processing Constraints**

Operating environment characteristics:
* Source: GitLab repository with mixed document formats
* File types: Text (TXT, MD) and binary (PDF, DOCX, JPG, PNG) formats
* Volume: Potentially hundreds of documents in various sizes
* Processing: Must handle both digital and scanned documents
* Storage: Local cache with metadata tracking
* Constraints: Jupyter Notebook cell-based implementation
* Memory: Must handle large documents efficiently
* Idempotency: Prevent reprocessing of unchanged documents

## IMPLEMENTATION BLUEPRINT
**Complete Document Ingestion Architecture**

### Architecture Overview
```
Cell 4: Document Management
├── DocumentManager class
│   ├── Document identification
│   ├── Format detection
│   ├── Validation pipeline
│   ├── Metadata extraction
│   └── State management
├── Batch processing
├── Progress tracking
└── Error recovery
```

### Code Structure
```python
# Cell 4: Document Ingestion Functions
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from enum import Enum

class DocumentFormat(Enum):
    TXT = "text/plain"
    MARKDOWN = "text/markdown"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    JPG = "image/jpeg"
    PNG = "image/png"

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Document:
    """Document metadata and processing state"""
    id: str
    path: str
    filename: str
    format: DocumentFormat
    size: int
    content_hash: str
    gitlab_path: str
    branch: str
    commit_id: Optional[str]
    created_at: datetime
    modified_at: datetime
    processing_status: ProcessingStatus
    processing_error: Optional[str]
    metadata: Dict[str, Any]

class DocumentManager:
    def __init__(self, cache_dir: Path, state_file: Path):
        self.cache_dir = cache_dir
        self.state_file = state_file
        self.documents: Dict[str, Document] = {}
        self.load_state()
        
    def ingest_documents(self, gitlab_client, folder_path: str, branch: str) -> List[Document]:
        """Ingest all documents from GitLab repository"""
        files = gitlab_client.list_repository_files(folder_path, branch)
        ingested = []
        
        for file_info in files:
            doc = self._process_file(gitlab_client, file_info, folder_path, branch)
            if doc:
                self.documents[doc.id] = doc
                ingested.append(doc)
        
        self.save_state()
        return ingested
    
    def _process_file(self, gitlab_client, file_info: Dict, folder_path: str, branch: str) -> Optional[Document]:
        """Process individual file from repository"""
        file_path = f"{folder_path}/{file_info['name']}"
        
        # Check if already processed
        doc_id = self._generate_document_id(file_path, branch)
        if doc_id in self.documents:
            existing = self.documents[doc_id]
            if existing.processing_status == ProcessingStatus.COMPLETED:
                return None  # Skip already processed
        
        try:
            # Download file content
            content = gitlab_client.download_file(file_path, branch)
            
            # Detect format
            format_type = self._detect_format(file_info['name'], content)
            if not format_type:
                return None
            
            # Calculate content hash
            content_hash = hashlib.sha256(content).hexdigest()
            
            # Create document record
            doc = Document(
                id=doc_id,
                path=str(self.cache_dir / f"{doc_id}_{file_info['name']}"),
                filename=file_info['name'],
                format=format_type,
                size=len(content),
                content_hash=content_hash,
                gitlab_path=file_path,
                branch=branch,
                commit_id=file_info.get('commit_id'),
                created_at=datetime.now(),
                modified_at=datetime.now(),
                processing_status=ProcessingStatus.PENDING,
                processing_error=None,
                metadata={
                    'original_path': file_info['path'],
                    'file_type': file_info['type'],
                    'mode': file_info.get('mode', '100644')
                }
            )
            
            # Save content to cache
            cache_path = Path(doc.path)
            cache_path.write_bytes(content)
            
            return doc
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _detect_format(self, filename: str, content: bytes) -> Optional[DocumentFormat]:
        """Detect document format from filename and content"""
        ext = Path(filename).suffix.lower()
        
        format_map = {
            '.txt': DocumentFormat.TXT,
            '.md': DocumentFormat.MARKDOWN,
            '.pdf': DocumentFormat.PDF,
            '.docx': DocumentFormat.DOCX,
            '.jpg': DocumentFormat.JPG,
            '.jpeg': DocumentFormat.JPG,
            '.png': DocumentFormat.PNG
        }
        
        if ext in format_map:
            return format_map[ext]
        
        # Fallback to mime type detection
        mime_type, _ = mimetypes.guess_type(filename)
        for format_enum in DocumentFormat:
            if format_enum.value == mime_type:
                return format_enum
        
        return None
    
    def _generate_document_id(self, file_path: str, branch: str) -> str:
        """Generate unique document identifier"""
        id_string = f"{file_path}:{branch}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def get_pending_documents(self) -> List[Document]:
        """Get all documents pending processing"""
        return [
            doc for doc in self.documents.values()
            if doc.processing_status == ProcessingStatus.PENDING
        ]
    
    def update_document_status(self, doc_id: str, status: ProcessingStatus, error: Optional[str] = None):
        """Update document processing status"""
        if doc_id in self.documents:
            self.documents[doc_id].processing_status = status
            self.documents[doc_id].processing_error = error
            self.documents[doc_id].modified_at = datetime.now()
            self.save_state()
    
    def load_state(self):
        """Load processing state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
                for doc_data in state_data.get('documents', []):
                    # Reconstruct document objects
                    doc_data['format'] = DocumentFormat[doc_data['format']]
                    doc_data['processing_status'] = ProcessingStatus[doc_data['processing_status']]
                    doc_data['created_at'] = datetime.fromisoformat(doc_data['created_at'])
                    doc_data['modified_at'] = datetime.fromisoformat(doc_data['modified_at'])
                    doc = Document(**doc_data)
                    self.documents[doc.id] = doc
    
    def save_state(self):
        """Save processing state to disk"""
        state_data = {
            'documents': [
                {
                    **asdict(doc),
                    'format': doc.format.name,
                    'processing_status': doc.processing_status.name,
                    'created_at': doc.created_at.isoformat(),
                    'modified_at': doc.modified_at.isoformat()
                }
                for doc in self.documents.values()
            ]
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

class BatchProcessor:
    """Handle batch processing of documents"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        
    def process_in_batches(self, documents: List[Document], process_func) -> Dict[str, Any]:
        """Process documents in configurable batches"""
        results = {
            'processed': [],
            'failed': [],
            'skipped': []
        }
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            for doc in batch:
                try:
                    if doc.processing_status == ProcessingStatus.COMPLETED:
                        results['skipped'].append(doc.id)
                        continue
                        
                    process_func(doc)
                    results['processed'].append(doc.id)
                    
                except Exception as e:
                    results['failed'].append({
                        'id': doc.id,
                        'error': str(e)
                    })
        
        return results
```

### Database Design
```python
# Document state schema (stored as JSON)
document_state_schema = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "path": {"type": "string"},
                    "filename": {"type": "string"},
                    "format": {"type": "string"},
                    "size": {"type": "integer"},
                    "content_hash": {"type": "string"},
                    "processing_status": {"type": "string"},
                    "processing_error": {"type": ["string", "null"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "modified_at": {"type": "string", "format": "date-time"}
                },
                "required": ["id", "path", "filename", "format", "processing_status"]
            }
        }
    }
}
```

### Error Handling
```python
class DocumentIngestionError(Exception):
    """Base exception for document ingestion errors"""
    pass

class UnsupportedFormatError(DocumentIngestionError):
    """Raised when document format is not supported"""
    pass

class DocumentValidationError(DocumentIngestionError):
    """Raised when document fails validation"""
    pass

def validate_document(doc: Document) -> bool:
    """Validate document before processing"""
    # Check file exists
    if not Path(doc.path).exists():
        raise DocumentValidationError(f"Document file not found: {doc.path}")
    
    # Check file size
    if doc.size == 0:
        raise DocumentValidationError(f"Document is empty: {doc.filename}")
    
    # Check format support
    if doc.format not in DocumentFormat:
        raise UnsupportedFormatError(f"Unsupported format: {doc.format}")
    
    return True
```

## VALIDATION LOOP
**Comprehensive Document Ingestion Testing**

### Unit Testing
```python
def test_format_detection():
    """Test document format detection"""
    manager = DocumentManager(CACHE_DIR, STATE_FILE)
    
    test_cases = [
        ("document.txt", DocumentFormat.TXT),
        ("README.md", DocumentFormat.MARKDOWN),
        ("report.pdf", DocumentFormat.PDF),
        ("contract.docx", DocumentFormat.DOCX),
        ("image.jpg", DocumentFormat.JPG),
        ("screenshot.png", DocumentFormat.PNG)
    ]
    
    for filename, expected_format in test_cases:
        detected = manager._detect_format(filename, b"dummy content")
        assert detected == expected_format

def test_document_id_generation():
    """Test deterministic document ID generation"""
    manager = DocumentManager(CACHE_DIR, STATE_FILE)
    
    id1 = manager._generate_document_id("path/to/file.pdf", "main")
    id2 = manager._generate_document_id("path/to/file.pdf", "main")
    id3 = manager._generate_document_id("path/to/file.pdf", "develop")
    
    assert id1 == id2  # Same input produces same ID
    assert id1 != id3  # Different branch produces different ID

def test_state_persistence():
    """Test saving and loading document state"""
    manager = DocumentManager(CACHE_DIR, STATE_FILE)
    
    # Create test document
    doc = Document(
        id="test123",
        path="/tmp/test.pdf",
        filename="test.pdf",
        format=DocumentFormat.PDF,
        size=1024,
        content_hash="abc123",
        gitlab_path="docs/test.pdf",
        branch="main",
        commit_id="commit123",
        created_at=datetime.now(),
        modified_at=datetime.now(),
        processing_status=ProcessingStatus.PENDING,
        processing_error=None,
        metadata={}
    )
    
    manager.documents[doc.id] = doc
    manager.save_state()
    
    # Load in new manager
    new_manager = DocumentManager(CACHE_DIR, STATE_FILE)
    assert "test123" in new_manager.documents
    assert new_manager.documents["test123"].filename == "test.pdf"
```

### Integration Testing
```python
def test_full_ingestion_workflow():
    """Test complete document ingestion from GitLab"""
    gitlab_client = GitLabClient(GITLAB_URL, GITLAB_TOKEN, GITLAB_PROJECT)
    gitlab_client.authenticate()
    
    manager = DocumentManager(CACHE_DIR, STATE_FILE)
    documents = manager.ingest_documents(gitlab_client, GITLAB_FOLDER, GITLAB_BRANCH)
    
    assert len(documents) > 0
    assert all(Path(doc.path).exists() for doc in documents)
    assert all(doc.processing_status == ProcessingStatus.PENDING for doc in documents)

def test_batch_processing():
    """Test batch document processing"""
    manager = DocumentManager(CACHE_DIR, STATE_FILE)
    processor = BatchProcessor(batch_size=5)
    
    # Get pending documents
    pending = manager.get_pending_documents()
    
    def mock_process(doc):
        manager.update_document_status(doc.id, ProcessingStatus.COMPLETED)
    
    results = processor.process_in_batches(pending, mock_process)
    
    assert len(results['processed']) == len(pending)
    assert len(results['failed']) == 0
```

### Performance Testing
* Ingestion speed: Target > 100 documents/minute
* Memory usage: Handle documents up to 100MB without OOM
* Cache efficiency: < 100ms for cached document retrieval
* State persistence: < 1 second for 1000 documents

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
* **Access Control**: Validate file paths to prevent directory traversal
* **Content Validation**: Scan for malicious content in uploaded files
* **Hash Verification**: Verify content integrity using SHA-256 hashes
* **Secure Storage**: Encrypt sensitive document metadata
* **Audit Trail**: Log all document access and modifications

### Performance Optimization
* **Parallel Ingestion**: Process multiple documents concurrently
* **Lazy Loading**: Load document content only when needed
* **Incremental Updates**: Process only new or modified documents
* **Memory Management**: Stream large files instead of loading entirely
* **Compression**: Compress cached documents to save disk space

### Maintenance Requirements
* **State Cleanup**: Periodically clean orphaned cache files
* **Format Updates**: Easy addition of new document formats
* **Monitoring**: Track ingestion rates and error frequencies
* **Recovery**: Automatic retry for failed document processing
* **Documentation**: Maintain format compatibility matrix