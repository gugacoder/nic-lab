# Metadata Management and NIC Schema - PRP

## ROLE
**Senior Data Engineer with metadata schema design and lineage tracking expertise**

Implement comprehensive metadata management system following the NIC Schema specifications for enriching chunks with document metadata, processing lineage, and contextual information. This role requires expertise in data modeling, schema validation, metadata extraction, and data governance practices.

## OBJECTIVE
**Complete metadata enrichment pipeline with NIC Schema compliance and validation**

Deliver a robust metadata management module that:
- Implements the complete NIC Schema for document and chunk metadata
- Enriches each chunk with comprehensive document metadata (title, description, status, created date)
- Extracts and attaches section metadata derived from Docling document structure
- Maintains complete processing lineage (OCR status, repository info, commit data, version tracking)
- Validates all metadata against NIC Schema specifications
- Provides metadata versioning and evolution support
- Implements efficient metadata caching and retrieval mechanisms
- Ensures data quality and consistency across all pipeline stages

Success criteria: 100% NIC Schema compliance, complete lineage tracking for all processed chunks, and metadata validation with zero schema violations.

## MOTIVATION
**Foundation for data governance, traceability, and enhanced search capabilities**

Rich metadata enables advanced search capabilities, data governance compliance, and complete traceability from source documents to final vector storage. The NIC Schema provides standardization across the knowledge base, enabling consistent metadata-based filtering, analytics, and content management.

Processing lineage ensures reproducibility, enables incremental updates, and supports data quality assessment throughout the pipeline lifecycle.

## CONTEXT
**Integration with all pipeline components for comprehensive metadata enrichment**

**NIC Schema Requirements:**
- Document metadata: title, description, status, created date, version info
- Section metadata: hierarchy, structure, content classification
- Processing lineage: OCR application, repository source, commit tracking, pipeline version
- Quality metrics: confidence scores, processing warnings, validation results

**Integration Points:**
- Input: Document metadata from GitLab, structure from Docling, chunks from chunking pipeline
- Processing: Schema validation, lineage tracking, metadata enrichment
- Output: Fully enriched chunks ready for embedding and Qdrant storage
- Monitoring: Metadata quality metrics and validation reports

**Technical Environment:**
- Python 3.8+ with Pydantic for schema validation
- JSON Schema for formal schema definition
- Integration with all pipeline components
- Metadata persistence and caching capabilities

## IMPLEMENTATION BLUEPRINT
**Comprehensive NIC Schema implementation with validation and lineage tracking**

### Architecture Overview
```python
# Core Components Architecture
MetadataManager
├── Schema Validator (NIC Schema compliance checking)
├── Document Enricher (document-level metadata extraction)
├── Section Enricher (structure-based metadata)
├── Lineage Tracker (processing provenance and history)
├── Quality Assessor (metadata completeness and accuracy)
├── Cache Manager (metadata storage and retrieval)
├── Version Manager (schema evolution and migration)
└── Export Formatter (output formatting for downstream)

# Metadata Flow
Raw Data → Schema Validation → Document Enrichment → Section Analysis → Lineage Tracking → Quality Assessment → Enriched Output
```

### Code Structure
```python
# File Organization
src/
├── metadata_management/
│   ├── __init__.py
│   ├── schema_manager.py      # NIC Schema definition and validation
│   ├── document_enricher.py   # Document-level metadata extraction
│   ├── section_enricher.py    # Section and structure metadata
│   ├── lineage_tracker.py     # Processing lineage management
│   ├── quality_assessor.py    # Metadata quality validation
│   ├── cache_manager.py       # Metadata caching and persistence
│   ├── version_manager.py     # Schema versioning and migration
│   └── export_formatter.py    # Output formatting for pipeline
├── schemas/
│   ├── nic_schema.json        # Formal JSON Schema definition
│   ├── document_schema.json   # Document metadata schema
│   ├── chunk_schema.json      # Chunk metadata schema
│   └── lineage_schema.json    # Processing lineage schema
└── config/
    └── metadata_config.py     # Configuration settings

# Key Classes
class MetadataManager:
    def enrich_chunk_metadata(self, chunk: Chunk, context: ProcessingContext) -> EnrichedChunk
    def validate_nic_schema(self, metadata: dict) -> ValidationResult
    def extract_document_metadata(self, file_info: FileInfo) -> DocumentMetadata
    def build_processing_lineage(self, context: ProcessingContext) -> LineageInfo

class NICSchemaValidator:
    def validate_document_metadata(self, metadata: dict) -> ValidationResult
    def validate_chunk_metadata(self, metadata: dict) -> ValidationResult
    def validate_lineage_metadata(self, metadata: dict) -> ValidationResult
    def get_schema_violations(self, metadata: dict) -> List[SchemaViolation]

class LineageTracker:
    def start_processing_session(self) -> ProcessingSession
    def track_operation(self, operation: str, metadata: dict) -> None
    def build_complete_lineage(self, chunk_id: str) -> CompleteLineage
    def export_lineage_report(self) -> LineageReport
```

### Database Design
```python
# NIC Schema Data Models
@dataclass
class NICMetadata:
    # Document Metadata (Required)
    document_id: str
    title: str
    description: str
    status: DocumentStatus           # draft, approved, published, archived
    created: datetime
    modified: datetime
    version: str
    
    # Section Metadata (Derived from Docling)
    section_hierarchy: List[str]     # e.g., ["1", "1.2", "1.2.3"]
    section_titles: List[str]        # Hierarchical section titles
    content_type: ContentType        # paragraph, table, list, heading, etc.
    page_number: Optional[int]
    region_id: Optional[str]
    
    # Chunk-Specific Metadata
    chunk_id: str
    chunk_index: int
    chunk_content_preview: str       # First 100 characters
    chunk_token_count: int
    chunk_start_position: int
    chunk_end_position: int
    
    # Processing Lineage (Critical for traceability)
    processing_timestamp: datetime
    pipeline_version: str
    ocr_applied: bool
    ocr_confidence: Optional[float]
    
    # Repository Information
    repository_url: str
    repository_branch: str
    repository_commit: str
    repository_commit_timestamp: datetime
    file_path: str
    is_latest: bool                  # Is this the latest version?
    
    # Quality Metrics
    processing_confidence: float
    metadata_completeness_score: float
    validation_warnings: List[str]
    
    # Custom Fields (Extensible)
    custom_fields: Dict[str, Any]

# Schema Validation Models
@dataclass
class ValidationResult:
    is_valid: bool
    schema_version: str
    violations: List[SchemaViolation]
    warnings: List[str]
    completeness_score: float
    
@dataclass
class SchemaViolation:
    field_path: str
    violation_type: ViolationType    # missing, invalid_type, invalid_format, constraint
    expected: str
    actual: str
    severity: SeverityLevel          # error, warning, info

# Lineage Tracking Models
@dataclass
class ProcessingLineage:
    session_id: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime]
    operations: List[ProcessingOperation]
    input_files: List[str]
    output_chunks: List[str]
    pipeline_config: dict
    
@dataclass
class ProcessingOperation:
    operation_id: str
    operation_type: str              # file_download, ocr, chunking, embedding
    start_time: datetime
    end_time: datetime
    status: OperationStatus          # success, failed, partial
    input_data: dict
    output_data: dict
    errors: List[str]
    metrics: dict

# Enumerations
class DocumentStatus(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ContentType(Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    CODE_BLOCK = "code_block"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    MIXED = "mixed"
```

### API Specifications
```python
# Main Metadata Management Interface
class MetadataManager:
    def __init__(self, 
                 schema_version: str = "1.0",
                 validation_strict: bool = True,
                 cache_enabled: bool = True):
        """Initialize metadata manager with NIC Schema configuration."""
        
    def enrich_processing_batch(self, 
                              chunks: List[Chunk],
                              processing_context: ProcessingContext) -> List[EnrichedChunk]:
        """Enrich entire batch with metadata and lineage."""
        
    def validate_metadata_compliance(self, 
                                   metadata: dict,
                                   schema_type: str = "chunk") -> ValidationResult:
        """Validate metadata against NIC Schema specifications."""
        
    def extract_document_metadata_from_git(self, 
                                          file_info: GitLabFileInfo) -> DocumentMetadata:
        """Extract document metadata from GitLab file information."""
        
    def build_section_metadata(self, 
                             structured_content: StructuredContent,
                             chunk: Chunk) -> SectionMetadata:
        """Build section metadata from Docling structured content."""

# Advanced Metadata Operations
class AdvancedMetadataOperations:
    def merge_metadata_sources(self, 
                             primary: dict,
                             secondary: dict,
                             resolution_strategy: str = "primary_wins") -> dict:
        """Merge metadata from multiple sources with conflict resolution."""
        
    def enrich_with_external_sources(self, 
                                   metadata: dict,
                                   external_apis: List[str]) -> EnrichedMetadata:
        """Enrich metadata with external data sources."""
        
    def generate_metadata_summary(self, 
                                chunk_metadata: List[dict]) -> MetadataSummary:
        """Generate summary statistics and quality reports."""

# Schema Management
class SchemaManager:
    def load_nic_schema(self, version: str = "latest") -> dict:
        """Load NIC Schema specification."""
        
    def validate_schema_evolution(self, 
                                old_version: str,
                                new_version: str) -> CompatibilityReport:
        """Validate schema version compatibility."""
        
    def migrate_metadata(self, 
                       metadata: dict,
                       from_version: str,
                       to_version: str) -> MigrationResult:
        """Migrate metadata between schema versions."""

# Quality Assessment
class MetadataQualityAssessor:
    def assess_completeness(self, metadata: dict) -> CompletenessReport:
        """Assess metadata completeness against schema requirements."""
        
    def detect_anomalies(self, metadata_batch: List[dict]) -> AnomalyReport:
        """Detect unusual patterns or inconsistencies in metadata."""
        
    def generate_quality_metrics(self, 
                                processing_session: ProcessingSession) -> QualityMetrics:
        """Generate comprehensive quality metrics for processing session."""
```

### User Interface Requirements
```python
# Jupyter Notebook Interface
def setup_metadata_management():
    """Interactive setup for metadata configuration and schema validation."""
    
def display_metadata_enrichment_results(results: List[EnrichedChunk]):
    """Rich display of metadata enrichment results with validation status."""
    
def metadata_browser_widget(chunks: List[EnrichedChunk]):
    """Interactive browser for exploring chunk metadata."""
    
def schema_validation_dashboard():
    """Interactive dashboard for schema validation and compliance monitoring."""

# Validation and Quality Tools
def visualize_metadata_completeness(metadata_batch: List[dict]):
    """Visualize metadata completeness across different fields."""
    
def display_lineage_graph(chunk_id: str):
    """Display processing lineage as interactive graph."""
    
def metadata_quality_report(processing_session: ProcessingSession):
    """Generate comprehensive metadata quality report."""

# Schema Development Tools
def schema_explorer():
    """Interactive tool for exploring NIC Schema structure."""
    
def metadata_validator_widget():
    """Interactive metadata validation tool for development."""
```

### Error Handling
```python
# Exception Hierarchy
class MetadataManagementError(Exception):
    """Base exception for metadata management errors."""
    
class SchemaValidationError(MetadataManagementError):
    """Schema validation failures."""
    
class MetadataExtractionError(MetadataManagementError):
    """Metadata extraction failures."""
    
class LineageTrackingError(MetadataManagementError):
    """Processing lineage tracking errors."""
    
class MetadataIncompleteError(MetadataManagementError):
    """Required metadata fields missing."""

# Robust Error Recovery
def handle_missing_metadata(chunk: Chunk, missing_fields: List[str]) -> PartialMetadata:
    """Handle missing metadata with fallback values."""
    
def recover_from_validation_failure(metadata: dict, violations: List[SchemaViolation]) -> dict:
    """Attempt to fix metadata validation failures."""
    
def fallback_metadata_extraction(file_info: FileInfo) -> MinimalMetadata:
    """Generate minimal required metadata when extraction fails."""

# Quality Gates
def enforce_metadata_quality_standards(metadata: dict) -> QualityGateResult:
    """Enforce minimum metadata quality standards."""
    
def validate_processing_lineage_completeness(lineage: ProcessingLineage) -> bool:
    """Validate completeness of processing lineage."""
```

## VALIDATION LOOP
**Comprehensive testing strategy for metadata management and NIC Schema compliance**

### Unit Testing
```python
# Test Coverage Requirements: 95% minimum
import pytest
from unittest.mock import Mock
import json

class TestMetadataManager:
    def test_nic_schema_validation(self):
        """Verify NIC Schema validation for all metadata types."""
        
    def test_document_metadata_extraction(self):
        """Test document metadata extraction from various sources."""
        
    def test_section_metadata_derivation(self):
        """Test section metadata derivation from Docling structure."""
        
    def test_lineage_tracking(self):
        """Test complete processing lineage tracking."""
        
    def test_metadata_enrichment(self):
        """Test chunk metadata enrichment process."""
        
    def test_schema_migration(self):
        """Test metadata migration between schema versions."""
        
    def test_quality_assessment(self):
        """Test metadata quality assessment and scoring."""

# Schema Testing
def test_schema_compliance():
    """Test compliance with formal JSON Schema definition."""
    
def test_schema_evolution():
    """Test backward compatibility during schema evolution."""

# Mock Data Generation
@pytest.fixture
def sample_metadata():
    """Generate test metadata following NIC Schema."""
    
@pytest.fixture
def mock_processing_context():
    """Mock processing context for testing."""
```

### Integration Testing
```python
# End-to-End Integration Tests
def test_gitlab_metadata_integration():
    """Test integration with GitLab metadata extraction."""
    
def test_docling_metadata_integration():
    """Test integration with Docling structured content."""
    
def test_qdrant_metadata_integration():
    """Test metadata compatibility with Qdrant storage."""
    
def test_pipeline_metadata_flow():
    """Test metadata flow through entire pipeline."""

# Quality Integration Tests
def test_metadata_consistency():
    """Test metadata consistency across pipeline stages."""
    
def test_lineage_completeness():
    """Test completeness of end-to-end lineage tracking."""
```

### Performance Testing
```python
# Performance Benchmarks
def benchmark_metadata_enrichment_speed():
    """Measure metadata enrichment throughput."""
    target_enrichment_rate = 1000_chunks_per_minute
    
def benchmark_schema_validation_speed():
    """Measure schema validation performance."""
    target_validation_rate = 5000_validations_per_minute
    
def benchmark_lineage_tracking_overhead():
    """Measure processing overhead from lineage tracking."""
    max_overhead_percentage = 5_percent

# Memory Usage Testing
def test_metadata_memory_efficiency():
    """Monitor memory usage during large-scale metadata processing."""
    max_memory_overhead = 256_MB
```

### Security Testing
```python
# Security Validation
def test_metadata_sanitization():
    """Ensure all metadata is properly sanitized."""
    
def test_sensitive_data_protection():
    """Prevent sensitive information in metadata."""
    
def test_lineage_data_security():
    """Ensure lineage data doesn't expose sensitive information."""
    
def test_schema_injection_prevention():
    """Prevent schema injection attacks."""
```

## ADDITIONAL NOTES
**Security, performance, and operational considerations**

### Security Considerations
- **Data Sanitization**: Sanitize all metadata fields to prevent injection attacks
- **Sensitive Information**: Ensure no sensitive data is included in metadata
- **Access Control**: Implement proper access controls for metadata viewing and editing
- **Audit Logging**: Log all metadata modifications for security auditing
- **Schema Security**: Protect schema definitions from unauthorized modifications
- **Lineage Security**: Ensure processing lineage doesn't expose sensitive processing details

### Performance Optimization
- **Metadata Caching**: Cache frequently accessed metadata to improve performance
- **Batch Processing**: Optimize metadata enrichment for batch operations
- **Lazy Loading**: Implement lazy loading for optional metadata fields
- **Index Optimization**: Optimize metadata storage and retrieval indexes
- **Memory Management**: Efficient memory usage during large-scale metadata processing
- **Parallel Processing**: Implement parallel metadata enrichment for independent chunks

### Maintenance Requirements
- **Schema Versioning**: Maintain backward compatibility during schema evolution
- **Metadata Quality Monitoring**: Continuously monitor metadata quality and completeness
- **Lineage Retention**: Implement appropriate retention policies for processing lineage
- **Performance Monitoring**: Track metadata processing performance and optimization opportunities
- **Documentation**: Maintain comprehensive documentation of NIC Schema and metadata flows
- **Configuration Management**: Externalize all metadata configuration parameters
- **Validation Rules**: Regularly review and update metadata validation rules and quality standards