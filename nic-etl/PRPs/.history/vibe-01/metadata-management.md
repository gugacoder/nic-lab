# Metadata Management - PRP

## ROLE
**Data Schema Engineer with Metadata Architecture expertise**

Specialized in schema design, data validation, and metadata governance. Responsible for implementing the NIC Schema specification, ensuring data consistency, provenance tracking, and providing comprehensive validation frameworks for metadata compliance across the entire ETL pipeline.

## OBJECTIVE
**NIC Schema Implementation and Validation Framework**

Deliver a production-ready Python module that:
- Implements the complete NIC Schema specification for document metadata
- Provides comprehensive validation for all metadata fields and types
- Ensures metadata consistency and integrity across pipeline stages
- Supports schema evolution and version management
- Implements provenance tracking and lineage management
- Provides metadata enrichment and augmentation capabilities
- Enables metadata querying and filtering for search operations

## MOTIVATION
**Data Quality and Governance Foundation**

Robust metadata management ensures data quality, compliance, and effective information retrieval. By implementing comprehensive NIC Schema validation and metadata enrichment, this module guarantees consistent data structure, enables effective search filtering, and provides the foundation for data governance and audit requirements.

## CONTEXT
**NIC Schema Specification Architecture**

- **Schema Version**: NIC Schema v1.0 with extensibility support
- **Validation Requirements**: Strict field validation, type checking, business rules
- **Lineage Tracking**: Complete processing provenance from source to vector storage
- **Integration**: Metadata flows through all pipeline stages
- **Performance**: Efficient validation with minimal processing overhead
- **Compliance**: Audit trail and data governance requirements

## IMPLEMENTATION BLUEPRINT
**Comprehensive Metadata Management Module**

### Architecture Overview
```python
# Module Structure: modules/metadata_management.py
class NICSchemaManager:
    """NIC Schema implementation and validation framework"""
    
    def __init__(self, schema_version: str = "1.0")
    def validate_document_metadata(self, metadata: Dict[str, Any]) -> ValidationResult
    def enrich_metadata(self, base_metadata: Dict[str, Any], context: EnrichmentContext) -> Dict[str, Any]
    def create_lineage_record(self, processing_stage: str, metadata: Dict[str, Any]) -> LineageRecord
    def merge_metadata(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]
    def extract_search_facets(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]
```

### Code Structure
**File Organization**: `modules/metadata_management.py`
```python
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging
from pathlib import Path

class MetadataStatus(Enum):
    """Document processing and approval status"""
    DRAFT = "draft"
    APPROVED = "approved"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class ProcessingStage(Enum):
    """Pipeline processing stages for lineage tracking"""
    INGESTION = "ingestion"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"

@dataclass
class NICSchemaFields:
    """Complete NIC Schema field definitions"""
    
    # Document Identity
    document_id: str = ""
    title: str = ""
    description: str = ""
    file_path: str = ""
    file_name: str = ""
    file_extension: str = ""
    file_size_bytes: int = 0
    file_hash: str = ""
    
    # Document Classification
    document_type: str = ""
    category: str = ""
    subcategory: str = ""
    tags: List[str] = field(default_factory=list)
    status: str = MetadataStatus.DRAFT.value
    
    # Temporal Information
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    approved_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    
    # Authorship and Responsibility
    author: str = ""
    reviewer: str = ""
    approver: str = ""
    department: str = ""
    organization: str = "NIC"
    
    # Content Structure
    language: str = "pt-BR"
    page_count: int = 0
    section_count: int = 0
    total_tokens: int = 0
    
    # Version Control
    version: str = "1.0"
    revision: int = 1
    is_latest: bool = True
    previous_version: Optional[str] = None
    
    # Processing Metadata
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_pipeline_version: str = "1.0"
    ocr_applied: bool = False
    ocr_confidence: float = 0.0
    quality_score: float = 0.0
    
    # GitLab Integration
    gitlab_project: str = ""
    gitlab_branch: str = "main"
    gitlab_commit: str = ""
    gitlab_url: str = ""
    gitlab_file_path: str = ""
    
    # Qdrant Integration
    qdrant_collection: str = "nic"
    total_chunks: int = 0
    vector_dimension: int = 1024
    embedding_model: str = "BAAI/bge-m3"

@dataclass
class LineageRecord:
    """Processing lineage tracking record"""
    stage: ProcessingStage
    timestamp: datetime
    input_metadata: Dict[str, Any]
    output_metadata: Dict[str, Any]
    processing_parameters: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Metadata validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    invalid_types: List[str] = field(default_factory=list)
    business_rule_violations: List[str] = field(default_factory=list)

@dataclass
class EnrichmentContext:
    """Context for metadata enrichment"""
    processing_stage: ProcessingStage
    source_metadata: Dict[str, Any]
    processing_results: Dict[str, Any]
    pipeline_config: Dict[str, Any]

class NICSchemaManager:
    """Production-ready NIC Schema management and validation"""
    
    SCHEMA_VERSION = "1.0"
    
    # Required fields for different validation levels
    CORE_REQUIRED_FIELDS = {
        'document_id', 'title', 'file_path', 'file_name', 'status',
        'created_date', 'author', 'organization', 'processing_timestamp'
    }
    
    EXTENDED_REQUIRED_FIELDS = CORE_REQUIRED_FIELDS | {
        'description', 'document_type', 'category', 'language',
        'version', 'gitlab_project', 'gitlab_branch'
    }
    
    # Field type mappings
    FIELD_TYPES = {
        'document_id': str,
        'title': str,
        'description': str,
        'file_path': str,
        'file_name': str,
        'file_extension': str,
        'file_size_bytes': int,
        'file_hash': str,
        'document_type': str,
        'category': str,
        'subcategory': str,
        'tags': list,
        'status': str,
        'author': str,
        'reviewer': str,
        'approver': str,
        'department': str,
        'organization': str,
        'language': str,
        'page_count': int,
        'section_count': int,
        'total_tokens': int,
        'version': str,
        'revision': int,
        'is_latest': bool,
        'previous_version': (str, type(None)),
        'processing_pipeline_version': str,
        'ocr_applied': bool,
        'ocr_confidence': float,
        'quality_score': float,
        'gitlab_project': str,
        'gitlab_branch': str,
        'gitlab_commit': str,
        'gitlab_url': str,
        'gitlab_file_path': str,
        'qdrant_collection': str,
        'total_chunks': int,
        'vector_dimension': int,
        'embedding_model': str
    }
    
    # Valid values for enumerated fields
    VALID_VALUES = {
        'status': [status.value for status in MetadataStatus],
        'language': ['pt-BR', 'en-US', 'es-ES'],
        'organization': ['NIC', 'Partner'],
        'document_type': ['policy', 'procedure', 'guideline', 'manual', 'report', 'presentation', 'form'],
        'embedding_model': ['BAAI/bge-m3', 'text-embedding-ada-002']
    }
    
    def __init__(self, schema_version: str = SCHEMA_VERSION):
        self.schema_version = schema_version
        self.logger = logging.getLogger(__name__)
        self.lineage_records = []
        
    def validate_document_metadata(self, metadata: Dict[str, Any], 
                                 validation_level: str = "extended") -> ValidationResult:
        """Comprehensive metadata validation against NIC Schema"""
        
        errors = []
        warnings = []
        missing_fields = []
        invalid_types = []
        business_rule_violations = []
        
        # Determine required fields based on validation level
        required_fields = (self.EXTENDED_REQUIRED_FIELDS 
                          if validation_level == "extended" 
                          else self.CORE_REQUIRED_FIELDS)
        
        # Check required fields
        for field in required_fields:
            if field not in metadata or metadata[field] is None:
                missing_fields.append(field)
                errors.append(f"Required field missing: {field}")
        
        # Validate field types
        for field, value in metadata.items():
            if field in self.FIELD_TYPES:
                expected_type = self.FIELD_TYPES[field]
                
                # Handle optional fields (can be None)
                if isinstance(expected_type, tuple):
                    if value is not None and not isinstance(value, expected_type[0]):
                        invalid_types.append(field)
                        errors.append(f"Field {field} has invalid type: expected {expected_type[0].__name__}, got {type(value).__name__}")
                else:
                    if value is not None and not isinstance(value, expected_type):
                        try:
                            # Attempt type conversion for basic types
                            if expected_type in [int, float, str, bool]:
                                metadata[field] = expected_type(value)
                                warnings.append(f"Converted {field} to {expected_type.__name__}")
                            else:
                                invalid_types.append(field)
                                errors.append(f"Field {field} has invalid type: expected {expected_type.__name__}, got {type(value).__name__}")
                        except (ValueError, TypeError):
                            invalid_types.append(field)
                            errors.append(f"Field {field} cannot be converted to {expected_type.__name__}")
        
        # Validate enumerated values
        for field, valid_values in self.VALID_VALUES.items():
            if field in metadata and metadata[field] not in valid_values:
                business_rule_violations.append(field)
                errors.append(f"Field {field} has invalid value: {metadata[field]}. Valid values: {valid_values}")
        
        # Business rule validations
        business_rule_violations.extend(self._validate_business_rules(metadata))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            missing_fields=missing_fields,
            invalid_types=invalid_types,
            business_rule_violations=business_rule_violations
        )
    
    def _validate_business_rules(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate NIC-specific business rules"""
        
        violations = []
        
        # Rule: Approved documents must have approver
        if metadata.get('status') == MetadataStatus.APPROVED.value:
            if not metadata.get('approver'):
                violations.append("Approved documents must have an approver specified")
        
        # Rule: File size should be reasonable
        file_size = metadata.get('file_size_bytes', 0)
        if file_size > 100 * 1024 * 1024:  # 100MB
            violations.append(f"File size {file_size} bytes exceeds recommended limit")
        
        # Rule: Page count should be consistent with content
        page_count = metadata.get('page_count', 0)
        total_tokens = metadata.get('total_tokens', 0)
        if page_count > 0 and total_tokens > 0:
            tokens_per_page = total_tokens / page_count
            if tokens_per_page > 2000:  # Very high density
                violations.append("Token density per page seems unusually high")
        
        # Rule: Quality score should be within valid range
        quality_score = metadata.get('quality_score', 0.0)
        if quality_score < 0.0 or quality_score > 1.0:
            violations.append("Quality score must be between 0.0 and 1.0")
        
        # Rule: OCR confidence validation
        if metadata.get('ocr_applied', False):
            ocr_confidence = metadata.get('ocr_confidence', 0.0)
            if ocr_confidence < 0.5:
                violations.append("OCR confidence below recommended threshold (0.5)")
        
        # Rule: Vector dimension must match embedding model
        vector_dim = metadata.get('vector_dimension', 0)
        embedding_model = metadata.get('embedding_model', '')
        if embedding_model == 'BAAI/bge-m3' and vector_dim != 1024:
            violations.append("BAAI/bge-m3 model requires 1024-dimensional vectors")
        
        return violations
    
    def enrich_metadata(self, base_metadata: Dict[str, Any], 
                       context: EnrichmentContext) -> Dict[str, Any]:
        """Enrich metadata with additional context and computed fields"""
        
        enriched = base_metadata.copy()
        
        # Add processing timestamp if not present
        if 'processing_timestamp' not in enriched:
            enriched['processing_timestamp'] = datetime.utcnow().isoformat()
        
        # Enrich based on processing stage
        if context.processing_stage == ProcessingStage.INGESTION:
            self._enrich_ingestion_metadata(enriched, context)
        elif context.processing_stage == ProcessingStage.PROCESSING:
            self._enrich_processing_metadata(enriched, context)
        elif context.processing_stage == ProcessingStage.CHUNKING:
            self._enrich_chunking_metadata(enriched, context)
        elif context.processing_stage == ProcessingStage.EMBEDDING:
            self._enrich_embedding_metadata(enriched, context)
        elif context.processing_stage == ProcessingStage.STORAGE:
            self._enrich_storage_metadata(enriched, context)
        
        # Add computed fields
        enriched.update(self._compute_derived_fields(enriched))
        
        return enriched
    
    def _enrich_ingestion_metadata(self, metadata: Dict[str, Any], context: EnrichmentContext):
        """Enrich metadata during ingestion stage"""
        
        source_meta = context.source_metadata
        
        # Extract file information
        if 'file_path' in source_meta:
            file_path = Path(source_meta['file_path'])
            metadata['file_name'] = file_path.name
            metadata['file_extension'] = file_path.suffix.lower()
        
        # GitLab lineage information
        metadata.update({
            'gitlab_project': source_meta.get('project_path', ''),
            'gitlab_branch': source_meta.get('branch', 'main'),
            'gitlab_commit': source_meta.get('commit_sha', ''),
            'gitlab_url': source_meta.get('gitlab_url', ''),
            'gitlab_file_path': source_meta.get('file_path', '')
        })
        
        # Processing pipeline version
        metadata['processing_pipeline_version'] = context.pipeline_config.get('version', '1.0')
    
    def _enrich_processing_metadata(self, metadata: Dict[str, Any], context: EnrichmentContext):
        """Enrich metadata during document processing stage"""
        
        results = context.processing_results
        
        # OCR information
        metadata['ocr_applied'] = results.get('ocr_applied', False)
        metadata['ocr_confidence'] = results.get('ocr_confidence', 0.0)
        
        # Document structure information
        metadata['page_count'] = results.get('page_count', 0)
        metadata['section_count'] = results.get('section_count', 0)
        
        # Quality assessment
        metadata['quality_score'] = results.get('quality_score', 0.0)
    
    def _enrich_chunking_metadata(self, metadata: Dict[str, Any], context: EnrichmentContext):
        """Enrich metadata during chunking stage"""
        
        results = context.processing_results
        
        # Chunking information
        metadata['total_chunks'] = results.get('total_chunks', 0)
        metadata['total_tokens'] = results.get('total_tokens', 0)
        
        # Chunking configuration
        chunking_config = context.pipeline_config.get('chunking', {})
        metadata['chunk_size'] = chunking_config.get('target_chunk_size', 500)
        metadata['chunk_overlap'] = chunking_config.get('overlap_size', 100)
    
    def _enrich_embedding_metadata(self, metadata: Dict[str, Any], context: EnrichmentContext):
        """Enrich metadata during embedding generation stage"""
        
        results = context.processing_results
        
        # Embedding model information
        metadata['embedding_model'] = results.get('model_name', 'BAAI/bge-m3')
        metadata['vector_dimension'] = results.get('vector_dimension', 1024)
        
        # Processing performance
        metadata['embedding_generation_time'] = results.get('processing_time', 0.0)
        metadata['embedding_batch_size'] = results.get('batch_size', 32)
    
    def _enrich_storage_metadata(self, metadata: Dict[str, Any], context: EnrichmentContext):
        """Enrich metadata during storage stage"""
        
        results = context.processing_results
        
        # Qdrant storage information
        metadata['qdrant_collection'] = results.get('collection_name', 'nic')
        metadata['storage_timestamp'] = datetime.utcnow().isoformat()
        
        # Final processing status
        metadata['processing_complete'] = True
        metadata['final_quality_score'] = results.get('final_quality_score', 0.0)
    
    def _compute_derived_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived metadata fields"""
        
        derived = {}
        
        # Generate document ID if not present
        if 'document_id' not in metadata:
            id_components = [
                metadata.get('file_path', ''),
                metadata.get('gitlab_commit', ''),
                str(metadata.get('processing_timestamp', ''))
            ]
            id_string = '|'.join(id_components)
            derived['document_id'] = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        
        # Compute content density metrics
        total_tokens = metadata.get('total_tokens', 0)
        page_count = metadata.get('page_count', 0)
        if page_count > 0 and total_tokens > 0:
            derived['tokens_per_page'] = total_tokens / page_count
        
        # Compute processing efficiency metrics
        total_chunks = metadata.get('total_chunks', 0)
        if total_chunks > 0 and total_tokens > 0:
            derived['avg_tokens_per_chunk'] = total_tokens / total_chunks
        
        # Compute document complexity score
        factors = [
            metadata.get('page_count', 0) / 10,  # Normalize by typical page count
            metadata.get('section_count', 0) / 5,  # Normalize by typical section count
            1.0 if metadata.get('ocr_applied', False) else 0.0,  # OCR adds complexity
            len(metadata.get('tags', [])) / 5  # Tag complexity
        ]
        derived['complexity_score'] = min(1.0, sum(factors) / len(factors))
        
        return derived
    
    def create_lineage_record(self, processing_stage: ProcessingStage, 
                            input_metadata: Dict[str, Any],
                            output_metadata: Dict[str, Any],
                            processing_params: Dict[str, Any],
                            performance_metrics: Dict[str, Any]) -> LineageRecord:
        """Create processing lineage record"""
        
        lineage_record = LineageRecord(
            stage=processing_stage,
            timestamp=datetime.utcnow(),
            input_metadata=input_metadata,
            output_metadata=output_metadata,
            processing_parameters=processing_params,
            performance_metrics=performance_metrics
        )
        
        self.lineage_records.append(lineage_record)
        return lineage_record
    
    def merge_metadata(self, sources: List[Dict[str, Any]], 
                      priority_order: Optional[List[str]] = None) -> Dict[str, Any]:
        """Merge metadata from multiple sources with priority handling"""
        
        if not sources:
            return {}
        
        # Default priority order
        if priority_order is None:
            priority_order = ['user_provided', 'computed', 'extracted', 'default']
        
        merged = {}
        
        # Group sources by priority
        prioritized_sources = {}
        for source in sources:
            source_type = source.get('_source_type', 'default')
            if source_type not in prioritized_sources:
                prioritized_sources[source_type] = []
            prioritized_sources[source_type].append(source)
        
        # Merge in priority order
        for priority in reversed(priority_order):  # Start with lowest priority
            if priority in prioritized_sources:
                for source in prioritized_sources[priority]:
                    for key, value in source.items():
                        if key != '_source_type' and value is not None:
                            merged[key] = value
        
        return merged
    
    def extract_search_facets(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract search facets from metadata for filtering"""
        
        facets = {}
        
        # Direct categorical facets
        categorical_fields = [
            'status', 'document_type', 'category', 'subcategory',
            'author', 'department', 'organization', 'language'
        ]
        
        for field in categorical_fields:
            if field in metadata and metadata[field]:
                facets[field] = [str(metadata[field])]
        
        # Multi-value facets
        if 'tags' in metadata and metadata['tags']:
            facets['tags'] = metadata['tags']
        
        # Computed facets
        if 'created_date' in metadata and metadata['created_date']:
            date_obj = metadata['created_date']
            if isinstance(date_obj, str):
                date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
            
            facets['creation_year'] = [str(date_obj.year)]
            facets['creation_month'] = [f"{date_obj.year}-{date_obj.month:02d}"]
        
        # Quality facets
        quality_score = metadata.get('quality_score', 0.0)
        if quality_score >= 0.8:
            facets['quality_tier'] = ['high']
        elif quality_score >= 0.6:
            facets['quality_tier'] = ['medium']
        else:
            facets['quality_tier'] = ['low']
        
        # Processing facets
        if metadata.get('ocr_applied', False):
            facets['processing_type'] = ['ocr_processed']
        else:
            facets['processing_type'] = ['digital_native']
        
        return facets

def create_nic_schema_manager(schema_version: str = "1.0") -> NICSchemaManager:
    """Factory function for NIC Schema manager creation"""
    return NICSchemaManager(schema_version=schema_version)
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_metadata_management.py
import pytest
from datetime import datetime
from modules.metadata_management import NICSchemaManager, MetadataStatus, ProcessingStage

class TestNICSchemaManager:
    
    @pytest.fixture
    def schema_manager(self):
        return NICSchemaManager()
    
    @pytest.fixture
    def valid_metadata(self):
        return {
            'document_id': 'test_doc_001',
            'title': 'Test Document',
            'description': 'A test document for validation',
            'file_path': '/path/to/test.pdf',
            'file_name': 'test.pdf',
            'status': MetadataStatus.APPROVED.value,
            'created_date': datetime.utcnow(),
            'author': 'Test Author',
            'organization': 'NIC',
            'processing_timestamp': datetime.utcnow(),
            'document_type': 'policy',
            'category': 'technical',
            'language': 'pt-BR',
            'version': '1.0',
            'gitlab_project': 'nic/docs',
            'gitlab_branch': 'main'
        }
    
    def test_valid_metadata_validation(self, schema_manager, valid_metadata):
        """Test validation of valid metadata"""
        result = schema_manager.validate_document_metadata(valid_metadata)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.missing_fields) == 0
    
    def test_missing_required_fields(self, schema_manager):
        """Test validation with missing required fields"""
        incomplete_metadata = {
            'title': 'Test Document'
        }
        
        result = schema_manager.validate_document_metadata(incomplete_metadata)
        
        assert result.is_valid is False
        assert len(result.missing_fields) > 0
        assert 'document_id' in result.missing_fields
    
    def test_invalid_field_types(self, schema_manager, valid_metadata):
        """Test validation with invalid field types"""
        invalid_metadata = valid_metadata.copy()
        invalid_metadata['page_count'] = "not_a_number"
        
        result = schema_manager.validate_document_metadata(invalid_metadata)
        
        # Should attempt type conversion
        assert 'page_count' in result.warnings[0] if result.warnings else True
    
    def test_business_rule_validation(self, schema_manager, valid_metadata):
        """Test business rule validation"""
        # Test approved document without approver
        invalid_metadata = valid_metadata.copy()
        invalid_metadata['status'] = MetadataStatus.APPROVED.value
        del invalid_metadata['approver'] if 'approver' in invalid_metadata else None
        
        result = schema_manager.validate_document_metadata(invalid_metadata)
        
        assert len(result.business_rule_violations) > 0
    
    def test_metadata_enrichment(self, schema_manager, valid_metadata):
        """Test metadata enrichment functionality"""
        from modules.metadata_management import EnrichmentContext
        
        context = EnrichmentContext(
            processing_stage=ProcessingStage.PROCESSING,
            source_metadata={'ocr_applied': True},
            processing_results={'quality_score': 0.85},
            pipeline_config={'version': '1.0'}
        )
        
        enriched = schema_manager.enrich_metadata(valid_metadata, context)
        
        assert 'processing_timestamp' in enriched
        assert enriched['quality_score'] == 0.85
    
    def test_search_facet_extraction(self, schema_manager, valid_metadata):
        """Test search facet extraction"""
        facets = schema_manager.extract_search_facets(valid_metadata)
        
        assert 'status' in facets
        assert 'document_type' in facets
        assert 'author' in facets
        assert facets['status'] == [MetadataStatus.APPROVED.value]
```

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **Data Privacy**: Ensure sensitive metadata fields are handled securely
- **Access Control**: Implement field-level access controls for sensitive metadata
- **Audit Trails**: Maintain complete audit logs for metadata changes
- **Validation Security**: Prevent injection attacks through metadata validation

### Performance Optimization
- **Validation Caching**: Cache validation results for identical metadata
- **Batch Processing**: Support batch metadata validation and enrichment
- **Schema Compilation**: Pre-compile validation rules for better performance
- **Memory Efficiency**: Optimize metadata objects for memory usage

### Maintenance Requirements
- **Schema Evolution**: Support backward-compatible schema changes
- **Version Management**: Track schema versions and migration paths
- **Quality Monitoring**: Monitor metadata quality and compliance metrics
- **Documentation**: Maintain comprehensive schema documentation