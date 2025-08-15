# Metadata Management - PRP

## ROLE
**Data Architect with Schema Design Expertise**

Specialist in metadata schema design, data lineage tracking, and information governance. Expert in implementing comprehensive metadata frameworks that ensure data quality, traceability, and compliance. Proficient in designing extensible schema structures that support evolving requirements while maintaining backward compatibility.

## OBJECTIVE
**Implement NIC Schema Compliant Metadata Management**

Create a metadata management system within Jupyter Notebook cells that:
* Enforces NIC Schema compliance across all pipeline stages
* Tracks complete data lineage from source to vector storage
* Manages document versioning and change detection
* Validates metadata integrity before storage
* Provides metadata enrichment and normalization
* Enables metadata-based search and filtering
* Supports schema evolution and migration

## MOTIVATION
**Data Governance for Intelligent Knowledge Management**

Comprehensive metadata management ensures data quality, enables accurate search results, and provides complete audit trails for compliance. Rich metadata enables sophisticated filtering, improves search relevance, and supports data governance requirements essential for enterprise knowledge management systems.

## CONTEXT
**NIC Schema Implementation Environment**

Technical requirements:
* Schema: NIC Schema for document metadata
* Compliance: Full metadata validation and enrichment
* Lineage: End-to-end processing lineage tracking
* Storage: JSON schema with validation
* Integration: All pipeline components
* Constraints: Jupyter Notebook implementation
* Evolution: Support for schema updates

## IMPLEMENTATION BLUEPRINT
**Metadata Management Architecture**

### Code Structure
```python
# Cell 9: Metadata Management (NIC Schema)
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import jsonschema
from enum import Enum

class DocumentStatus(Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    ARCHIVED = "archived"
    PENDING_REVIEW = "pending_review"

@dataclass
class NICMetadata:
    # Document identification
    document_id: str
    title: str
    description: Optional[str]
    filename: str
    file_type: str
    
    # Status and versioning
    status: DocumentStatus
    version: str
    is_latest: bool
    
    # Temporal metadata
    created_at: datetime
    modified_at: datetime
    approved_at: Optional[datetime]
    
    # Source information
    repository_url: str
    branch: str
    commit_id: str
    source_path: str
    
    # Processing lineage
    processing_metadata: Dict[str, Any]
    confidence_scores: Dict[str, float]
    
    # Content metadata
    language: str
    page_count: int
    section_count: int
    
    # Classification
    categories: List[str]
    tags: List[str]
    
    # Quality metrics
    completeness_score: float
    extraction_quality: float

class MetadataManager:
    def __init__(self, schema_path: Optional[str] = None):
        self.schema = self._load_nic_schema(schema_path)
        
    def _load_nic_schema(self, schema_path: Optional[str]) -> Dict[str, Any]:
        \"\"\"Load NIC Schema definition\"\"\"
        # Default NIC Schema structure
        return {
            "type": "object",
            "required": ["document_id", "title", "filename", "status", "created_at"],
            "properties": {
                "document_id": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1},
                "description": {"type": ["string", "null"]},
                "filename": {"type": "string", "minLength": 1},
                "file_type": {"type": "string"},
                "status": {"enum": ["draft", "approved", "archived", "pending_review"]},
                "version": {"type": "string"},
                "is_latest": {"type": "boolean"},
                "created_at": {"type": "string", "format": "date-time"},
                "modified_at": {"type": "string", "format": "date-time"},
                "approved_at": {"type": ["string", "null"], "format": "date-time"},
                "repository_url": {"type": "string", "format": "uri"},
                "branch": {"type": "string"},
                "commit_id": {"type": "string"},
                "source_path": {"type": "string"},
                "processing_metadata": {"type": "object"},
                "confidence_scores": {"type": "object"},
                "language": {"type": "string"},
                "page_count": {"type": "integer", "minimum": 0},
                "section_count": {"type": "integer", "minimum": 0},
                "categories": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "completeness_score": {"type": "number", "minimum": 0, "maximum": 1},
                "extraction_quality": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
    
    def create_document_metadata(self,
                                document: Document,
                                docling_result: Dict[str, Any]) -> NICMetadata:
        \"\"\"Create comprehensive document metadata\"\"\"
        
        # Extract processing metadata
        processing_meta = docling_result.get('processing_metadata', {})
        confidence_scores = processing_meta.get('confidence_scores', {})
        
        # Create metadata
        metadata = NICMetadata(
            document_id=document.id,
            title=self._extract_title(docling_result),
            description=self._extract_description(docling_result),
            filename=document.filename,
            file_type=document.format.name,
            
            status=DocumentStatus.APPROVED,  # From GitLab approved folder
            version="1.0",
            is_latest=True,
            
            created_at=document.created_at,
            modified_at=document.modified_at,
            approved_at=document.created_at,  # Assume approved when in approved folder
            
            repository_url="http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git",
            branch=document.branch or "main",
            commit_id=document.commit_id or "",
            source_path=document.gitlab_path,
            
            processing_metadata=processing_meta,
            confidence_scores=confidence_scores,
            
            language=self._detect_language(docling_result),
            page_count=len(docling_result.get('structured_content', {}).get('page_structure', [])),
            section_count=len(docling_result.get('structured_content', {}).get('sections', [])),
            
            categories=self._extract_categories(docling_result),
            tags=self._extract_tags(docling_result),
            
            completeness_score=self._calculate_completeness(docling_result),
            extraction_quality=confidence_scores.get('overall', 1.0)
        )
        
        return metadata
    
    def validate_metadata(self, metadata: NICMetadata) -> Dict[str, Any]:
        \"\"\"Validate metadata against NIC Schema\"\"\"
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Convert to dict for validation
            metadata_dict = asdict(metadata)
            
            # Convert datetime objects to strings
            for key, value in metadata_dict.items():
                if isinstance(value, datetime):
                    metadata_dict[key] = value.isoformat()
                elif isinstance(value, DocumentStatus):
                    metadata_dict[key] = value.value
            
            # Validate against schema
            jsonschema.validate(metadata_dict, self.schema)
            
        except jsonschema.ValidationError as e:
            validation_result['valid'] = False
            validation_result['errors'].append(str(e))
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        # Additional business rule validation
        self._validate_business_rules(metadata, validation_result)
        
        return validation_result
    
    def _extract_title(self, docling_result: Dict[str, Any]) -> str:
        \"\"\"Extract document title from Docling result\"\"\"
        structured = docling_result.get('structured_content', {})
        
        # Try title from structured content
        if structured.get('title'):
            return structured['title']
        
        # Try first section heading
        sections = structured.get('sections', [])
        if sections:
            return sections[0].get('text', 'Untitled Document')
        
        # Fallback to filename
        return docling_result.get('filename', 'Untitled Document')
    
    def _extract_description(self, docling_result: Dict[str, Any]) -> Optional[str]:
        \"\"\"Extract document description\"\"\"
        structured = docling_result.get('structured_content', {})
        paragraphs = structured.get('paragraphs', [])
        
        if paragraphs:
            # Use first paragraph as description
            first_para = paragraphs[0].get('text', '')
            if len(first_para) > 500:
                return first_para[:497] + '...'
            return first_para
        
        return None
    
    def _detect_language(self, docling_result: Dict[str, Any]) -> str:
        \"\"\"Detect document language\"\"\"
        # Simple detection based on content
        markdown = docling_result.get('markdown', '')
        
        portuguese_indicators = ['ção', 'ões', 'ão', 'através', 'também', 'será', 'está']
        english_indicators = ['the', 'and', 'that', 'this', 'with', 'for', 'will', 'are']
        
        markdown_lower = markdown.lower()
        
        pt_count = sum(1 for word in portuguese_indicators if word in markdown_lower)
        en_count = sum(1 for word in english_indicators if word in markdown_lower)
        
        if pt_count > en_count:
            return 'pt'
        elif en_count > pt_count:
            return 'en'
        else:
            return 'mixed'
    
    def _extract_categories(self, docling_result: Dict[str, Any]) -> List[str]:
        \"\"\"Extract document categories based on content\"\"\"
        categories = []
        
        # Extract from filename and path
        filename = docling_result.get('filename', '').lower()
        
        # Define category mappings
        category_keywords = {
            'policy': ['política', 'policy', 'regulamento', 'norma'],
            'procedure': ['procedimento', 'procedure', 'processo', 'passo'],
            'manual': ['manual', 'guia', 'guide', 'instruções'],
            'report': ['relatório', 'report', 'análise', 'estudo'],
            'specification': ['especificação', 'specification', 'requisitos', 'requirements']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in filename for keyword in keywords):
                categories.append(category)
        
        # Default category if none found
        if not categories:
            categories.append('document')
        
        return categories
    
    def _extract_tags(self, docling_result: Dict[str, Any]) -> List[str]:
        \"\"\"Extract relevant tags from content\"\"\"
        tags = []
        
        # Extract from structured content
        structured = docling_result.get('structured_content', {})
        
        # Add processing-based tags
        if docling_result.get('processing_metadata', {}).get('ocr_applied'):
            tags.append('ocr-processed')
        
        if structured.get('tables'):
            tags.append('contains-tables')
        
        if structured.get('figures'):
            tags.append('contains-figures')
        
        # Add language tag
        language = self._detect_language(docling_result)
        tags.append(f'lang-{language}')
        
        return tags
    
    def _calculate_completeness(self, docling_result: Dict[str, Any]) -> float:
        \"\"\"Calculate metadata completeness score\"\"\"
        structured = docling_result.get('structured_content', {})
        
        # Count available data points
        score = 0.0
        max_score = 10.0
        
        # Title presence
        if structured.get('title'):
            score += 2.0
        
        # Content presence
        if structured.get('paragraphs'):
            score += 3.0
        
        # Structure presence
        if structured.get('sections'):
            score += 2.0
        
        # Metadata presence
        if docling_result.get('processing_metadata'):
            score += 2.0
        
        # Confidence scores
        if docling_result.get('processing_metadata', {}).get('confidence_scores'):
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    def _validate_business_rules(self, metadata: NICMetadata, validation_result: Dict[str, Any]):
        \"\"\"Validate business-specific rules\"\"\"
        
        # Check required fields for approved documents
        if metadata.status == DocumentStatus.APPROVED:
            if not metadata.title or len(metadata.title.strip()) < 3:
                validation_result['warnings'].append("Approved documents should have meaningful titles")
            
            if metadata.extraction_quality < 0.8:
                validation_result['warnings'].append("Low extraction quality for approved document")
        
        # Check confidence scores
        if metadata.confidence_scores.get('overall', 1.0) < 0.7:
            validation_result['warnings'].append("Low confidence scores may indicate processing issues")
        
        # Check completeness
        if metadata.completeness_score < 0.8:
            validation_result['warnings'].append("Incomplete metadata may affect search quality")

class MetadataEnricher:
    \"\"\"Enrich metadata with additional information\"\"\"
    
    @staticmethod
    def enrich_metadata(metadata: NICMetadata, 
                       additional_data: Dict[str, Any]) -> NICMetadata:
        \"\"\"Enrich metadata with external data\"\"\"
        
        # Update categories based on external classification
        if 'auto_categories' in additional_data:
            metadata.categories.extend(additional_data['auto_categories'])
            metadata.categories = list(set(metadata.categories))  # Remove duplicates
        
        # Add automatic tags
        if 'auto_tags' in additional_data:
            metadata.tags.extend(additional_data['auto_tags'])
            metadata.tags = list(set(metadata.tags))
        
        # Update confidence if external validation available
        if 'external_quality_score' in additional_data:
            external_score = additional_data['external_quality_score']
            metadata.extraction_quality = (metadata.extraction_quality + external_score) / 2
        
        return metadata
```

## VALIDATION LOOP

### Unit Testing
```python
def test_metadata_creation():
    \"\"\"Test metadata creation from document and Docling result\"\"\"
    manager = MetadataManager()
    
    document = create_test_document()
    docling_result = create_test_docling_result()
    
    metadata = manager.create_document_metadata(document, docling_result)
    
    assert metadata.document_id == document.id
    assert metadata.filename == document.filename
    assert metadata.status == DocumentStatus.APPROVED

def test_schema_validation():
    \"\"\"Test metadata validation against NIC Schema\"\"\"
    manager = MetadataManager()
    
    valid_metadata = create_valid_test_metadata()
    validation = manager.validate_metadata(valid_metadata)
    
    assert validation['valid'] == True
    assert len(validation['errors']) == 0

def test_language_detection():
    \"\"\"Test language detection functionality\"\"\"
    manager = MetadataManager()
    
    portuguese_result = {'markdown': 'Este é um documento em português com informações importantes.'}
    english_result = {'markdown': 'This is an English document with important information.'}
    
    assert manager._detect_language(portuguese_result) == 'pt'
    assert manager._detect_language(english_result) == 'en'
```

## ADDITIONAL NOTES

### Security Considerations
* **Schema Validation**: Prevent injection through metadata fields
* **Access Control**: Implement metadata-based access controls
* **Audit Trail**: Log all metadata changes
* **Data Privacy**: Handle PII in metadata appropriately

### Performance Optimization
* **Caching**: Cache validated metadata structures
* **Batch Processing**: Validate multiple metadata objects together
* **Indexing**: Create metadata indexes for fast filtering

### Maintenance Requirements
* **Schema Evolution**: Support backward-compatible schema updates
* **Quality Monitoring**: Track metadata quality metrics
* **Documentation**: Maintain comprehensive schema documentation