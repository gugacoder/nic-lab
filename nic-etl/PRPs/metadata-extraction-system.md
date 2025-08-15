# Metadata Extraction System - PRP

## ROLE
**Python Developer with Metadata Management and Schema Design expertise**

Responsible for implementing comprehensive metadata extraction system that applies NIC Schema metadata to document chunks. Must have experience with JSON schema validation, document metadata extraction, lineage tracking, and hierarchical data structures.

## OBJECTIVE
**Extract and apply NIC Schema metadata to document chunks for enhanced searchability**

Develop a robust metadata extraction system that:
- Extracts document-level metadata following NIC Schema specification
- Generates section-level metadata for chunk categorization
- Implements lineage tracking (OCR status, repository info, versioning)
- Validates metadata against defined JSON schema
- Enriches chunks with hierarchical metadata structure
- Maintains metadata consistency across processing pipeline
- Provides metadata quality assessment and validation

Success criteria: Successfully extract and validate 100% of required metadata fields with >95% accuracy for document classification and lineage tracking.

## MOTIVATION
**Enable rich metadata-driven search and document organization**

Comprehensive metadata extraction transforms raw document chunks into structured, searchable knowledge units. By applying consistent schema and tracking document lineage, the system enables advanced filtering, categorization, and contextual search capabilities that significantly improve knowledge retrieval and management.

## CONTEXT
**NIC ETL Pipeline - Metadata Extraction Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- JSON Schema validation for NIC Schema compliance
- Input from text chunking and structure analysis phases
- Output to embedding generation and QDrant integration

NIC Schema Structure:
- Document metadata: title, description, status, created, author, tags, related, up
- Section metadata: section titles, hierarchy levels, element types
- Processing lineage: OCR status, repository info, commit details, processing timestamps

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Chunked Documents → Document Metadata Extraction → Section Metadata Generation → Lineage Tracking → Schema Validation → Enriched Chunks
```

### Code Structure
```python
# File organization
src/
├── metadata/
│   ├── __init__.py
│   ├── nic_schema_validator.py    # NIC Schema validation
│   ├── document_metadata.py       # Document-level metadata extraction
│   ├── section_metadata.py        # Section-level metadata generation
│   ├── lineage_tracker.py         # Processing lineage tracking
│   ├── metadata_enricher.py       # Chunk metadata enrichment
│   └── metadata_orchestrator.py   # Main metadata pipeline
├── schemas/
│   ├── nic_schema.json            # NIC Schema definition
│   └── validation_schemas.json    # Additional validation schemas
└── notebooks/
    └── 06_metadata_extraction.ipynb
```

### NIC Schema Implementation
```python
import json
import jsonschema
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

class NICSchemaValidator:
    """Validator for NIC Schema compliance"""
    
    def __init__(self, schema_path: str = "schemas/nic_schema.json"):
        self.logger = logging.getLogger(__name__)
        self.schema = self._load_schema(schema_path)
        self.validator = None
        self._initialize_validator()
    
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load NIC Schema definition"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load NIC Schema: {e}")
            # Fallback to embedded schema
            return self._get_embedded_schema()
    
    def _get_embedded_schema(self) -> Dict[str, Any]:
        """Embedded NIC Schema as fallback"""
        return {
            "document": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["rascunho", "revisão", "publicado", "arquivado"]
                    },
                    "created": {"type": "string", "format": "date"},
                    "author": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "related": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "up": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["title", "description", "status", "created"]
            },
            "sections": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "level": {"type": "integer", "minimum": 1},
                        "section_type": {"type": "string"},
                        "parent_section": {"type": "string"}
                    }
                }
            }
        }
    
    def _initialize_validator(self):
        """Initialize JSON Schema validator"""
        try:
            self.validator = jsonschema.Draft7Validator(self.schema)
        except Exception as e:
            self.logger.error(f"Validator initialization failed: {e}")
    
    def validate_document_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document metadata against NIC Schema"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            if not self.validator:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Schema validator not initialized")
                return validation_result
            
            # Validate document section
            if 'document' in metadata:
                doc_errors = list(self.validator.iter_errors(metadata['document']))
                if doc_errors:
                    validation_result['is_valid'] = False
                    validation_result['errors'].extend([error.message for error in doc_errors])
            else:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Missing required 'document' section")
            
            # Validate sections if present
            if 'sections' in metadata:
                self._validate_sections(metadata['sections'], validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Metadata validation failed: {e}")
            return {
                'is_valid': False,
                'errors': [str(e)]
            }
    
    def _validate_sections(self, sections: Dict[str, Any], validation_result: Dict[str, Any]):
        """Validate sections metadata"""
        for section_id, section_data in sections.items():
            if not isinstance(section_data, dict):
                validation_result['warnings'].append(f"Section {section_id} is not an object")
                continue
            
            # Check required section fields
            required_section_fields = ['title', 'level', 'section_type']
            for field in required_section_fields:
                if field not in section_data:
                    validation_result['warnings'].append(f"Section {section_id} missing field: {field}")

class DocumentMetadataExtractor:
    """Extract document-level metadata from various sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_document_metadata(self, document_data: Dict[str, Any], 
                                 source_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract comprehensive document metadata"""
        try:
            # Initialize metadata structure
            doc_metadata = {
                'document': {
                    'title': '',
                    'description': '',
                    'status': 'publicado',  # Default status
                    'created': datetime.now().isoformat()[:10],  # YYYY-MM-DD format
                    'author': [],
                    'tags': [],
                    'related': [],
                    'up': []
                },
                'extraction_metadata': {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'extraction_method': 'automated',
                    'confidence_score': 0.0
                }
            }
            
            # Extract title from multiple sources
            title = self._extract_title(document_data, source_metadata)
            if title:
                doc_metadata['document']['title'] = title
            
            # Extract description
            description = self._extract_description(document_data)
            if description:
                doc_metadata['document']['description'] = description
            
            # Extract creation date
            created_date = self._extract_creation_date(document_data, source_metadata)
            if created_date:
                doc_metadata['document']['created'] = created_date
            
            # Extract author information
            authors = self._extract_authors(document_data, source_metadata)
            if authors:
                doc_metadata['document']['author'] = authors
            
            # Extract tags
            tags = self._extract_tags(document_data)
            if tags:
                doc_metadata['document']['tags'] = tags
            
            # Extract related documents
            related = self._extract_related_documents(document_data)
            if related:
                doc_metadata['document']['related'] = related
            
            # Calculate confidence score
            doc_metadata['extraction_metadata']['confidence_score'] = self._calculate_extraction_confidence(doc_metadata['document'])
            
            return doc_metadata
            
        except Exception as e:
            self.logger.error(f"Document metadata extraction failed: {e}")
            return {'error': str(e)}
    
    def _extract_title(self, document_data: Dict[str, Any], 
                      source_metadata: Dict[str, Any] = None) -> str:
        """Extract document title from multiple sources"""
        # Priority order for title extraction
        title_sources = [
            # From document structure
            lambda: self._get_first_heading(document_data),
            # From filename
            lambda: self._extract_title_from_filename(source_metadata),
            # From document metadata
            lambda: self._get_document_property(document_data, 'title'),
            # From GitLab metadata
            lambda: source_metadata.get('filename', '').replace('.pdf', '').replace('.docx', '') if source_metadata else ''
        ]
        
        for extractor in title_sources:
            try:
                title = extractor()
                if title and len(title.strip()) > 3:
                    return title.strip()
            except:
                continue
        
        return "Documento sem título"
    
    def _get_first_heading(self, document_data: Dict[str, Any]) -> str:
        """Extract first heading as title"""
        # From structure analysis
        if 'hierarchy' in document_data:
            for child in document_data['hierarchy'].get('children', []):
                if child.get('type') in ['section', 'subsection'] and child.get('title'):
                    return child['title']
        
        # From elements
        if 'elements' in document_data:
            for element in document_data['elements']:
                if 'heading' in element.get('type', '') and element.get('content'):
                    return element['content']
        
        # From normalized text (first line if it looks like a title)
        if 'normalized_text' in document_data:
            lines = document_data['normalized_text'].split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if len(line) > 10 and len(line) < 100 and not line.endswith('.'):
                    return line
        
        return ""
    
    def _extract_description(self, document_data: Dict[str, Any]) -> str:
        """Extract document description"""
        # Try to find first substantial paragraph
        if 'elements' in document_data:
            for element in document_data['elements']:
                if element.get('type') == 'paragraph':
                    content = element.get('content', '').strip()
                    if len(content) > 50 and len(content) < 500:
                        return content
        
        # Fallback to first chunk or beginning of normalized text
        if 'normalized_text' in document_data:
            text = document_data['normalized_text'].strip()
            if text:
                # Take first 200 characters
                description = text[:200]
                # Cut at last complete sentence
                last_sentence = description.rfind('.')
                if last_sentence > 100:
                    description = description[:last_sentence + 1]
                return description
        
        return "Documento processado automaticamente"
    
    def _extract_creation_date(self, document_data: Dict[str, Any], 
                             source_metadata: Dict[str, Any] = None) -> str:
        """Extract document creation date"""
        # From source metadata (GitLab)
        if source_metadata:
            if 'last_modified' in source_metadata and source_metadata['last_modified']:
                try:
                    # Parse and format date
                    return source_metadata['last_modified'][:10]  # YYYY-MM-DD
                except:
                    pass
        
        # From document properties
        if 'metadata' in document_data:
            doc_meta = document_data['metadata']
            if 'created' in doc_meta:
                return doc_meta['created'][:10] if doc_meta['created'] else None
            if 'modified' in doc_meta:
                return doc_meta['modified'][:10] if doc_meta['modified'] else None
        
        # Default to current date
        return datetime.now().isoformat()[:10]
    
    def _extract_authors(self, document_data: Dict[str, Any], 
                        source_metadata: Dict[str, Any] = None) -> List[str]:
        """Extract document authors"""
        authors = []
        
        # From document metadata
        if 'metadata' in document_data:
            doc_meta = document_data['metadata']
            if 'author' in doc_meta and doc_meta['author']:
                if isinstance(doc_meta['author'], str):
                    authors.append(doc_meta['author'])
                elif isinstance(doc_meta['author'], list):
                    authors.extend(doc_meta['author'])
        
        # From source metadata (GitLab)
        if source_metadata and 'author' in source_metadata:
            authors.append(source_metadata['author'])
        
        # Clean and deduplicate
        clean_authors = list(set([author.strip() for author in authors if author and author.strip()]))
        
        return clean_authors if clean_authors else ["Sistema NIC"]
    
    def _extract_tags(self, document_data: Dict[str, Any]) -> List[str]:
        """Extract document tags based on content analysis"""
        tags = []
        
        # Analyze content for automatic tag generation
        if 'normalized_text' in document_data:
            text = document_data['normalized_text'].lower()
            
            # Technical domain tags
            technical_keywords = {
                'programação': ['python', 'java', 'javascript', 'código', 'desenvolvimento'],
                'dados': ['database', 'sql', 'dados', 'banco', 'tabela'],
                'sistema': ['sistema', 'arquitetura', 'infraestrutura', 'servidor'],
                'processo': ['processo', 'procedimento', 'workflow', 'fluxo'],
                'documentação': ['manual', 'guia', 'tutorial', 'instrução'],
                'segurança': ['segurança', 'autenticação', 'autorização', 'criptografia']
            }
            
            for tag, keywords in technical_keywords.items():
                if any(keyword in text for keyword in keywords):
                    tags.append(tag)
        
        # Limit to most relevant tags
        return tags[:5]
    
    def _calculate_extraction_confidence(self, doc_metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for metadata extraction"""
        score = 0.0
        
        # Required fields present
        required_fields = ['title', 'description', 'status', 'created']
        present_required = sum(1 for field in required_fields if doc_metadata.get(field))
        score += (present_required / len(required_fields)) * 40
        
        # Optional fields present
        optional_fields = ['author', 'tags', 'related']
        present_optional = sum(1 for field in optional_fields if doc_metadata.get(field))
        score += (present_optional / len(optional_fields)) * 30
        
        # Content quality indicators
        if doc_metadata.get('title') and len(doc_metadata['title']) > 10:
            score += 15
        
        if doc_metadata.get('description') and len(doc_metadata['description']) > 50:
            score += 15
        
        return min(100.0, score)
```

### Section Metadata Generator
```python
from typing import Dict, Any, List

class SectionMetadataGenerator:
    """Generate section-level metadata for chunks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_section_metadata(self, chunks: List[Dict[str, Any]], 
                                 document_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Generate section metadata for document chunks"""
        try:
            section_metadata = {
                'sections': {},
                'chunk_section_mapping': {},
                'section_statistics': {}
            }
            
            # Extract sections from document structure
            sections = self._extract_sections_from_structure(document_structure)
            
            # Generate metadata for each section
            for section_id, section_info in sections.items():
                section_metadata['sections'][section_id] = self._create_section_metadata(section_info)
            
            # Map chunks to sections
            section_metadata['chunk_section_mapping'] = self._map_chunks_to_sections(chunks, sections)
            
            # Generate section statistics
            section_metadata['section_statistics'] = self._calculate_section_statistics(
                section_metadata['sections'], section_metadata['chunk_section_mapping']
            )
            
            return section_metadata
            
        except Exception as e:
            self.logger.error(f"Section metadata generation failed: {e}")
            return {'error': str(e)}
    
    def _extract_sections_from_structure(self, document_structure: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract section information from document structure"""
        sections = {}
        
        def process_hierarchy_node(node, parent_path="", level=0):
            if node.get('type') in ['section', 'subsection']:
                section_id = self._generate_section_id(node.get('title', ''), parent_path)
                
                sections[section_id] = {
                    'title': node.get('title', ''),
                    'level': level,
                    'section_type': node.get('type', 'section'),
                    'parent_section': parent_path if parent_path else None,
                    'path': f"{parent_path}/{section_id}" if parent_path else section_id,
                    'children': [],
                    'content_elements': node.get('children', [])
                }
                
                # Process children recursively
                for child in node.get('children', []):
                    if child.get('type') in ['section', 'subsection']:
                        child_section_id = process_hierarchy_node(child, section_id, level + 1)
                        if child_section_id:
                            sections[section_id]['children'].append(child_section_id)
                
                return section_id
            
            return None
        
        # Process hierarchy if available
        if 'hierarchy' in document_structure:
            for child in document_structure['hierarchy'].get('children', []):
                process_hierarchy_node(child)
        
        # Fallback: create sections from document_sections
        if not sections and 'document_sections' in document_structure:
            for i, section in enumerate(document_structure['document_sections']):
                section_id = f"section_{i}"
                sections[section_id] = {
                    'title': section.get('title', f'Seção {i+1}'),
                    'level': section.get('level', 1),
                    'section_type': section.get('section_type', 'section'),
                    'parent_section': None,
                    'path': section_id,
                    'children': [],
                    'content_elements': section.get('content_elements', [])
                }
        
        return sections
    
    def _create_section_metadata(self, section_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metadata for a section"""
        return {
            'title': section_info.get('title', ''),
            'level': section_info.get('level', 1),
            'section_type': section_info.get('section_type', 'section'),
            'parent_section': section_info.get('parent_section'),
            'path': section_info.get('path', ''),
            'children': section_info.get('children', []),
            'element_count': len(section_info.get('content_elements', [])),
            'section_characteristics': self._analyze_section_characteristics(section_info)
        }
    
    def _analyze_section_characteristics(self, section_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze section characteristics for better categorization"""
        characteristics = {
            'content_types': [],
            'estimated_complexity': 'medium',
            'primary_purpose': 'informational'
        }
        
        # Analyze content elements
        content_elements = section_info.get('content_elements', [])
        
        element_types = [elem.get('type') for elem in content_elements if elem.get('type')]
        characteristics['content_types'] = list(set(element_types))
        
        # Estimate complexity based on content
        total_content = ' '.join([elem.get('content', '') for elem in content_elements])
        
        if len(total_content) > 2000:
            characteristics['estimated_complexity'] = 'high'
        elif len(total_content) < 500:
            characteristics['estimated_complexity'] = 'low'
        
        # Determine primary purpose (heuristic)
        title = section_info.get('title', '').lower()
        if any(word in title for word in ['introdução', 'overview', 'visão']):
            characteristics['primary_purpose'] = 'introductory'
        elif any(word in title for word in ['procedimento', 'steps', 'como']):
            characteristics['primary_purpose'] = 'procedural'
        elif any(word in title for word in ['resultado', 'conclusão', 'summary']):
            characteristics['primary_purpose'] = 'conclusive'
        
        return characteristics
    
    def _map_chunks_to_sections(self, chunks: List[Dict[str, Any]], 
                               sections: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Map each chunk to its corresponding section"""
        chunk_section_mapping = {}
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            section_id = self._determine_chunk_section(chunk, sections)
            chunk_section_mapping[chunk_id] = section_id
        
        return chunk_section_mapping
    
    def _determine_chunk_section(self, chunk: Dict[str, Any], 
                               sections: Dict[str, Dict[str, Any]]) -> str:
        """Determine which section a chunk belongs to"""
        # Check if chunk has explicit section information
        chunk_metadata = chunk.get('metadata', {})
        
        if 'section_path' in chunk_metadata:
            section_path = chunk_metadata['section_path']
            # Find matching section
            for section_id, section_info in sections.items():
                if section_path in section_info.get('path', ''):
                    return section_id
        
        # Fallback: match by paragraph indices or content similarity
        paragraph_indices = chunk_metadata.get('paragraph_indices', [])
        if paragraph_indices:
            # Simple heuristic: assign to first available section
            section_ids = list(sections.keys())
            if section_ids:
                return section_ids[0]
        
        return 'unknown_section'
```

### Lineage Tracker
```python
from typing import Dict, Any
from datetime import datetime
import uuid

class LineageTracker:
    """Track document processing lineage and metadata"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_lineage_metadata(self, processing_history: List[Dict[str, Any]], 
                              source_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create comprehensive lineage tracking metadata"""
        try:
            lineage_metadata = {
                'processing_lineage': {
                    'processing_id': str(uuid.uuid4()),
                    'processing_timestamp': datetime.now().isoformat(),
                    'processing_version': '1.0',
                    'pipeline_stages': processing_history,
                    'total_processing_time': self._calculate_total_processing_time(processing_history)
                },
                'source_lineage': self._create_source_lineage(source_metadata),
                'quality_metrics': self._calculate_quality_metrics(processing_history),
                'system_metadata': self._create_system_metadata()
            }
            
            return lineage_metadata
            
        except Exception as e:
            self.logger.error(f"Lineage metadata creation failed: {e}")
            return {'error': str(e)}
    
    def _create_source_lineage(self, source_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create source lineage information"""
        if not source_metadata:
            return {}
        
        return {
            'repository': {
                'url': source_metadata.get('gitlab_url', ''),
                'repository_path': source_metadata.get('repository', ''),
                'branch': source_metadata.get('branch', ''),
                'commit_id': source_metadata.get('commit_id', ''),
                'file_path': source_metadata.get('original_path', ''),
                'collection_timestamp': source_metadata.get('collected_at', '')
            },
            'document_properties': {
                'original_filename': source_metadata.get('filename', ''),
                'file_type': source_metadata.get('file_type', ''),
                'file_size_bytes': source_metadata.get('size_bytes', 0),
                'last_modified': source_metadata.get('last_modified', ''),
                'local_path': source_metadata.get('local_path', '')
            }
        }
    
    def _calculate_quality_metrics(self, processing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics from processing history"""
        quality_metrics = {
            'overall_quality_score': 0.0,
            'stage_quality_scores': {},
            'confidence_scores': {},
            'processing_success_rate': 0.0
        }
        
        total_stages = len(processing_history)
        successful_stages = 0
        quality_scores = []
        
        for stage in processing_history:
            stage_name = stage.get('stage', 'unknown')
            stage_success = stage.get('success', False)
            stage_quality = stage.get('quality_score', 0.0)
            
            if stage_success:
                successful_stages += 1
            
            quality_metrics['stage_quality_scores'][stage_name] = stage_quality
            
            if stage_quality > 0:
                quality_scores.append(stage_quality)
        
        # Calculate overall metrics
        quality_metrics['processing_success_rate'] = successful_stages / total_stages if total_stages > 0 else 0
        quality_metrics['overall_quality_score'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return quality_metrics
    
    def _create_system_metadata(self) -> Dict[str, Any]:
        """Create system-level metadata"""
        return {
            'nic_etl_version': '1.0',
            'python_version': '3.8+',
            'processing_environment': 'jupyter_notebook',
            'system_timestamp': datetime.now().isoformat(),
            'processor_id': f"nic-etl-{uuid.uuid4().hex[:8]}"
        }

class MetadataEnricher:
    """Enrich chunks with comprehensive metadata"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enrich_chunks_with_metadata(self, chunks: List[Dict[str, Any]], 
                                   document_metadata: Dict[str, Any],
                                   section_metadata: Dict[str, Any],
                                   lineage_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enrich chunks with comprehensive metadata"""
        try:
            enriched_chunks = []
            
            for chunk in chunks:
                enriched_chunk = chunk.copy()
                
                # Add document-level metadata
                enriched_chunk['document_metadata'] = document_metadata.get('document', {})
                
                # Add section-level metadata
                chunk_id = chunk.get('chunk_id', '')
                section_id = section_metadata.get('chunk_section_mapping', {}).get(chunk_id, 'unknown_section')
                enriched_chunk['section_metadata'] = section_metadata.get('sections', {}).get(section_id, {})
                
                # Add lineage metadata
                enriched_chunk['lineage_metadata'] = lineage_metadata
                
                # Create unified chunk metadata
                enriched_chunk['nic_metadata'] = self._create_unified_chunk_metadata(
                    enriched_chunk, section_id
                )
                
                enriched_chunks.append(enriched_chunk)
            
            return enriched_chunks
            
        except Exception as e:
            self.logger.error(f"Chunk metadata enrichment failed: {e}")
            return chunks  # Return original chunks if enrichment fails
    
    def _create_unified_chunk_metadata(self, chunk: Dict[str, Any], 
                                     section_id: str) -> Dict[str, Any]:
        """Create unified metadata structure for chunk"""
        return {
            'chunk_id': chunk.get('chunk_id', ''),
            'section_id': section_id,
            'document_title': chunk.get('document_metadata', {}).get('title', ''),
            'section_title': chunk.get('section_metadata', {}).get('title', ''),
            'section_path': chunk.get('section_metadata', {}).get('path', ''),
            'chunk_position': {
                'start_paragraph': chunk.get('start_paragraph', 0),
                'end_paragraph': chunk.get('end_paragraph', 0),
                'token_count': chunk.get('token_count', 0)
            },
            'quality_indicators': {
                'chunk_quality_score': chunk.get('quality_score', 0),
                'has_overlap': chunk.get('has_previous_overlap', False),
                'processing_confidence': chunk.get('lineage_metadata', {}).get('quality_metrics', {}).get('overall_quality_score', 0)
            },
            'processing_flags': {
                'ocr_applied': self._determine_ocr_applied(chunk),
                'structure_analyzed': self._determine_structure_analyzed(chunk),
                'is_latest_version': True  # Assume latest for now
            }
        }
    
    def _determine_ocr_applied(self, chunk: Dict[str, Any]) -> bool:
        """Determine if OCR was applied to this chunk"""
        processing_history = chunk.get('lineage_metadata', {}).get('processing_lineage', {}).get('pipeline_stages', [])
        return any(stage.get('stage') == 'ocr_processing' and stage.get('success', False) for stage in processing_history)
    
    def _determine_structure_analyzed(self, chunk: Dict[str, Any]) -> bool:
        """Determine if structure analysis was applied"""
        return chunk.get('section_metadata', {}).get('title') is not None
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from src.metadata.nic_schema_validator import NICSchemaValidator
from src.metadata.document_metadata import DocumentMetadataExtractor

class TestMetadataExtraction:
    def test_nic_schema_validation(self):
        validator = NICSchemaValidator()
        
        valid_metadata = {
            'document': {
                'title': 'Test Document',
                'description': 'Test description',
                'status': 'publicado',
                'created': '2023-01-01',
                'author': ['Test Author'],
                'tags': ['test'],
                'related': [],
                'up': []
            }
        }
        
        result = validator.validate_document_metadata(valid_metadata)
        assert result['is_valid'] == True
        assert len(result['errors']) == 0
    
    def test_document_metadata_extraction(self):
        extractor = DocumentMetadataExtractor()
        
        document_data = {
            'elements': [
                {'type': 'heading_1', 'content': 'Test Document Title'},
                {'type': 'paragraph', 'content': 'This is a test document for metadata extraction.'}
            ],
            'normalized_text': 'Test Document Title\n\nThis is a test document for metadata extraction.'
        }
        
        result = extractor.extract_document_metadata(document_data)
        assert 'document' in result
        assert result['document']['title'] == 'Test Document Title'
        assert len(result['document']['description']) > 20
    
    def test_lineage_tracking(self):
        from src.metadata.lineage_tracker import LineageTracker
        
        tracker = LineageTracker()
        processing_history = [
            {'stage': 'ingestion', 'success': True, 'quality_score': 95},
            {'stage': 'normalization', 'success': True, 'quality_score': 90}
        ]
        
        result = tracker.create_lineage_metadata(processing_history)
        assert 'processing_lineage' in result
        assert result['processing_lineage']['processing_id']
        assert result['quality_metrics']['processing_success_rate'] == 1.0
```

### Integration Testing
- End-to-end metadata extraction and validation workflow
- Schema compliance testing with various document types
- Lineage tracking accuracy across processing pipeline

### Performance Testing
- Process metadata for 1000+ chunks within 30 seconds
- Memory usage under 200MB for metadata operations
- Schema validation performance optimization

## ADDITIONAL NOTES

### Security Considerations
- Metadata sanitization to prevent injection attacks
- Sensitive information detection and redaction
- Audit logging for metadata access and modifications
- Secure storage of lineage and processing metadata

### Performance Optimization
- Caching of schema validation results
- Batch processing of metadata operations
- Efficient JSON schema compilation
- Memory-optimized metadata structures

### Maintenance Requirements
- Regular schema updates and version management
- Metadata quality monitoring and reporting
- Integration testing with downstream embedding systems
- Backup and recovery procedures for metadata