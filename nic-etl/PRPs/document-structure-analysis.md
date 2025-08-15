# Document Structure Analysis - PRP

## ROLE
**Python Developer with Document AI and NLP expertise**

Responsible for implementing intelligent document structure analysis using docling library to parse logical document elements (titles, sections, paragraphs, tables). Must have experience with document parsing, layout analysis, and structured text extraction.

## OBJECTIVE
**Extract logical document structure for enhanced semantic processing**

Develop a comprehensive structure analysis system that:
- Identifies document hierarchy (headings, sections, subsections)
- Extracts and classifies text elements (paragraphs, lists, tables, captions)
- Maintains spatial relationships and reading order
- Generates structured metadata for each document element
- Provides layout-aware text extraction with preserved formatting
- Maps document structure to semantic sections for improved chunking

Success criteria: Accurately identify >95% of document structure elements with proper hierarchical relationships and reading order preservation.

## MOTIVATION
**Enable context-aware document processing and intelligent chunking**

Understanding document structure allows for more intelligent text processing, where context is preserved through logical sections. This enables better chunking strategies that respect document boundaries, improves search relevance by maintaining semantic context, and provides richer metadata for AI-powered document analysis.

## CONTEXT
**NIC ETL Pipeline - Document Structure Analysis Phase**

Technology Stack:
- Python 3.8+ with jupyter notebook environment
- docling library for document structure analysis
- Input from normalized documents and OCR processing
- Output to text chunking and metadata extraction phases

Document Types:
- Business reports with complex layouts
- Technical documentation with nested sections
- Forms with structured fields
- Multi-column documents and presentations

Integration Requirements:
- Process both digital and OCR-processed documents
- Handle multilingual content (Portuguese/English)
- Generate structured output compatible with chunking strategy
- Maintain document lineage and processing metadata

## IMPLEMENTATION BLUEPRINT

### Architecture Overview
```
Input Documents → Document Loading → Layout Analysis → Element Classification → Hierarchy Building → Structured Output
```

### Code Structure
```python
# File organization
src/
├── structure_analysis/
│   ├── __init__.py
│   ├── docling_wrapper.py        # Docling integration
│   ├── layout_analyzer.py        # Layout analysis logic
│   ├── element_classifier.py     # Document element classification
│   ├── hierarchy_builder.py      # Document hierarchy construction
│   ├── structure_validator.py    # Structure validation and quality
│   └── output_formatter.py       # Structured output formatting
├── models/
│   └── document_structure.py     # Data models for structure
└── notebooks/
    └── 04_structure_analysis.ipynb
```

### Docling Integration
```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions
from typing import Dict, Any, List
import logging

class DocumentStructureAnalyzer:
    """Main interface for document structure analysis using docling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Configure docling pipeline
        self.pipeline_options = PipelineOptions()
        self.pipeline_options.do_ocr = False  # OCR already handled in previous stage
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter()
    
    def analyze_document_structure(self, file_path: str, 
                                 previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze document structure and extract logical elements"""
        try:
            # Determine input format
            input_format = self._determine_input_format(file_path)
            
            # Configure for different input types
            if previous_results and previous_results.get('ocr_applied', False):
                # Use OCR text if available
                content = previous_results.get('normalized_text', '')
                result = self._analyze_text_structure(content, file_path)
            else:
                # Process original document
                result = self._analyze_document_with_docling(file_path, input_format)
            
            # Enhance with additional analysis
            enhanced_result = self._enhance_structure_analysis(result)
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Document structure analysis failed for {file_path}: {e}")
            return {
                'success': False,
                'structure': {},
                'elements': [],
                'error': str(e)
            }
    
    def _analyze_document_with_docling(self, file_path: str, input_format: InputFormat) -> Dict[str, Any]:
        """Use docling to analyze document structure"""
        try:
            # Convert document
            conv_result = self.converter.convert(file_path, options=self.pipeline_options)
            
            if not conv_result.document:
                raise ValueError("Failed to convert document")
            
            doc = conv_result.document
            
            # Extract structured information
            structure_data = {
                'success': True,
                'document_info': {
                    'title': getattr(doc, 'title', ''),
                    'page_count': len(doc.pages) if hasattr(doc, 'pages') else 1,
                    'language': getattr(doc, 'language', 'unknown')
                },
                'elements': self._extract_document_elements(doc),
                'hierarchy': self._build_document_hierarchy(doc),
                'tables': self._extract_table_data(doc),
                'metadata': self._extract_document_metadata(doc)
            }
            
            return structure_data
            
        except Exception as e:
            self.logger.error(f"Docling processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_document_elements(self, document) -> List[Dict[str, Any]]:
        """Extract all document elements with classification"""
        elements = []
        
        try:
            # Iterate through document structure
            if hasattr(document, 'body'):
                for element in document.body:
                    element_data = {
                        'type': self._classify_element_type(element),
                        'content': self._extract_element_content(element),
                        'level': self._determine_element_level(element),
                        'position': self._get_element_position(element),
                        'formatting': self._extract_formatting_info(element),
                        'confidence': self._calculate_element_confidence(element)
                    }
                    elements.append(element_data)
            
            # Sort elements by reading order
            elements.sort(key=lambda x: (x['position']['page'], x['position']['y'], x['position']['x']))
            
            return elements
            
        except Exception as e:
            self.logger.warning(f"Element extraction failed: {e}")
            return []
    
    def _classify_element_type(self, element) -> str:
        """Classify document element type"""
        # Map docling element types to our classification
        type_mapping = {
            'title': 'heading_1',
            'heading': 'heading',
            'paragraph': 'paragraph',
            'list': 'list',
            'table': 'table',
            'figure': 'figure',
            'caption': 'caption',
            'footer': 'footer',
            'header': 'header'
        }
        
        element_type = getattr(element, 'type', 'paragraph')
        return type_mapping.get(element_type, 'paragraph')
    
    def _extract_element_content(self, element) -> str:
        """Extract text content from element"""
        try:
            if hasattr(element, 'text'):
                return element.text.strip()
            elif hasattr(element, 'content'):
                return element.content.strip()
            else:
                return str(element).strip()
        except:
            return ""
    
    def _build_document_hierarchy(self, document) -> Dict[str, Any]:
        """Build hierarchical representation of document structure"""
        hierarchy = {
            'type': 'document',
            'children': [],
            'metadata': {}
        }
        
        try:
            current_section = None
            current_subsection = None
            
            if hasattr(document, 'body'):
                for element in document.body:
                    element_type = self._classify_element_type(element)
                    content = self._extract_element_content(element)
                    
                    if element_type == 'heading_1':
                        # New main section
                        current_section = {
                            'type': 'section',
                            'title': content,
                            'level': 1,
                            'children': [],
                            'metadata': {'element_count': 0}
                        }
                        hierarchy['children'].append(current_section)
                        current_subsection = None
                        
                    elif element_type == 'heading' and current_section:
                        # New subsection
                        current_subsection = {
                            'type': 'subsection',
                            'title': content,
                            'level': 2,
                            'children': [],
                            'metadata': {'element_count': 0}
                        }
                        current_section['children'].append(current_subsection)
                        
                    else:
                        # Content element
                        element_data = {
                            'type': element_type,
                            'content': content,
                            'metadata': self._get_element_position(element)
                        }
                        
                        # Add to appropriate parent
                        if current_subsection:
                            current_subsection['children'].append(element_data)
                            current_subsection['metadata']['element_count'] += 1
                        elif current_section:
                            current_section['children'].append(element_data)
                            current_section['metadata']['element_count'] += 1
                        else:
                            hierarchy['children'].append(element_data)
            
            return hierarchy
            
        except Exception as e:
            self.logger.warning(f"Hierarchy building failed: {e}")
            return hierarchy
    
    def _extract_table_data(self, document) -> List[Dict[str, Any]]:
        """Extract structured table information"""
        tables = []
        
        try:
            if hasattr(document, 'tables'):
                for i, table in enumerate(document.tables):
                    table_data = {
                        'table_id': f"table_{i}",
                        'rows': self._process_table_rows(table),
                        'columns': getattr(table, 'num_cols', 0),
                        'row_count': getattr(table, 'num_rows', 0),
                        'caption': self._find_table_caption(table),
                        'position': self._get_element_position(table)
                    }
                    tables.append(table_data)
            
            return tables
            
        except Exception as e:
            self.logger.warning(f"Table extraction failed: {e}")
            return []
    
    def _process_table_rows(self, table) -> List[List[str]]:
        """Process table rows and cells"""
        try:
            rows = []
            if hasattr(table, 'data'):
                for row in table.data:
                    cell_texts = []
                    for cell in row:
                        cell_text = str(cell).strip() if cell else ""
                        cell_texts.append(cell_text)
                    rows.append(cell_texts)
            return rows
        except:
            return []
```

### Structure Validation and Quality Assessment
```python
from typing import Dict, Any, List
import re

class StructureValidator:
    """Validate and assess quality of extracted document structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_structure(self, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation of document structure extraction"""
        try:
            validation_result = {
                'is_valid': True,
                'quality_score': 0,
                'issues': [],
                'recommendations': [],
                'statistics': {}
            }
            
            # Validate hierarchy consistency
            hierarchy_score = self._validate_hierarchy(structure_result.get('hierarchy', {}))
            
            # Validate element completeness
            completeness_score = self._validate_completeness(structure_result.get('elements', []))
            
            # Validate reading order
            order_score = self._validate_reading_order(structure_result.get('elements', []))
            
            # Validate content quality
            content_score = self._validate_content_quality(structure_result.get('elements', []))
            
            # Calculate overall quality score
            validation_result['quality_score'] = (
                hierarchy_score * 0.25 +
                completeness_score * 0.25 +
                order_score * 0.25 +
                content_score * 0.25
            )
            
            validation_result['component_scores'] = {
                'hierarchy': hierarchy_score,
                'completeness': completeness_score,
                'reading_order': order_score,
                'content_quality': content_score
            }
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_improvement_recommendations(
                validation_result['component_scores']
            )
            
            # Mark as invalid if score is too low
            validation_result['is_valid'] = validation_result['quality_score'] >= 70
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Structure validation failed: {e}")
            return {
                'is_valid': False,
                'quality_score': 0,
                'error': str(e)
            }
    
    def _validate_hierarchy(self, hierarchy: Dict[str, Any]) -> float:
        """Validate document hierarchy consistency"""
        if not hierarchy or not hierarchy.get('children'):
            return 50  # Minimum score for flat structure
        
        score = 100
        issues = []
        
        # Check for proper nesting
        for section in hierarchy['children']:
            if section.get('type') == 'section':
                if not section.get('title'):
                    issues.append("Section without title found")
                    score -= 10
                
                # Check subsection consistency
                for subsection in section.get('children', []):
                    if subsection.get('type') == 'subsection':
                        if not subsection.get('title'):
                            issues.append("Subsection without title found")
                            score -= 5
        
        # Penalize excessive nesting or too flat structure
        max_depth = self._calculate_max_depth(hierarchy)
        if max_depth > 5:
            score -= 10  # Too deep
        elif max_depth < 2:
            score -= 5   # Too flat
        
        return max(0, score)
    
    def _validate_completeness(self, elements: List[Dict[str, Any]]) -> float:
        """Validate that all important content is captured"""
        if not elements:
            return 0
        
        # Count different element types
        element_counts = {}
        for element in elements:
            elem_type = element.get('type', 'unknown')
            element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
        
        score = 100
        
        # Expect reasonable distribution of elements
        total_elements = len(elements)
        paragraph_ratio = element_counts.get('paragraph', 0) / total_elements
        
        if paragraph_ratio < 0.3:  # Too few paragraphs
            score -= 20
        elif paragraph_ratio > 0.9:  # Only paragraphs
            score -= 10
        
        # Bonus for diverse content types
        content_diversity = len(element_counts)
        if content_diversity >= 4:
            score += 10
        elif content_diversity <= 1:
            score -= 20
        
        return max(0, score)
    
    def _validate_content_quality(self, elements: List[Dict[str, Any]]) -> float:
        """Validate quality of extracted content"""
        if not elements:
            return 0
        
        score = 100
        total_content_length = 0
        empty_elements = 0
        
        for element in elements:
            content = element.get('content', '')
            if not content.strip():
                empty_elements += 1
            else:
                total_content_length += len(content)
        
        # Penalize excessive empty elements
        empty_ratio = empty_elements / len(elements)
        if empty_ratio > 0.3:
            score -= 30
        
        # Check average content length
        avg_content_length = total_content_length / (len(elements) - empty_elements) if (len(elements) - empty_elements) > 0 else 0
        
        if avg_content_length < 10:  # Very short content
            score -= 20
        elif avg_content_length > 1000:  # Very long content (might be extraction error)
            score -= 10
        
        return max(0, score)
```

### Output Formatting for Chunking Pipeline
```python
from typing import Dict, Any, List

class StructureOutputFormatter:
    """Format structure analysis results for downstream processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_for_chunking(self, structure_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format structure data for optimal chunking strategy"""
        try:
            formatted_output = {
                'document_sections': [],
                'chunking_metadata': {},
                'element_map': {},
                'processing_instructions': {}
            }
            
            # Process hierarchy for section-aware chunking
            hierarchy = structure_result.get('hierarchy', {})
            formatted_output['document_sections'] = self._format_sections_for_chunking(hierarchy)
            
            # Create element mapping for reference
            elements = structure_result.get('elements', [])
            formatted_output['element_map'] = self._create_element_map(elements)
            
            # Generate chunking instructions
            formatted_output['processing_instructions'] = self._generate_chunking_instructions(
                structure_result
            )
            
            # Add metadata for chunking strategy
            formatted_output['chunking_metadata'] = {
                'preferred_strategy': self._recommend_chunking_strategy(structure_result),
                'section_boundaries': self._identify_section_boundaries(hierarchy),
                'special_elements': self._identify_special_elements(elements),
                'document_complexity': self._assess_document_complexity(structure_result)
            }
            
            return formatted_output
            
        except Exception as e:
            self.logger.error(f"Structure output formatting failed: {e}")
            return {
                'document_sections': [],
                'error': str(e)
            }
    
    def _format_sections_for_chunking(self, hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format document sections for chunking pipeline"""
        sections = []
        
        def process_section(section_data, parent_path=""):
            section_path = f"{parent_path}/{section_data.get('title', 'untitled')}" if parent_path else section_data.get('title', 'root')
            
            section_info = {
                'section_path': section_path,
                'section_type': section_data.get('type', 'unknown'),
                'title': section_data.get('title', ''),
                'level': section_data.get('level', 0),
                'content_elements': [],
                'subsections': []
            }
            
            # Process children
            for child in section_data.get('children', []):
                if child.get('type') in ['section', 'subsection']:
                    # Recursive processing for nested sections
                    subsection = process_section(child, section_path)
                    section_info['subsections'].append(subsection)
                else:
                    # Content element
                    section_info['content_elements'].append({
                        'type': child.get('type'),
                        'content': child.get('content', ''),
                        'metadata': child.get('metadata', {})
                    })
            
            return section_info
        
        # Process root level
        for child in hierarchy.get('children', []):
            section = process_section(child)
            sections.append(section)
        
        return sections
    
    def _recommend_chunking_strategy(self, structure_result: Dict[str, Any]) -> str:
        """Recommend optimal chunking strategy based on document structure"""
        elements = structure_result.get('elements', [])
        hierarchy = structure_result.get('hierarchy', {})
        
        # Count structural elements
        heading_count = len([e for e in elements if 'heading' in e.get('type', '')])
        paragraph_count = len([e for e in elements if e.get('type') == 'paragraph'])
        table_count = len([e for e in elements if e.get('type') == 'table'])
        
        # Decision logic
        if heading_count > 5 and paragraph_count > 20:
            return 'section_aware'  # Use section boundaries
        elif table_count > 3:
            return 'table_preserving'  # Keep tables intact
        elif paragraph_count > 50:
            return 'paragraph_based'  # Standard paragraph chunking
        else:
            return 'content_based'  # Simple content chunking
    
    def _identify_section_boundaries(self, hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify natural section boundaries for chunking"""
        boundaries = []
        
        def extract_boundaries(section_data, depth=0):
            if section_data.get('type') in ['section', 'subsection']:
                boundaries.append({
                    'title': section_data.get('title', ''),
                    'level': depth,
                    'type': section_data.get('type'),
                    'element_count': section_data.get('metadata', {}).get('element_count', 0)
                })
                
                # Process children
                for child in section_data.get('children', []):
                    extract_boundaries(child, depth + 1)
        
        # Extract from hierarchy
        for child in hierarchy.get('children', []):
            extract_boundaries(child)
        
        return boundaries
```

## VALIDATION LOOP

### Unit Testing
```python
import pytest
from src.structure_analysis.docling_wrapper import DocumentStructureAnalyzer

class TestDocumentStructureAnalysis:
    def test_structure_extraction_accuracy(self):
        analyzer = DocumentStructureAnalyzer()
        result = analyzer.analyze_document_structure('test_documents/structured_report.pdf')
        
        assert result['success'] == True
        assert len(result['elements']) > 0
        assert 'hierarchy' in result
        assert result['hierarchy']['children']  # Has structure
    
    def test_element_classification(self):
        analyzer = DocumentStructureAnalyzer()
        # Test with document containing known elements
        result = analyzer.analyze_document_structure('test_documents/mixed_content.pdf')
        
        element_types = [e['type'] for e in result['elements']]
        assert 'heading' in element_types
        assert 'paragraph' in element_types
    
    def test_hierarchy_building(self):
        analyzer = DocumentStructureAnalyzer()
        result = analyzer.analyze_document_structure('test_documents/hierarchical_doc.pdf')
        
        hierarchy = result['hierarchy']
        assert len(hierarchy['children']) > 0
        # Check for proper nesting
        sections = [c for c in hierarchy['children'] if c['type'] == 'section']
        assert len(sections) > 0
```

### Integration Testing
- End-to-end structure analysis with various document types
- Validation against manually annotated ground truth
- Integration with upstream (OCR) and downstream (chunking) components

### Performance Testing
- Process complex documents (50+ pages) within 2 minutes
- Memory usage under 1GB for large documents
- Accuracy >95% on business document structures

## ADDITIONAL NOTES

### Security Considerations
- Validate document format and size before processing
- Prevent resource exhaustion with large documents
- Secure handling of sensitive document content
- Access control for structure analysis results

### Performance Optimization
- Caching of structure analysis results
- Incremental processing for similar documents
- Parallel processing of document pages
- Memory-efficient handling of large documents

### Maintenance Requirements
- Regular docling library updates and compatibility testing
- Structure quality monitoring and reporting
- Ground truth validation for continuous accuracy assessment
- Integration monitoring with chunking pipeline performance