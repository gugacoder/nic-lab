# Docling Processing - PRP

## ROLE
**Document Processing Engineer with OCR and AI Document Analysis expertise**

Specialized in Docling framework integration, document format normalization, OCR pipeline optimization, and structured content extraction. Responsible for implementing robust document processing workflows that handle both digital and scanned documents with high-quality text extraction and structural analysis.

## OBJECTIVE
**Unified Document Processing and Structuring Module**

Deliver a production-ready Python module that:
- Processes all document formats (PDF, DOCX, images) through unified Docling interface
- Automatically detects digital vs scanned content for optimal processing paths
- Applies OCR when necessary using Docling's integrated engines
- Extracts structured content (titles, sections, paragraphs, lists, tables, figures)
- Produces canonical JSON/Markdown output with consistent formatting
- Provides confidence scores and quality metrics for downstream validation
- Maintains processing provenance and lineage tracking

## MOTIVATION
**Consistent Document Understanding Foundation**

This module standardizes document processing across all input formats, ensuring consistent text extraction quality and structural understanding. By centralizing document processing through Docling, the system achieves deterministic results, eliminates format-specific processing branches, and provides the structured foundation required for effective chunking and embedding generation.

## CONTEXT
**Docling-Centralized Processing Architecture**

- **Input Formats**: PDF (digital/scanned), DOCX, TXT, MD, JPG, PNG
- **Processing Engine**: Docling framework with integrated OCR capabilities
- **Output Format**: Structured JSON with optional Markdown export
- **Quality Requirements**: Confidence scoring, provenance tracking, deterministic processing
- **Integration Pattern**: Modular Python module called from pipeline orchestrator
- **Performance Goals**: Process documents efficiently while maintaining accuracy

## IMPLEMENTATION BLUEPRINT
**Comprehensive Docling Processing Module**

### Architecture Overview
```python
# Module Structure: modules/docling_processing.py
class DoclingProcessor:
    """Unified document processing through Docling framework"""
    
    def __init__(self, config: ProcessingConfig)
    def process_document(self, file_path: str, content: bytes) -> ProcessedDocument
    def detect_document_type(self, content: bytes, file_extension: str) -> DocumentType
    def extract_structure(self, docling_result: Any) -> StructuredContent
    def apply_quality_gates(self, processed_doc: ProcessedDocument) -> QualityAssessment
    def export_canonical_format(self, structured_content: StructuredContent) -> Dict[str, Any]
```

### Code Structure
**File Organization**: `modules/docling_processing.py`
```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
from enum import Enum

class DocumentType(Enum):
    """Document type classification"""
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    WORD_DOCUMENT = "word_document"
    TEXT_DOCUMENT = "text_document"
    IMAGE_DOCUMENT = "image_document"
    MARKDOWN = "markdown"

class ProcessingQuality(Enum):
    """Processing quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    ocr_engine: str = "easyocr"  # Docling OCR engine
    confidence_threshold: float = 0.8
    max_file_size_mb: int = 100
    enable_table_extraction: bool = True
    enable_figure_extraction: bool = True
    output_format: str = "json"  # json, markdown, both
    quality_gates_enabled: bool = True

@dataclass
class StructuredContent:
    """Structured document content with hierarchy"""
    title: Optional[str] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAssessment:
    """Document processing quality metrics"""
    overall_quality: ProcessingQuality
    confidence_score: float
    text_extraction_confidence: float
    structure_detection_confidence: float
    ocr_applied: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ProcessedDocument:
    """Complete processed document with all extracted information"""
    file_path: str
    file_hash: str
    document_type: DocumentType
    structured_content: StructuredContent
    quality_assessment: QualityAssessment
    processing_metadata: Dict[str, Any]
    canonical_output: Dict[str, Any]
    processing_timestamp: datetime
    docling_version: str

class DoclingProcessor:
    """Production-ready document processor using Docling framework"""
    
    SUPPORTED_FORMATS = {
        '.pdf': InputFormat.PDF,
        '.docx': InputFormat.DOCX,
        '.txt': InputFormat.TEXT,
        '.md': InputFormat.TEXT,
        '.jpg': InputFormat.IMAGE,
        '.jpeg': InputFormat.IMAGE,
        '.png': InputFormat.IMAGE
    }
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docling converter with optimized pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = config.enable_table_extraction
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.IMAGE],
            pdf_backend=DoclingParseDocumentBackend,
            pipeline_options=pipeline_options
        )
        
        self.logger.info(f"Docling processor initialized with config: {config}")
    
    def process_document(self, file_path: str, content: bytes, 
                        metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """Process document through Docling with comprehensive error handling"""
        
        try:
            # Validate input
            if len(content) > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File size exceeds limit: {len(content) / (1024*1024):.1f}MB")
            
            # Generate content hash for idempotency
            file_hash = hashlib.sha256(content).hexdigest()
            
            # Detect document type
            file_extension = Path(file_path).suffix.lower()
            doc_type = self.detect_document_type(content, file_extension)
            
            self.logger.info(f"Processing {doc_type.value} document: {file_path}")
            
            # Process through Docling
            docling_result = self._process_with_docling(content, file_path, doc_type)
            
            # Extract structured content
            structured_content = self.extract_structure(docling_result)
            
            # Generate canonical output
            canonical_output = self.export_canonical_format(structured_content)
            
            # Create processed document
            processed_doc = ProcessedDocument(
                file_path=file_path,
                file_hash=file_hash,
                document_type=doc_type,
                structured_content=structured_content,
                quality_assessment=QualityAssessment(
                    overall_quality=ProcessingQuality.HIGH,
                    confidence_score=0.0,  # Will be calculated
                    text_extraction_confidence=0.0,
                    structure_detection_confidence=0.0,
                    ocr_applied=False  # Will be determined
                ),
                processing_metadata={
                    'docling_backend': 'DoclingParseDocumentBackend',
                    'input_metadata': metadata or {},
                    'file_size_bytes': len(content)
                },
                canonical_output=canonical_output,
                processing_timestamp=datetime.utcnow(),
                docling_version=self._get_docling_version()
            )
            
            # Apply quality gates
            if self.config.quality_gates_enabled:
                processed_doc.quality_assessment = self.apply_quality_gates(processed_doc)
            
            self.logger.info(f"Successfully processed document: {file_path}")
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Failed to process document {file_path}: {e}")
            
            # Return failed processing result for pipeline continuity
            return ProcessedDocument(
                file_path=file_path,
                file_hash=hashlib.sha256(content).hexdigest(),
                document_type=DocumentType.DIGITAL_PDF,  # Default
                structured_content=StructuredContent(),
                quality_assessment=QualityAssessment(
                    overall_quality=ProcessingQuality.FAILED,
                    confidence_score=0.0,
                    text_extraction_confidence=0.0,
                    structure_detection_confidence=0.0,
                    ocr_applied=False,
                    issues=[str(e)]
                ),
                processing_metadata={'error': str(e)},
                canonical_output={},
                processing_timestamp=datetime.utcnow(),
                docling_version=self._get_docling_version()
            )
    
    def detect_document_type(self, content: bytes, file_extension: str) -> DocumentType:
        """Intelligent document type detection"""
        
        if file_extension in ['.jpg', '.jpeg', '.png']:
            return DocumentType.IMAGE_DOCUMENT
        elif file_extension == '.docx':
            return DocumentType.WORD_DOCUMENT
        elif file_extension in ['.txt', '.md']:
            return DocumentType.TEXT_DOCUMENT if file_extension == '.txt' else DocumentType.MARKDOWN
        elif file_extension == '.pdf':
            # For PDFs, we'll let Docling determine if OCR is needed
            # This is a simplified detection - Docling will handle the complexity
            return DocumentType.DIGITAL_PDF  # Will be refined during processing
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _process_with_docling(self, content: bytes, file_path: str, doc_type: DocumentType) -> Any:
        """Core Docling processing with error handling"""
        
        try:
            # Create temporary file for Docling processing
            temp_path = f"/tmp/{Path(file_path).name}"
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            # Process through Docling converter
            result = self.converter.convert(temp_path)
            
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Docling processing failed for {file_path}: {e}")
            raise
    
    def extract_structure(self, docling_result: Any) -> StructuredContent:
        """Extract structured content from Docling result"""
        
        structured_content = StructuredContent()
        
        try:
            # Access the document from Docling result
            document = docling_result.document
            
            # Extract title (first heading or from metadata)
            structured_content.title = self._extract_title(document)
            
            # Extract sections and hierarchy
            structured_content.sections = self._extract_sections(document)
            
            # Extract paragraphs with metadata
            structured_content.paragraphs = self._extract_paragraphs(document)
            
            # Extract lists
            structured_content.lists = self._extract_lists(document)
            
            # Extract tables if enabled
            if self.config.enable_table_extraction:
                structured_content.tables = self._extract_tables(document)
            
            # Extract figures if enabled
            if self.config.enable_figure_extraction:
                structured_content.figures = self._extract_figures(document)
            
            # Add processing metadata
            structured_content.metadata = {
                'page_count': len(document.pages) if hasattr(document, 'pages') else 0,
                'element_count': len(document.texts) if hasattr(document, 'texts') else 0,
                'has_tables': len(structured_content.tables) > 0,
                'has_figures': len(structured_content.figures) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Structure extraction failed: {e}")
            # Return minimal structure to prevent pipeline failure
            structured_content.metadata = {'extraction_error': str(e)}
        
        return structured_content
    
    def _extract_title(self, document: Any) -> Optional[str]:
        """Extract document title from Docling document"""
        # Implementation depends on Docling's document structure
        # This is a placeholder for the actual Docling API calls
        try:
            # Look for title in document metadata or first heading
            if hasattr(document, 'meta') and document.meta.title:
                return document.meta.title
            
            # Find first heading element
            for element in document.texts:
                if hasattr(element, 'label') and 'title' in element.label.lower():
                    return element.text
            
            return None
            
        except Exception:
            return None
    
    def _extract_sections(self, document: Any) -> List[Dict[str, Any]]:
        """Extract document sections with hierarchy"""
        sections = []
        
        try:
            current_section = None
            
            for element in document.texts:
                if hasattr(element, 'label') and 'section' in element.label.lower():
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'title': element.text,
                        'level': self._determine_heading_level(element),
                        'content': [],
                        'page_number': getattr(element, 'page', 0),
                        'position': getattr(element, 'bbox', None)
                    }
                elif current_section and hasattr(element, 'text'):
                    current_section['content'].append({
                        'text': element.text,
                        'type': getattr(element, 'label', 'paragraph')
                    })
            
            # Add final section
            if current_section:
                sections.append(current_section)
                
        except Exception as e:
            self.logger.warning(f"Section extraction warning: {e}")
        
        return sections
    
    def _extract_paragraphs(self, document: Any) -> List[Dict[str, Any]]:
        """Extract paragraphs with positioning metadata"""
        paragraphs = []
        
        try:
            for element in document.texts:
                if hasattr(element, 'label') and 'paragraph' in element.label.lower():
                    paragraphs.append({
                        'text': element.text,
                        'page_number': getattr(element, 'page', 0),
                        'position': getattr(element, 'bbox', None),
                        'confidence': getattr(element, 'confidence', 1.0)
                    })
                    
        except Exception as e:
            self.logger.warning(f"Paragraph extraction warning: {e}")
        
        return paragraphs
    
    def _extract_tables(self, document: Any) -> List[Dict[str, Any]]:
        """Extract tables with structure preservation"""
        tables = []
        
        try:
            if hasattr(document, 'tables'):
                for table in document.tables:
                    tables.append({
                        'data': self._serialize_table_data(table),
                        'page_number': getattr(table, 'page', 0),
                        'position': getattr(table, 'bbox', None),
                        'row_count': len(table.data) if hasattr(table, 'data') else 0,
                        'column_count': len(table.data[0]) if hasattr(table, 'data') and table.data else 0
                    })
                    
        except Exception as e:
            self.logger.warning(f"Table extraction warning: {e}")
        
        return tables
    
    def _extract_figures(self, document: Any) -> List[Dict[str, Any]]:
        """Extract figures and images"""
        figures = []
        
        try:
            if hasattr(document, 'pictures'):
                for figure in document.pictures:
                    figures.append({
                        'caption': getattr(figure, 'caption', ''),
                        'page_number': getattr(figure, 'page', 0),
                        'position': getattr(figure, 'bbox', None),
                        'image_data': getattr(figure, 'image', None)  # Base64 or reference
                    })
                    
        except Exception as e:
            self.logger.warning(f"Figure extraction warning: {e}")
        
        return figures
    
    def apply_quality_gates(self, processed_doc: ProcessedDocument) -> QualityAssessment:
        """Apply quality assessment and confidence scoring"""
        
        assessment = QualityAssessment(
            overall_quality=ProcessingQuality.HIGH,
            confidence_score=1.0,
            text_extraction_confidence=1.0,
            structure_detection_confidence=1.0,
            ocr_applied=self._was_ocr_applied(processed_doc)
        )
        
        # Calculate confidence scores
        text_length = sum(len(p['text']) for p in processed_doc.structured_content.paragraphs)
        
        if text_length < 100:
            assessment.warnings.append("Low text content extracted")
            assessment.text_extraction_confidence = 0.5
        
        # Check structure detection quality
        structure_elements = (
            len(processed_doc.structured_content.sections) +
            len(processed_doc.structured_content.paragraphs) +
            len(processed_doc.structured_content.lists)
        )
        
        if structure_elements < 5:
            assessment.warnings.append("Limited document structure detected")
            assessment.structure_detection_confidence = 0.6
        
        # Calculate overall confidence
        assessment.confidence_score = (
            assessment.text_extraction_confidence * 0.6 +
            assessment.structure_detection_confidence * 0.4
        )
        
        # Determine overall quality
        if assessment.confidence_score >= 0.8:
            assessment.overall_quality = ProcessingQuality.HIGH
        elif assessment.confidence_score >= 0.6:
            assessment.overall_quality = ProcessingQuality.MEDIUM
        else:
            assessment.overall_quality = ProcessingQuality.LOW
        
        return assessment
    
    def export_canonical_format(self, structured_content: StructuredContent) -> Dict[str, Any]:
        """Export structured content to canonical JSON format"""
        
        canonical = {
            'document_structure': {
                'title': structured_content.title,
                'sections': structured_content.sections,
                'paragraphs': structured_content.paragraphs,
                'lists': structured_content.lists,
                'tables': structured_content.tables,
                'figures': structured_content.figures
            },
            'metadata': structured_content.metadata,
            'format_version': '1.0',
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        # Optional Markdown export
        if self.config.output_format in ['markdown', 'both']:
            canonical['markdown'] = self._convert_to_markdown(structured_content)
        
        return canonical
    
    def _convert_to_markdown(self, structured_content: StructuredContent) -> str:
        """Convert structured content to Markdown format"""
        md_lines = []
        
        # Add title
        if structured_content.title:
            md_lines.append(f"# {structured_content.title}")
            md_lines.append("")
        
        # Add sections
        for section in structured_content.sections:
            level = section.get('level', 1)
            md_lines.append(f"{'#' * level} {section['title']}")
            md_lines.append("")
            
            for content_item in section.get('content', []):
                md_lines.append(content_item['text'])
                md_lines.append("")
        
        # Add standalone paragraphs
        for paragraph in structured_content.paragraphs:
            md_lines.append(paragraph['text'])
            md_lines.append("")
        
        return "\n".join(md_lines)
    
    def _determine_heading_level(self, element: Any) -> int:
        """Determine heading level from Docling element"""
        # Implementation depends on Docling's element structure
        return getattr(element, 'level', 1)
    
    def _serialize_table_data(self, table: Any) -> List[List[str]]:
        """Serialize table data for JSON storage"""
        # Implementation depends on Docling's table structure
        if hasattr(table, 'data'):
            return table.data
        return []
    
    def _was_ocr_applied(self, processed_doc: ProcessedDocument) -> bool:
        """Determine if OCR was applied during processing"""
        # This would be determined from Docling's processing metadata
        return processed_doc.document_type in [DocumentType.SCANNED_PDF, DocumentType.IMAGE_DOCUMENT]
    
    def _get_docling_version(self) -> str:
        """Get Docling framework version"""
        try:
            import docling
            return docling.__version__
        except:
            return "unknown"

def create_docling_processor(config_dict: Dict[str, Any]) -> DoclingProcessor:
    """Factory function for Docling processor creation"""
    config = ProcessingConfig(**config_dict)
    return DoclingProcessor(config)
```

### Error Handling
**Comprehensive Quality Control and Error Management**
```python
class DoclingProcessingError(Exception):
    """Base exception for Docling processing errors"""
    pass

class DocumentFormatError(DoclingProcessingError):
    """Unsupported or corrupted document format"""
    pass

class OCRProcessingError(DoclingProcessingError):
    """OCR processing failures"""
    pass

class StructureExtractionError(DoclingProcessingError):
    """Document structure extraction failures"""
    pass

# Quality gate implementations
def validate_processing_quality(processed_doc: ProcessedDocument) -> bool:
    """Validate processing meets quality thresholds"""
    if processed_doc.quality_assessment.overall_quality == ProcessingQuality.FAILED:
        return False
    
    if processed_doc.quality_assessment.confidence_score < 0.5:
        return False
    
    # Ensure minimum content was extracted
    total_text = sum(len(p['text']) for p in processed_doc.structured_content.paragraphs)
    if total_text < 50:  # Minimum 50 characters
        return False
    
    return True
```

## VALIDATION LOOP
**Comprehensive Testing Strategy**

### Unit Testing
```python
# tests/test_docling_processing.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from modules.docling_processing import DoclingProcessor, ProcessingConfig, DocumentType

class TestDoclingProcessor:
    
    @pytest.fixture
    def default_config(self):
        return ProcessingConfig(
            ocr_engine="easyocr",
            confidence_threshold=0.8,
            quality_gates_enabled=True
        )
    
    @pytest.fixture
    def sample_pdf_content(self):
        # Mock PDF content
        return b'%PDF-1.4 mock content...'
    
    def test_document_type_detection(self, default_config):
        """Test document type detection logic"""
        processor = DoclingProcessor(default_config)
        
        assert processor.detect_document_type(b'content', '.pdf') == DocumentType.DIGITAL_PDF
        assert processor.detect_document_type(b'content', '.docx') == DocumentType.WORD_DOCUMENT
        assert processor.detect_document_type(b'content', '.jpg') == DocumentType.IMAGE_DOCUMENT
    
    @patch('modules.docling_processing.DocumentConverter')
    def test_successful_processing(self, mock_converter, default_config, sample_pdf_content):
        """Test successful document processing flow"""
        # Mock Docling converter result
        mock_result = MagicMock()
        mock_document = MagicMock()
        mock_document.texts = []
        mock_document.pages = [MagicMock()]
        mock_result.document = mock_document
        
        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance
        
        processor = DoclingProcessor(default_config)
        result = processor.process_document('test.pdf', sample_pdf_content)
        
        assert result.file_path == 'test.pdf'
        assert result.document_type == DocumentType.DIGITAL_PDF
        assert result.quality_assessment is not None
    
    def test_quality_assessment(self, default_config):
        """Test quality assessment logic"""
        processor = DoclingProcessor(default_config)
        
        # Create mock processed document with minimal content
        mock_doc = MagicMock()
        mock_doc.structured_content.paragraphs = [{'text': 'short'}]
        mock_doc.structured_content.sections = []
        mock_doc.structured_content.lists = []
        
        assessment = processor.apply_quality_gates(mock_doc)
        
        assert assessment.confidence_score < 0.8  # Should be low due to minimal content
        assert len(assessment.warnings) > 0
```

### Integration Testing
```python
# tests/integration/test_docling_live.py
@pytest.mark.integration
def test_real_document_processing():
    """Integration test with real documents"""
    config = ProcessingConfig(quality_gates_enabled=True)
    processor = DoclingProcessor(config)
    
    # Test with sample PDF
    with open('tests/fixtures/sample.pdf', 'rb') as f:
        content = f.read()
    
    result = processor.process_document('sample.pdf', content)
    
    assert result.quality_assessment.overall_quality != ProcessingQuality.FAILED
    assert len(result.structured_content.paragraphs) > 0
    assert result.canonical_output['document_structure'] is not None
```

### Performance Testing
- **Processing Speed**: Target <30 seconds for typical documents (10-50 pages)
- **Memory Usage**: Monitor memory consumption for large documents
- **OCR Accuracy**: Validate OCR quality on scanned documents
- **Structure Detection**: Test hierarchical content extraction accuracy

## ADDITIONAL NOTES
**Security, Performance & Maintenance**

### Security Considerations
- **Input Validation**: Strict file format validation and size limits
- **Sandboxed Processing**: Isolate document processing to prevent malicious content execution
- **Data Privacy**: Ensure sensitive document content is not logged or cached inappropriately
- **Temporary File Management**: Secure cleanup of temporary processing files

### Performance Optimization
- **Processing Pipeline**: Optimize Docling configuration for speed vs accuracy trade-offs
- **Memory Management**: Stream processing for large documents to minimize memory usage
- **Parallel Processing**: Support concurrent document processing with resource limits
- **Caching Strategy**: Cache processing results for identical documents (based on hash)

### Maintenance Requirements
- **Docling Updates**: Monitor Docling framework updates and compatibility
- **OCR Engine Maintenance**: Regular evaluation of OCR accuracy and performance
- **Quality Metrics**: Continuous monitoring of processing quality and confidence scores
- **Error Analytics**: Track and analyze processing failures for system improvement